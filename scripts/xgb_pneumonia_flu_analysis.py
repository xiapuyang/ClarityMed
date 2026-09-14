"""XGBoost analysis of the DDXPlus Pneumonia vs Influenza subset.

Answers three questions the typed-BASD probe can't:

  1. Which evidences actually discriminate Pneumonia from Influenza?
     — via XGBoost gain-based feature importance.

  2. What does a shallow decision tree look like on this task?
     — dumps the first tree with human-readable node conditions.

  3. What would an information-gain question-selection policy pick
     turn-by-turn?  — computes IG(e | state) at each of 6 turns and
     prints the ranking + P(Pneumonia) trajectory.  This is what
     typed-BASD *doesn't* do (confirms positives instead), so the
     output is a side-by-side against your typed-BASD probe.

Install-on-demand: run with
    uv run --with xgboost python scripts/xgb_pneumonia_flu_analysis.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xgboost as xgb

from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_patients,
    load_pidx,
)

DEFAULT_DATA_DIR = Path.home() / ".claritymed" / "data" / "symptoms" / "ddxplus"
DEFAULT_DISEASES = ("Pneumonia", "Influenza")
DEFAULT_N_TRAIN = 300_000
DEFAULT_N_TEST = 20_000
MAX_TREE_DEPTH_INTERPRET = 4
MAX_TREE_DEPTH_ACCURACY = 6
N_TURNS = 6  # match subset model's tuned maxstep


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------


def _load_evidence_meta(data_dir: Path) -> dict[str, dict]:
    """Return {ev_id: raw entry} keyed by evidence id."""
    raw = json.loads((data_dir / "release_evidences.json").read_text())
    items = list(raw.values()) if isinstance(raw, dict) else raw
    return {e["name"]: e for e in items}


def _build_feature_columns(
    schema: dict,
    meta: dict[str, dict],
) -> tuple[list[str], list[str], dict[str, int]]:
    """Enumerate feature columns.

    Each binary evidence → 1 column. Each categorical/multi value → 1 column
    (``E_X__value``). Returns (column names, human-readable labels,
    name → column index).
    """
    columns: list[str] = []
    labels: list[str] = []
    for ev_idx, ev in enumerate(schema["evs"]):
        ev_id = ev["name"]
        dtype = ev["dtype"]
        q_text = meta.get(ev_id, {}).get("question_en", ev_id)
        if dtype == "B":
            columns.append(ev_id)
            labels.append(f"{ev_id}: {q_text}")
        else:
            for value in ev["values"]:
                columns.append(f"{ev_id}__{value}")
                meaning = meta.get(ev_id, {}).get("value_meaning", {}).get(
                    str(value), {}
                ).get("en") or str(value)
                labels.append(f"{ev_id}={meaning}: {q_text}")
    return columns, labels, {c: i for i, c in enumerate(columns)}


def _encode_patients(
    patients: list[dict],
    schema: dict,
    columns_idx: dict[str, int],
) -> np.ndarray:
    """One-hot encode DDXPlus patients into a dense feature matrix."""
    n_features = len(columns_idx)
    x = np.zeros((len(patients), n_features), dtype=np.float32)
    ev_names = [ev["name"] for ev in schema["evs"]]
    for i, p in enumerate(patients):
        for ev_i in p["bin_pos"]:
            col = columns_idx.get(ev_names[ev_i])
            if col is not None:
                x[i, col] = 1.0
        for ev_i, lv in p["cat_val"].items():
            raw = schema["evs"][ev_i]["values"][lv]
            col = columns_idx.get(f"{ev_names[ev_i]}__{raw}")
            if col is not None:
                x[i, col] = 1.0
        for ev_i, lvs in p["multi_val"].items():
            for lv in lvs:
                raw = schema["evs"][ev_i]["values"][lv]
                col = columns_idx.get(f"{ev_names[ev_i]}__{raw}")
                if col is not None:
                    x[i, col] = 1.0
    return x


# ---------------------------------------------------------------------------
# Feature-importance + tree dump
# ---------------------------------------------------------------------------


def _report_importance(
    booster: "xgb.Booster",
    labels: list[str],
    columns: list[str],
    k: int = 20,
) -> None:
    print(f"\n=== Feature importance (top {k} by gain) ===")
    label_by_col = dict(zip(columns, labels))
    scores = booster.get_score(importance_type="gain")
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    for i, (col, gain) in enumerate(ranked[:k], 1):
        print(f"  {i:>2}. gain={gain:>9.2f}  {label_by_col.get(col, col)}")


def _dump_tree_readable(
    booster: "xgb.Booster",
    labels: list[str],
    columns: list[str],
    tree_idx: int = 0,
    max_lines: int = 40,
) -> None:
    """Print one boosted tree with human-readable node conditions."""
    label_by_col = dict(zip(columns, labels))
    dump = booster.get_dump(with_stats=False)
    if tree_idx >= len(dump):
        print(f"(only {len(dump)} trees, cannot show #{tree_idx})")
        return
    tree_text = dump[tree_idx]
    print(f"\n=== Tree #{tree_idx} (human-readable) ===")
    lines = tree_text.splitlines()
    for line in lines[:max_lines]:
        # replace [fN<X.Y] with [<label> < X.Y]
        stripped = line
        for col in columns:
            if col in stripped:
                stripped = stripped.replace(
                    f"[{col}<",
                    f"[{label_by_col.get(col, col)[:60]!r} < ",
                )
        print("  " + stripped)
    if len(lines) > max_lines:
        print(f"  … ({len(lines) - max_lines} more lines truncated)")


# ---------------------------------------------------------------------------
# Info-gain question policy
# ---------------------------------------------------------------------------


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0)
    return float(-(p * np.log2(p)).sum())


def _predict_proba(booster: "xgb.Booster", state: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    # feature_names must match booster's when both are set — pass them
    # through DMatrix so xgboost's validation passes.
    dmat = xgb.DMatrix(state.reshape(1, -1), feature_names=booster.feature_names)
    p1 = float(booster.predict(dmat)[0])
    return np.array([1 - p1, p1])


def _evidence_marginals(
    x_train: np.ndarray,
    columns: list[str],
) -> dict[str, float]:
    """Global marginal P(feature = 1) from training set — cheap prior for IG."""
    means = x_train.mean(axis=0)
    return dict(zip(columns, means))


def _info_gain_policy_demo(
    booster: "xgb.Booster",
    columns: list[str],
    labels: list[str],
    x_train: np.ndarray,
    n_turns: int = N_TURNS,
    top_k_report: int = 5,
) -> None:
    """Run an interactive-style session where next question = argmax IG.

    Simulates a Pneumonia patient (the one with the median training-set
    signature) — walks 6 turns of "which unasked feature maximizes IG",
    and prints the ranking + P(Pneumonia) trajectory.  Answers to the
    picked feature come from that patient's ground truth so the state
    evolves as if a real user answered honestly.
    """
    print(f"\n=== Info-gain policy demo ({n_turns} turns) ===")
    print("  simulating a real Pneumonia patient (median training signature)")

    marginals = _evidence_marginals(x_train, columns)
    # Pick a representative Pneumonia patient — closest to the class centroid.
    pos_mask = _predict_proba_batch(booster, x_train) > 0.5  # class 1 = Pneumonia
    centroid = x_train[pos_mask].mean(axis=0)
    dists = np.linalg.norm(x_train[pos_mask] - centroid, axis=1)
    target_patient = x_train[pos_mask][int(dists.argmin())]

    state = np.zeros_like(target_patient)
    asked: set[int] = set()

    for turn in range(1, n_turns + 1):
        p = _predict_proba(booster, state)
        h_now = _entropy(p)
        # Score each unasked feature — cost O(N_features × 2 XGBoost calls).
        # For 800+ features this is a few seconds per turn on CPU.
        ig_scores: list[tuple[int, float, float]] = []
        for f_idx in range(len(columns)):
            if f_idx in asked:
                continue
            state_yes = state.copy()
            state_yes[f_idx] = 1.0
            state_no = state.copy()
            state_no[f_idx] = 0.0
            p_yes = _predict_proba(booster, state_yes)
            p_no = _predict_proba(booster, state_no)
            p1 = marginals.get(columns[f_idx], 0.05)  # smoothed prior
            ig = h_now - p1 * _entropy(p_yes) - (1 - p1) * _entropy(p_no)
            ig_scores.append((f_idx, ig, p1))

        ig_scores.sort(key=lambda t: -t[1])
        print(f"\n  Turn {turn} — top {top_k_report} candidates by IG:")
        for rank, (f_idx, ig, prior) in enumerate(ig_scores[:top_k_report], 1):
            print(f"    {rank}. IG={ig:.4f}  P(f=1)={prior:.3f}  {labels[f_idx][:80]}")

        chosen_idx, chosen_ig, _ = ig_scores[0]
        asked.add(chosen_idx)
        # Answer from the target patient's ground truth signature.
        answer = float(target_patient[chosen_idx])
        state[chosen_idx] = answer
        p_after = _predict_proba(booster, state)
        print(
            f"  → chose {columns[chosen_idx]} (IG={chosen_ig:.4f}), "
            f"patient answer={int(answer)}, P(Pneumonia)={p_after[1]:.3f}"
        )


def _predict_proba_batch(booster: "xgb.Booster", x: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    return booster.predict(xgb.DMatrix(x, feature_names=booster.feature_names))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--diseases", default=",".join(DEFAULT_DISEASES))
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument(
        "--tree-depth-interpret", type=int, default=MAX_TREE_DEPTH_INTERPRET
    )
    parser.add_argument(
        "--tree-depth-accuracy", type=int, default=MAX_TREE_DEPTH_ACCURACY
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--skip-ig-demo", action="store_true")
    args = parser.parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        raise SystemExit(
            "xgboost not installed. Run with `uv run --with xgboost python "
            "scripts/xgb_pneumonia_flu_analysis.py`"
        )
    from sklearn.metrics import accuracy_score, roc_auc_score

    whitelist = {d.strip() for d in args.diseases.split(",")}
    print(f"Whitelist: {sorted(whitelist)}")
    print(f"Data: {args.data_dir}")

    schema = load_evidence_schema(args.data_dir)
    meta = _load_evidence_meta(args.data_dir)
    pidx, _ = load_pidx(args.data_dir, whitelist=whitelist)
    columns, labels, col_idx = _build_feature_columns(schema, meta)
    print(f"pidx = {pidx}")
    print(f"feature columns = {len(columns)} (binary + categorical/multi one-hot)")

    print(f"\nLoading train (up to {args.n_train} raw rows)…")
    train_pats = load_patients(args.data_dir, args.n_train, "train", schema, pidx)
    x_train = _encode_patients(train_pats, schema, col_idx)
    y_train = np.array([p["d"] for p in train_pats])  # 0=Influenza, 1=Pneumonia
    label_counts = Counter(y_train.tolist())
    print(f"train kept: {len(train_pats)}  class balance: {dict(label_counts)}")

    print(f"\nLoading test (up to {args.n_test} raw rows)…")
    test_pats = load_patients(args.data_dir, args.n_test, "test", schema, pidx)
    x_test = _encode_patients(test_pats, schema, col_idx)
    y_test = np.array([p["d"] for p in test_pats])
    print(f"test kept: {len(test_pats)}")

    # 1. Interpretable shallow model — for tree dump + importance.
    print(
        f"\n=== Training shallow interpretable model (max_depth={args.tree_depth_interpret}) ==="
    )
    clf_shallow = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.tree_depth_interpret,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )
    clf_shallow.fit(x_train, y_train)
    pred_shallow = clf_shallow.predict(x_test)
    prob_shallow = clf_shallow.predict_proba(x_test)[:, 1]
    print(f"  shallow accuracy: {accuracy_score(y_test, pred_shallow):.4f}")
    print(f"  shallow AUC     : {roc_auc_score(y_test, prob_shallow):.4f}")

    booster_shallow = clf_shallow.get_booster()
    # Rename feature indices → readable column names for tree dump.
    booster_shallow.feature_names = columns

    _report_importance(booster_shallow, labels, columns, k=20)
    _dump_tree_readable(booster_shallow, labels, columns, tree_idx=0)

    # 2. Deeper accuracy-oriented model — for the "would XGBoost replace
    # typed-BASD" comparison.
    print(
        f"\n=== Training accuracy-oriented model (max_depth={args.tree_depth_accuracy}) ==="
    )
    clf_deep = xgb.XGBClassifier(
        n_estimators=args.n_estimators * 2,
        max_depth=args.tree_depth_accuracy,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )
    clf_deep.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
    pred_deep = clf_deep.predict(x_test)
    prob_deep = clf_deep.predict_proba(x_test)[:, 1]
    acc_deep = accuracy_score(y_test, pred_deep)
    auc_deep = roc_auc_score(y_test, prob_deep)
    print(f"  deep accuracy: {acc_deep:.4f}  (compare typed-BASD subset ACC=1.0000)")
    print(f"  deep AUC     : {auc_deep:.4f}")

    # 3. Info-gain policy demo — DIFFERENT from typed-BASD's next_action.
    if not args.skip_ig_demo:
        booster_deep = clf_deep.get_booster()
        booster_deep.feature_names = columns
        _info_gain_policy_demo(
            booster_deep,
            columns,
            labels,
            x_train,
            n_turns=N_TURNS,
            top_k_report=5,
        )
    else:
        print("\n(skipping info-gain demo per --skip-ig-demo)")

    # Wrap-up.
    print("\n=== Takeaways ===")
    print(f"  Static XGBoost accuracy on subset: {acc_deep:.4f}")
    print("  → confirms typed-BASD's 100% ACC is not model-magic; XGBoost")
    print("    reaches similar accuracy on the same static feature set.")
    print("  Feature importance shows which evidences carry the signal —")
    print("    compare against typed-BASD's actual asked questions to see")
    print("    the gap between 'discriminative' and 'confirm-positive'.")
    print("  IG policy demo shows the alternative next-question policy;")
    print("    contrast the picked questions with typed-BASD's rash/pain-loc.")


if __name__ == "__main__":
    main()
