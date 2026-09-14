"""Subset-parametric quality eval for XGBoost symptom-prediction models.

Handles both model layouts transparently — auto-detected from the
classifier's ``classes_`` count:

* **49-class + projection** (Phase C, PoC): the model outputs a raw
  49-class posterior; the script projects to ``{target_1, ..., target_N,
  Other}`` at eval time to score the subset-focused metrics.
* **N+1-class native** (v3): the model was trained with :func:`train
  --targets` and already outputs an ``(N+1,)`` posterior; no projection
  needed. Test patient labels are relabelled to subset space at load
  time so ground truth aligns with the classifier's own class layout.

Subset is fully parameterized via ``--targets "A,B,..."`` — every layer
(relabel, static eval, interactive eval, metric bucketing, output tables)
takes the same list. Swapping the target set needs a CLI change, not a
code change. This mirrors what the v3 config layer does: push the
"which conditions are targets" decision to a single list at the edge of
the system.

Two eval regimes on the SAME 49-class test set:

  1) Static eval — encode each test patient's FULL evidence signature and
     hand it to the classifier at once (no question loop). Measures the
     ceiling of the trained model's target-vs-Other calibration.

  2) Interactive eval — run the standard IG question loop, cap at
     ``--maxstep`` questions, stop when ``max(P) > --stop-thres``, then
     project the final posterior into subset + Other space. Measures what
     the model actually delivers to a user under the current policy.

Metrics (per target class, plus 3-way aggregate):

  * PR-AUC — average precision score over all thresholds.
  * Recall @ Precision=0.70 — how many true positives we catch when we
    keep the false-alarm rate reasonable.
  * Precision @ Recall=0.90 — how many alarms are true when we don't
    miss ≥90% of positives.
  * ECE — expected calibration error, 10 equal-mass buckets.

Then a decision-support summary at the bottom: if static-eval PR-AUC is
already above ``--pr-auc-floor`` (default 0.70) for every target class,
the classifier itself is good enough and the fix is purely policy + UI.
Otherwise, retraining as native N+1-class is worthwhile.

Usage::

    uv run python scripts/eval_xgb_subset_metrics.py \\
        --weights ~/.claritymed/models/symptoms/ddxplus/run/xgb_49class_poc_20260708/weights.pkl \\
        --targets "Pneumonia,Influenza" \\
        --n-test 20000 --maxstep 6 --stop-thres 0.90
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
from typing import Sequence

import numpy as np

from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_patients,
    load_pidx,
)
from claritymed.ingest.symptoms.typed_basd import TypedEnv, seed_everything
from claritymed.ingest.symptoms.xgb.algorithm import XgbAgent
from claritymed.ingest.symptoms.xgb.encoding import (
    encode_patient_batch,
    feature_columns_from_schema,
    load_evidence_meta,
)

DEFAULT_DATA_DIR = pathlib.Path.home() / ".claritymed/data/symptoms/ddxplus"
DEFAULT_N_TEST = 20_000
DEFAULT_MAXSTEP = 6
DEFAULT_STOP_THRES = 0.90
DEFAULT_PR_AUC_FLOOR = 0.70
ECE_BINS = 10
# Operating points reported alongside PR-AUC. Recall @ Precision=0.70
# reflects "modest false-alarm tolerance"; Precision @ Recall=0.90 reflects
# "must catch ≥ 90% of positives" — both are clinically motivated defaults
# and can be shifted by the caller if a target class needs different
# operating points.
RECALL_AT_PRECISION = 0.70
PRECISION_AT_RECALL = 0.90


# ---------------------------------------------------------------------------
# Projection helpers — pure functions, no classifier state.
# ---------------------------------------------------------------------------


def project_probs(
    probs_full: np.ndarray, target_idxs: Sequence[int], n_classes: int
) -> np.ndarray:
    """Reduce ``(batch, n_classes)`` posterior into ``(batch, N+1)``.

    ``target_idxs[k]`` becomes projected class ``k``; the trailing column
    is the sum over all non-target classes ('Other'). Sum-to-1 is
    preserved because we're summing over a disjoint partition of the raw
    class space. When ``probs_full.shape[-1] == len(target_idxs) + 1``
    the model is already native subset (v3) — the input is returned
    unchanged so callers don't need to branch on layout.
    """
    if probs_full.ndim == 1:
        probs_full = probs_full[np.newaxis, :]
    n_targets = len(target_idxs)
    if probs_full.shape[-1] == n_targets + 1:
        return probs_full
    out = np.zeros((probs_full.shape[0], n_targets + 1), dtype=probs_full.dtype)
    for k, idx in enumerate(target_idxs):
        out[:, k] = probs_full[:, idx]
    other_mask = np.ones(n_classes, dtype=bool)
    other_mask[list(target_idxs)] = False
    out[:, n_targets] = probs_full[:, other_mask].sum(axis=1)
    return out


def project_labels(labels_full: np.ndarray, target_idxs: Sequence[int]) -> np.ndarray:
    """Reduce integer disease labels into subset-index space.

    ``labels_full[i] == target_idxs[k]`` → returns ``k``.
    ``labels_full[i] not in target_idxs`` → returns ``len(target_idxs)``
    (the trailing 'Other' bucket).
    """
    n_targets = len(target_idxs)
    out = np.full(labels_full.shape[0], n_targets, dtype=np.int64)
    for k, idx in enumerate(target_idxs):
        out[labels_full == idx] = k
    return out


# ---------------------------------------------------------------------------
# Metrics — subset-target-aware.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TargetMetrics:
    """Per-target quality snapshot on a single eval regime."""

    name: str
    pr_auc: float
    recall_at_precision: float  # recall achievable at fixed precision floor
    precision_at_recall: float  # precision achievable at fixed recall floor
    ece: float  # expected calibration error


@dataclasses.dataclass
class RegimeReport:
    """One eval regime's full report (static or interactive-mode)."""

    label: str
    per_target: list[TargetMetrics]
    three_way_confusion: np.ndarray  # (N+1, N+1); rows=true, cols=pred
    argmax_acc: float  # 3-way ACC on projected labels
    il_mean: float | None  # None for static eval


def _expected_calibration_error(
    y_true_binary: np.ndarray, prob_positive: np.ndarray, n_bins: int = ECE_BINS
) -> float:
    """ECE on a binary target.

    Bins predicted probability into ``n_bins`` equal-width buckets. For
    each bucket, compare the predicted mean to the observed positive
    rate. Weighted L1 by bucket size gives ECE. Returns 0 when no
    predictions land in any bucket (defensive; shouldn't happen with
    real classifier outputs).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true_binary)
    if n == 0:
        return float("nan")
    for lo, hi in zip(bins[:-1], bins[1:]):
        # Right-open bins except the last (include 1.0).
        if hi == bins[-1]:
            mask = (prob_positive >= lo) & (prob_positive <= hi)
        else:
            mask = (prob_positive >= lo) & (prob_positive < hi)
        if not mask.any():
            continue
        acc_bin = float(y_true_binary[mask].mean())
        conf_bin = float(prob_positive[mask].mean())
        weight = mask.sum() / n
        ece += weight * abs(acc_bin - conf_bin)
    return float(ece)


def _target_metrics(
    y_true_proj: np.ndarray,
    probs_proj: np.ndarray,
    target_names: list[str],
) -> list[TargetMetrics]:
    """Compute per-target PR-AUC + operating-point + ECE metrics."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    out: list[TargetMetrics] = []
    for k, name in enumerate(target_names):
        y_binary = (y_true_proj == k).astype(np.int64)
        prob_k = probs_proj[:, k]
        if y_binary.sum() == 0:
            # No positives in test set for this target — PR-AUC undefined.
            out.append(
                TargetMetrics(
                    name, float("nan"), float("nan"), float("nan"), float("nan")
                )
            )
            continue
        pr_auc = float(average_precision_score(y_binary, prob_k))
        prec, rec, _thres = precision_recall_curve(y_binary, prob_k)
        # Recall at Precision >= floor: pick highest recall among points
        # meeting the precision floor. precision_recall_curve returns
        # arrays sorted by decreasing threshold (increasing recall);
        # inspect the qualifying tail.
        qualifying_precision = prec >= RECALL_AT_PRECISION
        recall_at_prec = (
            float(rec[qualifying_precision].max())
            if qualifying_precision.any()
            else 0.0
        )
        # Precision at Recall >= floor: highest precision among points
        # with recall >= floor.
        qualifying_recall = rec >= PRECISION_AT_RECALL
        prec_at_recall = (
            float(prec[qualifying_recall].max()) if qualifying_recall.any() else 0.0
        )
        ece = _expected_calibration_error(y_binary, prob_k)
        out.append(TargetMetrics(name, pr_auc, recall_at_prec, prec_at_recall, ece))
    return out


def _confusion_matrix(
    y_true_proj: np.ndarray, y_pred_proj: np.ndarray, n_buckets: int
) -> np.ndarray:
    """Build a ``(n_buckets, n_buckets)`` confusion matrix.

    Rows = true bucket; columns = predicted bucket. Kept as a plain
    ndarray so the CLI printer can format its own output without sklearn.
    """
    m = np.zeros((n_buckets, n_buckets), dtype=np.int64)
    for t, p in zip(y_true_proj, y_pred_proj):
        m[t, p] += 1
    return m


# ---------------------------------------------------------------------------
# Static regime — full patient signatures at once (upper bound).
# ---------------------------------------------------------------------------


def eval_static(
    agent: XgbAgent,
    patients: list[dict],
    schema: dict,
    target_idxs: Sequence[int],
    target_names: list[str],
    n_classes: int,
) -> RegimeReport:
    """Static classifier eval on fully-revealed patient signatures."""
    meta = load_evidence_meta(DEFAULT_DATA_DIR)
    columns, _labels, columns_idx = feature_columns_from_schema(schema, meta)
    x_full = encode_patient_batch(patients, schema, columns_idx)
    probs_full = agent.classifier.predict_proba(x_full)
    probs_proj = project_probs(probs_full, target_idxs, n_classes)
    y_true_full = np.asarray([p["d"] for p in patients], dtype=np.int64)
    y_true_proj = project_labels(y_true_full, target_idxs)
    y_pred_proj = np.argmax(probs_proj, axis=1)
    per_target = _target_metrics(y_true_proj, probs_proj, target_names)
    conf = _confusion_matrix(y_true_proj, y_pred_proj, n_buckets=len(target_names) + 1)
    argmax_acc = float((y_pred_proj == y_true_proj).mean())
    return RegimeReport(
        label="Static (full signatures)",
        per_target=per_target,
        three_way_confusion=conf,
        argmax_acc=argmax_acc,
        il_mean=None,
    )


# ---------------------------------------------------------------------------
# Interactive regime — mini loop that reveals up to ``maxstep`` evidences.
# ---------------------------------------------------------------------------


def _mini_interactive(
    agent: XgbAgent,
    patients: list[dict],
    schema: dict,
    n_dis: int,
    maxstep: int,
    stop_check,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the standard IG loop; return (final_probs [B, n_classes], il [B]).

    ``stop_check(probs_full)`` → bool array (B,) governs early termination
    per patient. Kept as a callable so the caller can pass a projected
    stop, a raw stop, or Variant A without the agent knowing.
    """
    env = TypedEnv(list(patients), schema, n_dis)
    n = len(patients)
    state, done_env = env.initialize_state(n)
    il = np.zeros(n, dtype=np.int64)
    stopped = np.zeros(n, dtype=bool)
    for _step in range(maxstep):
        # Compute posterior at current state to check stop condition.
        _, probs = agent.diagnose(state)
        stop_now = stop_check(probs)
        stopped |= stop_now
        if stopped.all():
            break
        next_evs = agent.next_action(state)
        # env.reveal returns a NEW state — it copies, writes evidence
        # slots for non-``done`` rows, and returns the copy. Discarding
        # the return value leaves ``state`` frozen at the init snapshot
        # and every subsequent iteration is a no-op — the exact bug that
        # made the earlier Phase C / v3 interactive numbers meaningless.
        state = env.reveal(state, next_evs, stopped)
        il[~stopped] += 1
    _, probs_final = agent.diagnose(state)
    return probs_final, il


def eval_interactive(
    label: str,
    agent: XgbAgent,
    patients: list[dict],
    schema: dict,
    n_dis: int,
    target_idxs: Sequence[int],
    target_names: list[str],
    n_classes: int,
    maxstep: int,
    stop_check,
) -> RegimeReport:
    """Run mini interactive eval + project + compute per-target metrics."""
    probs_full, il = _mini_interactive(
        agent, patients, schema, n_dis, maxstep, stop_check
    )
    probs_proj = project_probs(probs_full, target_idxs, n_classes)
    y_true_full = np.asarray([p["d"] for p in patients], dtype=np.int64)
    y_true_proj = project_labels(y_true_full, target_idxs)
    y_pred_proj = np.argmax(probs_proj, axis=1)
    per_target = _target_metrics(y_true_proj, probs_proj, target_names)
    conf = _confusion_matrix(y_true_proj, y_pred_proj, n_buckets=len(target_names) + 1)
    argmax_acc = float((y_pred_proj == y_true_proj).mean())
    return RegimeReport(
        label=label,
        per_target=per_target,
        three_way_confusion=conf,
        argmax_acc=argmax_acc,
        il_mean=float(il.mean()),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(report: RegimeReport, target_names: list[str]) -> None:
    """Human-readable one-regime block for the CLI."""
    print(f"\n=== {report.label} ===")
    if report.il_mean is not None:
        print(f"  IL mean: {report.il_mean:.3f}")
    print(f"  3-way ACC: {report.argmax_acc * 100:.2f}%")
    print("  Per-target metrics:")
    print(
        f"    {'target':<20s}  {'PR-AUC':>7s}  "
        f"{'R@P=' + str(RECALL_AT_PRECISION):>10s}  "
        f"{'P@R=' + str(PRECISION_AT_RECALL):>10s}  {'ECE':>6s}"
    )
    for m in report.per_target:
        print(
            f"    {m.name:<20s}  {m.pr_auc:>7.3f}  "
            f"{m.recall_at_precision:>10.3f}  "
            f"{m.precision_at_recall:>10.3f}  {m.ece:>6.3f}"
        )
    labels = list(target_names) + ["Other"]
    print("  Confusion (rows=true, cols=pred):")
    print("    " + "".join(f"{name[:8]:>10s}" for name in labels))
    for i, row in enumerate(report.three_way_confusion):
        print(f"    {labels[i][:8]:<10s}" + "".join(f"{v:>10d}" for v in row))


def _make_stop_checks(target_idxs: Sequence[int], n_classes: int, thres: float):
    """Return three ``stop_check`` callables for the interactive modes."""

    def stop_raw(probs_full: np.ndarray) -> np.ndarray:
        return probs_full.max(axis=1) > thres

    def stop_proj_max(probs_full: np.ndarray) -> np.ndarray:
        proj = project_probs(probs_full, target_idxs, n_classes)
        return proj.max(axis=1) > thres

    def stop_variant_a(probs_full: np.ndarray) -> np.ndarray:
        proj = project_probs(probs_full, target_idxs, n_classes)
        n_targets = len(target_idxs)
        target_max = proj[:, :n_targets].max(axis=1)
        other = proj[:, n_targets]
        return (target_max > 0.60) | (other > 0.85)

    return stop_raw, stop_proj_max, stop_variant_a


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry — parameterizes everything through --targets."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--weights", type=pathlib.Path, required=True)
    ap.add_argument(
        "--targets",
        default="Pneumonia,Influenza",
        help="Comma-separated target condition names. Every non-target class "
        "lumps into 'Other'. Order defines projected class indices 0..N-1.",
    )
    ap.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    ap.add_argument("--maxstep", type=int, default=DEFAULT_MAXSTEP)
    ap.add_argument("--stop-thres", type=float, default=DEFAULT_STOP_THRES)
    ap.add_argument(
        "--pr-auc-floor",
        type=float,
        default=DEFAULT_PR_AUC_FLOOR,
        help="If all target PR-AUCs exceed this floor in static eval, the "
        "classifier is deemed 'good enough' and Phase B retrain is optional.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-interactive",
        action="store_true",
        help="Only run static eval (fast; skips the IG loops).",
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    schema = load_evidence_schema(args.data_dir)
    agent = XgbAgent.load(args.weights, schema)
    n_classes = int(agent.classifier.classes_.shape[0])
    print(f"model: {args.weights.name}   n_classes={n_classes}")

    target_names = [n.strip() for n in args.targets.split(",") if n.strip()]
    if len(target_names) < 1:
        raise SystemExit("--targets needs at least one name")

    # Load full 49-class test set (do NOT whitelist — PR-AUC needs negatives).
    full_pidx, _ = load_pidx(args.data_dir, whitelist=None)
    missing = [n for n in target_names if n not in full_pidx]
    if missing:
        raise SystemExit(f"targets not in pidx: {missing}")
    test_pats = load_patients(args.data_dir, args.n_test, "test", schema, full_pidx)
    print(f"test kept: {len(test_pats)}")

    # Auto-detect model layout:
    #   * ``n_classes == len(target_names) + 1`` → v3 native model. Test
    #     patient labels get relabelled from the 49-class ``pidx`` space to
    #     the subset space (targets first, then Other). ``target_idxs`` is
    #     just ``[0, 1, ..., N-1]`` — the classifier already emits N+1-class
    #     probs so no projection is needed at eval time.
    #   * Otherwise → PoC / v2 49-class model. Test labels stay in full
    #     pidx space and ``target_idxs`` points to the raw pidx positions
    #     of the target names; probs get projected 49 → N+1 at eval.
    n_subset = len(target_names) + 1
    is_native_subset = n_classes == n_subset
    if is_native_subset:
        print(f"detected v3 native subset model ({n_subset}-class output)")
        # Relabel test patients into subset space in-place.
        subset_mapping = {full_pidx[n]: k for k, n in enumerate(target_names)}
        for full_name, full_idx in full_pidx.items():
            if full_idx not in subset_mapping:
                subset_mapping[full_idx] = len(target_names)
        for p in test_pats:
            p["d"] = subset_mapping[p["d"]]
        target_idxs = list(range(len(target_names)))
    else:
        print(f"detected legacy {n_classes}-class model — projecting at eval")
        target_idxs = [full_pidx[n] for n in target_names]
    print(f"targets → class idxs: {dict(zip(target_names, target_idxs))}")

    # --- Static eval (upper bound) ---
    static_report = eval_static(
        agent, test_pats, schema, target_idxs, target_names, n_classes
    )
    _print_report(static_report, target_names)

    if args.skip_interactive:
        _print_decision(static_report, args.pr_auc_floor)
        return

    # --- Interactive eval, 3 modes ---
    # Modes reuse the same underlying classifier; only the stop check and
    # (optionally) the projection passed to the IG policy differ.
    stop_raw, stop_proj_max, stop_variant_a = _make_stop_checks(
        target_idxs, n_classes, args.stop_thres
    )

    # Mode 1: raw IG + raw stop. Baseline of the untouched agent surface.
    def _reset(target: list[int] | None, policy: str = "proj_max") -> None:
        # Drop any cached projection so switching modes doesn't reuse stale.
        try:
            object.__delattr__(agent, "_cached_projection")
        except AttributeError:
            pass
        agent.target_class_idxs = target
        agent.stop_policy = policy

    _reset(target=None)
    m1 = eval_interactive(
        f"Interactive mode 1: raw IG + max(raw)>{args.stop_thres}",
        agent,
        test_pats,
        schema,
        len(full_pidx),
        target_idxs,
        target_names,
        n_classes,
        args.maxstep,
        stop_raw,
    )
    _print_report(m1, target_names)

    # Mode 2: projected IG + projected stop.
    _reset(target=list(target_idxs), policy="proj_max")
    m2 = eval_interactive(
        f"Interactive mode 2: projected IG + max(P_proj)>{args.stop_thres}",
        agent,
        test_pats,
        schema,
        len(full_pidx),
        target_idxs,
        target_names,
        n_classes,
        args.maxstep,
        stop_proj_max,
    )
    _print_report(m2, target_names)

    # Mode 3: projected IG + Variant A stop (asymmetric).
    _reset(target=list(target_idxs), policy="proj_variant_a")
    m3 = eval_interactive(
        "Interactive mode 3: projected IG + Variant A (target>0.60 or Other>0.85)",
        agent,
        test_pats,
        schema,
        len(full_pidx),
        target_idxs,
        target_names,
        n_classes,
        args.maxstep,
        stop_variant_a,
    )
    _print_report(m3, target_names)

    _print_decision(static_report, args.pr_auc_floor, interactive_reports=[m1, m2, m3])


def _print_decision(
    static_report: RegimeReport,
    pr_auc_floor: float,
    interactive_reports: list[RegimeReport] | None = None,
) -> None:
    """Emit a Phase-A/Phase-B decision line at the bottom of the report."""
    print("\n=== Decision ===")
    all_pass = all(m.pr_auc >= pr_auc_floor for m in static_report.per_target)
    print(f"  PR-AUC floor: {pr_auc_floor:.2f}")
    print("  Static-eval PR-AUC per target:")
    for m in static_report.per_target:
        pass_ = "PASS" if m.pr_auc >= pr_auc_floor else "FAIL"
        print(f"    {m.name}: {m.pr_auc:.3f}  [{pass_}]")
    if all_pass:
        print("\n  → Classifier already has target signal. Phase B (retrain 3-class)")
        print("    is OPTIONAL. Focus on policy + UI: pick an interactive mode with")
        print("    good IL/PR-AUC trade-off, show P(target) explicitly in the UI.")
    else:
        print("\n  → Classifier lacks target signal at the required floor.")
        print("    Phase B REQUIRED: retrain as 3-class native with sample_weight")
        print("    boost on target classes + Platt calibration.")


if __name__ == "__main__":
    main()
