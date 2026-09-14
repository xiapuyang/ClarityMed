"""Post-hoc isotonic calibration for a fitted XGBoost symptoms model.

Loads an existing checkpoint (e.g. ``xgb_pne_inf_v5_recallig``), fits a
``CalibratedClassifierCV(method='isotonic')`` on the DDXPlus validation
split using ``FrozenEstimator`` (matches the pre-fit semantics without
retraining the base XGBoost), and writes a new checkpoint whose
``classifier`` attribute is the calibrated wrapper. The base model's
tree structure is untouched — only its output probabilities are
remapped through a per-class monotonic function.

Purpose: reduce the false-positive rate on rare classes (Pne/Flu) when
the base model is over-confident. Because isotonic is monotonic within
each class, ranking is preserved — recall stays put for well-supported
positives; only borderline scores near the stop threshold shift.

Usage::

    uv run python scripts/calibrate_symptoms_model.py \\
        --source-model-id xgb_pne_inf_v5_recallig \\
        --target-model-id xgb_pne_inf_v5_iso \\
        --n-val 20000

Downstream: pair with the new model_id in ``scripts/eval_symptoms_metrics.py``
for A/B comparison against the source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import yaml

from claritymed.core.symptoms.datasets.canonical import slugify_condition
from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_patients,
    load_pidx,
)
from claritymed.ingest.symptoms.typed_basd import TypedEnv, seed_everything
from claritymed.ingest.symptoms.xgb.algorithm import XgbAgent
from claritymed.ingest.symptoms.xgb.encoding import (
    encode_patient_batch,
    encode_typed_state_batch,
    feature_columns_from_schema,
    load_evidence_meta,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "symptoms.yaml"
DEFAULT_DATA_DIR = pathlib.Path.home() / ".claritymed" / "data" / "symptoms" / "ddxplus"
DEFAULT_MODELS_ROOT = pathlib.Path.home() / ".claritymed" / "models" / "symptoms"
DEFAULT_DATASET_ID = "ddxplus_pneumonia_flu"
DEFAULT_SOURCE_MODEL_ID = "xgb_pne_inf_v5_recallig"
DEFAULT_TARGET_MODEL_ID = "xgb_pne_inf_v5_iso"
DEFAULT_N_VAL = 20_000
DEFAULT_SEED = 42
DEFAULT_METHOD = "isotonic"
SUPPORTED_METHODS = ("isotonic", "sigmoid", "platt")
DEFAULT_SAMPLING = "full"
SUPPORTED_SAMPLING = ("full", "rollout")


def _resolve_source_spec(model_id: str, dataset_id: str) -> tuple[dict, list[str]]:
    """Look up source model + its dataset's target_condition_ids.

    Returns ``(model_spec_dict, target_condition_ids)``. Both are lifted
    from ``configs/symptoms.yaml``; the model spec dict is returned raw
    so the caller can copy every field into the target model entry.
    """
    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    datasets = {d["id"]: d for d in cfg["datasets"]}
    if dataset_id not in datasets:
        raise SystemExit(f"dataset {dataset_id!r} not in {list(datasets)}")
    target_ids = list(datasets[dataset_id]["target_condition_ids"])
    models = {m["id"]: m for m in cfg["models"]}
    if model_id not in models:
        raise SystemExit(f"model {model_id!r} not in models[] of {CONFIG_PATH}")
    return models[model_id], target_ids


def _target_slugs_to_pidx(
    target_slugs: list[str], full_pidx: dict[str, int]
) -> dict[str, int]:
    slug_to_idx: dict[str, int] = {
        slugify_condition(display): idx for display, idx in full_pidx.items()
    }
    missing = [s for s in target_slugs if s not in slug_to_idx]
    if missing:
        raise SystemExit(f"target slugs {missing} have no pidx match")
    return {slug: slug_to_idx[slug] for slug in target_slugs}


def _relabel_to_subset(
    patients: list[dict], slug_to_idx: dict[str, int], target_ids: list[str]
) -> None:
    mapping = {slug_to_idx[name]: k for k, name in enumerate(target_ids)}
    other_idx = len(target_ids)
    for p in patients:
        p["d"] = mapping.get(p["d"], other_idx)


def _rollout_snapshots(
    agent: XgbAgent,
    patients: list[dict],
    schema: dict,
    n_dis: int,
    maxstep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot XGBoost input vectors at every turn of the IG loop.

    Runs the exact same loop production uses (agent.should_stop /
    next_action / TypedEnv.reveal), capturing the encoded state fed
    into the classifier at each still-running turn. Returns
    ``(X [total_snapshots, n_features], y [total_snapshots])`` — the
    calibration training set that reflects the distribution of
    classifier inputs the model actually sees at diagnosis time,
    NOT the full-signature distribution the training loop saw.

    A patient contributes one snapshot per turn it is still running.
    Patients that fire ``should_stop`` at turn 0 contribute exactly
    one snapshot (their init state), matching what the stop gate
    evaluated on.
    """
    env = TypedEnv(list(patients), schema, n_dis)
    n = len(patients)
    state, _done_env = env.initialize_state(n)
    stopped = np.zeros(n, dtype=bool)
    y_true = np.asarray([p["d"] for p in patients], dtype=np.int64)

    x_snapshots: list[np.ndarray] = []
    y_snapshots: list[np.ndarray] = []
    per_turn_counts: list[int] = []

    for step in range(maxstep):
        active = ~stopped
        if not active.any():
            break
        active_states = state[active]
        x_xgb, _asked = encode_typed_state_batch(
            active_states, schema, agent.ev_col_index, agent.n_features
        )
        x_snapshots.append(x_xgb)
        y_snapshots.append(y_true[active])
        per_turn_counts.append(int(active.sum()))

        stop_now = agent.should_stop(state)
        stopped |= stop_now
        if stopped.all():
            break
        next_evs = agent.next_action(state)
        state = env.reveal(state, next_evs, stopped)

    x_cat = np.vstack(x_snapshots)
    y_cat = np.concatenate(y_snapshots)
    print(
        f"[rollout snapshots] per-turn active counts: {per_turn_counts}  "
        f"total_samples={len(y_cat)}"
    )
    return x_cat, y_cat


def _fit_calibrator(clf, x_val: np.ndarray, y_val: np.ndarray, method: str):
    """Wrap the fitted classifier in per-class calibrated one-vs-rest heads.

    Uses :class:`FrozenEstimator` so the underlying XGBoost trees stay
    intact; sklearn>=1.6 removed the ``cv="prefit"`` string in favour
    of this explicit wrapper. ``method`` is one of ``isotonic`` (non-
    parametric, prone to overfitting on small val sets) or ``sigmoid``
    (Platt scaling, 2-parameter, smoother — recommended default for
    XGBoost per train.py:378-384). ``platt`` is an operator-friendly
    alias for ``sigmoid``.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    sk_method = "sigmoid" if method == "platt" else method
    cal = CalibratedClassifierCV(FrozenEstimator(clf), method=sk_method)
    cal.fit(x_val, y_val)
    return cal


def _log_loss_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Mean negative log-likelihood on the true class.

    Kept as a hand-rolled function (rather than pulling sklearn's
    log_loss) so the eps clipping is explicit — matches what production
    stop logic effectively cares about (over-confidence on wrong class).
    """
    eps = 1e-12
    n = len(y_true)
    return float(-np.log(np.clip(probs[np.arange(n), y_true], eps, 1.0)).mean())


def _brier_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    n, k = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y_true] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_and_bump_manifest(
    source_dir: pathlib.Path,
    target_dir: pathlib.Path,
    target_model_id: str,
    new_weights_sha: str,
    calibration_info: dict,
) -> pathlib.Path:
    """Duplicate the source manifest, retag it, and rewrite the sha256.

    Keeps every other field (feature_columns, diseases_trained,
    algorithm_module, algorithm_module_version) intact so the adapter's
    manifest-verification checks pass identically.
    """
    src = source_dir / "manifest.json"
    if not src.exists():
        raise SystemExit(f"source manifest not found: {src}")
    manifest = json.loads(src.read_text(encoding="utf-8"))
    manifest["model_id"] = target_model_id
    manifest["sha256"] = new_weights_sha
    manifest["calibration"] = calibration_info
    dst = target_dir / "manifest.json"
    dst.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-model-id", default=DEFAULT_SOURCE_MODEL_ID)
    ap.add_argument("--target-model-id", default=DEFAULT_TARGET_MODEL_ID)
    ap.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    ap.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--models-root", type=pathlib.Path, default=DEFAULT_MODELS_ROOT)
    ap.add_argument(
        "--n-val",
        type=int,
        default=DEFAULT_N_VAL,
        help="Validation patients for fitting the calibrator.",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default=DEFAULT_METHOD,
        help="Calibration algorithm. isotonic: non-parametric, "
        "overfits on small val sets. sigmoid/platt: 2-param, smoother.",
    )
    ap.add_argument(
        "--sampling",
        choices=SUPPORTED_SAMPLING,
        default=DEFAULT_SAMPLING,
        help="Calibration training data: 'full' encodes each patient's "
        "complete evidence signature (matches train-time regime, wrong "
        "distribution for BASD stop). 'rollout' runs the IG loop on "
        "validation and snapshots the encoded state at every turn — "
        "matches production inference distribution.",
    )
    ap.add_argument(
        "--maxstep",
        type=int,
        default=None,
        help="Cap rollout depth. Defaults to the source model's config maxstep.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target checkpoint dir.",
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    source_spec, target_ids = _resolve_source_spec(
        args.source_model_id, args.dataset_id
    )
    if source_spec.get("algorithm_module") != "xgb":
        raise SystemExit("source model must be algorithm_module=xgb")

    source_dir = args.models_root / source_spec["weights_subpath"]
    target_subpath = (
        pathlib.PurePosixPath(source_spec["weights_subpath"]).parent
        / args.target_model_id
    )
    target_dir = args.models_root / target_subpath

    print(f"source: {source_dir}")
    print(f"target: {target_dir}")
    if target_dir.exists() and not args.overwrite:
        raise SystemExit(
            f"target dir already exists (pass --overwrite to replace): {target_dir}"
        )
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True)

    schema = load_evidence_schema(args.data_dir)
    full_pidx, _ = load_pidx(args.data_dir, whitelist=None)
    slug_to_idx = _target_slugs_to_pidx(target_ids, full_pidx)

    # Validation split (kept strictly separate from test split used by
    # scripts/eval_symptoms_metrics.py — no data leakage into the
    # A/B comparison downstream).
    val_patients = load_patients(
        args.data_dir, args.n_val, "validate", schema, full_pidx
    )
    _relabel_to_subset(val_patients, slug_to_idx, target_ids)
    print(f"validation patients: {len(val_patients)} (requested {args.n_val})")

    src_weights = source_dir / "weights.pkl"
    if not src_weights.exists():
        raise SystemExit(f"source weights not found: {src_weights}")
    agent = XgbAgent.load(src_weights, schema)

    if args.sampling == "rollout":
        # Mirror the production-side overrides so the rollout the
        # calibrator sees matches what BASD actually does at serve time.
        # Without this, the snapshots would come from the checkpoint-
        # baked defaults (e.g. wrong stop policy, no antecedent penalty),
        # and calibration would target the wrong distribution.
        agent.thres = float(source_spec.get("stop_thres", agent.thres))
        if source_spec.get("antecedent_penalty") is not None:
            agent.antecedent_penalty = float(source_spec["antecedent_penalty"])
        if source_spec.get("ig_recall_weight") is not None:
            agent.ig_recall_weight = float(source_spec["ig_recall_weight"])
        if source_spec.get("ig_recall_mode") is not None:
            agent.ig_recall_mode = source_spec["ig_recall_mode"]
        if source_spec.get("stop_policy") is not None:
            agent.stop_policy = source_spec["stop_policy"]
        if source_spec.get("stop_target_thres") is not None:
            tt = float(source_spec["stop_target_thres"])
            agent.variant_a_target_thres = tt
            agent.target_sum_target_thres = tt
        if source_spec.get("stop_other_thres") is not None:
            ot = float(source_spec["stop_other_thres"])
            agent.variant_a_other_thres = ot
            agent.target_sum_other_thres = ot
        # v3 native subset — first N target class indices.
        agent.target_class_idxs = list(range(len(target_ids)))

    y_val = np.asarray([p["d"] for p in val_patients], dtype=np.int64)
    if args.sampling == "full":
        meta = load_evidence_meta(args.data_dir)
        _cols, _labels, columns_idx = feature_columns_from_schema(schema, meta)
        x_val = encode_patient_batch(val_patients, schema, columns_idx)
    else:
        maxstep = args.maxstep or int(source_spec.get("maxstep", 18))
        print(f"[rollout] maxstep={maxstep}  stop_policy={agent.stop_policy}")
        x_val, y_val = _rollout_snapshots(
            agent, val_patients, schema, n_dis=len(full_pidx), maxstep=maxstep
        )

    # Pre-calibration metrics — establish baseline before we overwrite classifier.
    t0 = time.perf_counter()
    probs_pre = agent.classifier.predict_proba(x_val)
    pre_ll = _log_loss_multiclass(y_val, probs_pre)
    pre_brier = _brier_multiclass(y_val, probs_pre)
    pre_acc = float((probs_pre.argmax(axis=1) == y_val).mean())
    print(
        f"[pre-calibration on validation] log_loss={pre_ll:.4f}  "
        f"brier={pre_brier:.4f}  acc={pre_acc:.4f}"
    )

    print(f"fitting CalibratedClassifierCV(method={args.method!r}, FrozenEstimator)...")
    calibrated = _fit_calibrator(agent.classifier, x_val, y_val, args.method)
    probs_post = calibrated.predict_proba(x_val)
    post_ll = _log_loss_multiclass(y_val, probs_post)
    post_brier = _brier_multiclass(y_val, probs_post)
    post_acc = float((probs_post.argmax(axis=1) == y_val).mean())
    fit_seconds = time.perf_counter() - t0
    print(
        f"[post-calibration on validation] log_loss={post_ll:.4f}  "
        f"brier={post_brier:.4f}  acc={post_acc:.4f}   "
        f"(fit+eval took {fit_seconds:.1f}s)"
    )

    # Swap in the calibrated classifier; the rest of the agent stays as-is
    # (ev_marginals, columns, ig knobs, etc. are untouched).
    agent.classifier = calibrated
    target_weights = target_dir / "weights.pkl"
    agent.save(target_weights)
    new_sha = _sha256(target_weights)
    print(f"wrote {target_weights} (sha256={new_sha[:16]}...)")

    calibration_info = {
        "method": args.method,
        "sampling": args.sampling,
        "wrapper": "sklearn.calibration.CalibratedClassifierCV + FrozenEstimator",
        "source_model_id": args.source_model_id,
        "n_val_patients": len(val_patients),
        "n_calibration_samples": int(len(y_val)),
        "seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validation_metrics": {
            "pre_log_loss": pre_ll,
            "post_log_loss": post_ll,
            "pre_brier": pre_brier,
            "post_brier": post_brier,
            "pre_argmax_acc": pre_acc,
            "post_argmax_acc": post_acc,
        },
    }
    manifest_path = _copy_and_bump_manifest(
        source_dir, target_dir, args.target_model_id, new_sha, calibration_info
    )
    print(f"wrote {manifest_path}")

    print(f"\nnew manifest_sha256 (for configs/symptoms.yaml): {new_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
