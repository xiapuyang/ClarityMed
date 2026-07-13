"""Training entry: ``claritymed-symptoms-xgb-train-ddxplus``.

Trains the :mod:`claritymed.ingest.symptoms.xgb` algorithm module on
DDXPlus and writes weights + manifest under
``CLARITYMED_HOME/models/symptoms/<weights_subpath>/``. Mirrors the
typed-BASD train CLI's flag surface where semantics overlap (``--diseases``,
``--data-dir``, ``--seed``, ``--smoke``) and diverges only where
XGBoost has genuinely different knobs.

Pipeline steps:

1. Load schema + pidx (subset-aware via ``--diseases``) + patients.
2. One-hot encode training patients to a dense ``(N, F)`` matrix +
   ``(N,)`` pathology label vector.
3. Fit the classifier (``binary:logistic`` for N=2, ``multi:softprob`` else).
4. Optionally wrap in :class:`CalibratedClassifierCV` (Platt or isotonic).
5. Build ``(X_masked, X_true)`` pairs by random-mask sampling on the
   training patients and fit a :class:`MultiOutputRegressor` around
   :class:`XGBRegressor` — this is the ev-marginals model the IG
   policy uses to estimate ``P(col=1 | state)`` (D1 in the plan;
   MultiOutputRegressor is the recommended start).
6. Evaluate on the test split via typed-BASD's ``interactive_eval``
   loop — XgbAgent's methods drop straight in.
7. Save weights (joblib pickle) + manifest.json.

The classifier / marginals / calibration split is intentional: keeping
these as three separate artifacts in the joblib checkpoint means the
adapter can swap any one at load time (e.g. try isotonic calibration
without re-training the classifier) without a fresh training run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from claritymed import config as _cfg
from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_patients,
    load_pidx,
)
from claritymed.ingest.symptoms.mlflow_utils import log_eval_metrics, symptom_run
from claritymed.ingest.symptoms.typed_basd import (
    EvalMetrics,
    TypedEnv,
    interactive_eval,
    seed_everything,
)
from claritymed.ingest.symptoms.xgb.algorithm import XgbAgent, build_xgb_agent
from claritymed.ingest.symptoms.xgb.encoding import (
    encode_patient_batch,
    feature_columns_from_schema,
    load_evidence_meta,
)

MANIFEST_VERSION = 1
ALGORITHM_MODULE_VERSION = 1
DEFAULT_N_TRAIN = 300_000
DEFAULT_N_TEST = 20_000
DEFAULT_EVAL_N_VAL = 2_000
DEFAULT_MAX_DEPTH = 6
DEFAULT_N_ESTIMATORS = 400
DEFAULT_LR = 0.05
DEFAULT_STOP_THRES = 0.90
DEFAULT_IG_SMOOTHING = 0.05
DEFAULT_MAXSTEP = 6
DEFAULT_KEEP_LO = 0.3
DEFAULT_KEEP_HI = 0.7
DEFAULT_TARGET_COST_MULTIPLIER = 1.0  # 1.0 → pure inverse-frequency balancing
FEATURE_IMPORTANCE_TOP_K = 30
# Placeholder severity for the synthetic ``Other`` class in --targets
# mode. interactive_eval uses severity only to compute DSR/PSR/PAR,
# which the v3 metric plan replaces anyway; this value is manifest-noise,
# not a clinical claim. 3 == Moderate ("see your doctor within a week"),
# the safest neutral default.
_OTHER_CLASS_SEVERITY = 3


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _models_dir() -> Path:
    return _cfg.CLARITYMED_HOME / "models" / "symptoms"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _resolve_subset_mapping(
    target_names: list[str],
    full_pidx: dict[str, int],
) -> tuple[dict[int, int], int]:
    """Build ``(full_idx → subset_idx, n_subset)`` for --targets mode.

    Each target name resolves to its corresponding ``full_pidx`` value
    and maps to its position in ``target_names`` (0..N-1). Every other
    full-pidx index maps to ``N`` — the trailing "Other" class. Fails
    loud if a target name isn't in the full pidx (typo protection).
    """
    missing = [n for n in target_names if n not in full_pidx]
    if missing:
        raise SystemExit(
            f"--targets contains names not in DDXPlus release_conditions.json: "
            f"{missing!r}. Available diseases: {sorted(full_pidx)}"
        )
    n_targets = len(target_names)
    mapping: dict[int, int] = {}
    for k, name in enumerate(target_names):
        mapping[full_pidx[name]] = k
    for full_idx in full_pidx.values():
        if full_idx not in mapping:
            mapping[full_idx] = n_targets
    return mapping, n_targets + 1


def _relabel_patients(patients: list[dict], mapping: dict[int, int]) -> None:
    """Rewrite each patient's ``d`` field via the subset mapping (in place).

    The ``diff`` vector — a probability distribution over pidx classes —
    is also collapsed into subset space so downstream DDR/DDP/DDF1
    metrics computed by :func:`interactive_eval` at training time don't
    broadcast-mismatch against the classifier's ``(N+1,)`` output.
    Metric semantics under the new class map are undocumented; treat
    them as sanity numbers, not tune signal. The subset-aware
    Phase 2 eval script is authoritative.
    """
    if not patients:
        return
    n_subset = max(mapping.values()) + 1
    for p in patients:
        p["d"] = mapping[p["d"]]
        old_diff = p["diff"]
        new_diff = np.zeros(n_subset, dtype=old_diff.dtype)
        for full_idx, val in enumerate(old_diff):
            new_diff[mapping[full_idx]] += val
        p["diff"] = new_diff


def _build_sample_weights(
    y: np.ndarray,
    n_classes: int,
    target_cost_multiplier: float = 1.0,
    n_targets: int | None = None,
) -> np.ndarray:
    """Return per-row weights that inverse-frequency balance all N+1 classes.

    Base rule per class ``k``: ``w_k = n_total / (n_classes * n_k)``. This
    makes every class contribute an equal-mass slice ``n_total / n_classes``
    to the training loss — the initial classifier posterior is uniform
    over the N+1 buckets regardless of the imbalance in the raw pidx
    (DDXPlus default: Pne ≈ 2.6%, Inf ≈ 2.7%, Other ≈ 94.7%).

    Automatic per-class balancing beats a hand-picked ``target/other``
    ratio because (a) it self-adjusts when the target subset changes,
    (b) it respects Pne vs Inf's own population difference (Pne is
    slightly less common → gets a slightly higher weight), and (c) it
    scales with ``n_train`` without requiring the operator to re-tune
    a scalar boost.

    ``target_cost_multiplier`` (optional > 1.0): after inverse-frequency
    balancing, multiply the first ``n_targets`` classes by this factor —
    used when missed-target cost (e.g. missed pneumonia) is asymmetric
    against missed-Other cost. When ``n_targets`` is None the multiplier
    applies to every class equally, which is a no-op. Set both together.

    Downstream classifiers (XGBoost, CalibratedClassifierCV) accept
    ``sample_weight`` natively; the weights are re-scaled by the
    classifier internally so absolute magnitude is unimportant.
    """
    if y.size == 0:
        return np.empty(0, dtype=np.float32)
    n_total = int(y.shape[0])
    class_counts = np.bincount(y, minlength=n_classes)
    weights = np.zeros(n_total, dtype=np.float32)
    for k in range(n_classes):
        if class_counts[k] == 0:
            # Class absent from this split — nothing to weight. Leaving
            # weights at 0 keeps the class out of the loss without
            # triggering a divide-by-zero.
            continue
        weights[y == k] = n_total / (n_classes * class_counts[k])
    if target_cost_multiplier != 1.0 and n_targets is not None:
        weights[y < n_targets] *= target_cost_multiplier
    return weights


def build_mask_pairs(
    x_true: np.ndarray,
    keep_lo: float,
    keep_hi: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly mask each row's set columns to simulate partial states.

    For each training row we pick a keep-rate uniformly in
    ``[keep_lo, keep_hi]`` and zero out ``1 - keep_rate`` of the row's
    set-to-1 columns. The regressor learns to predict the full
    signature from the masked version, which is what the IG policy
    needs at inference time — a partial state with some evidences
    revealed and others yet to be asked.

    ``keep_lo=1.0`` reproduces ``mask-policy=full``: no masking, the
    regressor sees full signatures (D4 in the plan).
    """
    x_masked = x_true.copy()
    for i in range(x_true.shape[0]):
        set_cols = np.flatnonzero(x_true[i])
        if len(set_cols) == 0:
            continue
        keep_rate = rng.uniform(keep_lo, keep_hi)
        n_keep = max(1, int(round(len(set_cols) * keep_rate)))
        drop = rng.choice(set_cols, size=len(set_cols) - n_keep, replace=False)
        x_masked[i, drop] = 0.0
    return x_masked, x_true


def _fit_classifier(
    x: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    *,
    max_depth: int,
    n_estimators: int,
    lr: float,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Fit an :class:`xgboost.XGBClassifier` with algorithm-appropriate objective.

    ``sample_weight`` is optional per-row row weight (typically the
    inverse-frequency balancing vector from :func:`_build_sample_weights`);
    XGBoost's histogram builder consumes it natively via ``fit(sample_weight=…)``.
    """
    import xgboost as xgb

    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    kwargs: dict[str, Any] = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        objective=objective,
        eval_metric="logloss" if n_classes == 2 else "mlogloss",
        n_jobs=-1,
        tree_method="hist",
        random_state=seed,
    )
    if n_classes != 2:
        kwargs["num_class"] = n_classes
    clf = xgb.XGBClassifier(**kwargs)
    clf.fit(x, y, sample_weight=sample_weight)
    return clf


def _calibrate(
    clf: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    method: str,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Wrap a fitted XGBoost classifier in :class:`CalibratedClassifierCV`.

    ``method="none"`` returns the classifier unchanged. Platt (sigmoid)
    is the recommended default for XGBoost — isotonic overfits on small
    calibration sets. In sklearn>=1.6 the pre-fit path goes through
    :class:`~sklearn.frozen.FrozenEstimator` (the ``cv="prefit"`` string
    was removed).
    """
    if method == "none":
        return clf
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    # Operator-friendly alias: "platt" is the paper's name for what
    # sklearn calls "sigmoid" (a logistic calibration head).
    sk_method = "sigmoid" if method == "platt" else method
    cal = CalibratedClassifierCV(FrozenEstimator(clf), method=sk_method)
    cal.fit(x_val, y_val, sample_weight=sample_weight)
    return cal


def _fit_ev_marginals(
    x_masked: np.ndarray,
    x_true: np.ndarray,
    *,
    max_depth: int,
    n_estimators: int,
    lr: float,
    seed: int,
) -> Any:
    """Fit a :class:`MultiOutputRegressor` around :class:`XGBRegressor`.

    D1 recommendation: single sklearn wrapper vs per-feature classifier.
    Recommendation start; revisit if IG picks look noisy.
    """
    from sklearn.multioutput import MultiOutputRegressor
    import xgboost as xgb

    base = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        objective="reg:squarederror",
        n_jobs=1,  # inner regressor per feature — outer MultiOutput does the parallelism
        tree_method="hist",
        random_state=seed,
    )
    reg = MultiOutputRegressor(base, n_jobs=-1)
    reg.fit(x_masked, x_true)
    return reg


def _feature_importance_top_k(
    clf: Any, columns: list[str], k: int
) -> list[dict[str, Any]]:
    """Extract (feature, gain) pairs from the underlying booster.

    Kept in the manifest so operators can eyeball the trained model's
    top signals without loading joblib pickles. When ``clf`` is a
    :class:`CalibratedClassifierCV` wrapper we descend into the wrapped
    base classifier — the calibration layer doesn't produce importances
    of its own.
    """
    base = clf
    if hasattr(clf, "calibrated_classifiers_"):
        base = clf.calibrated_classifiers_[0].estimator
    if not hasattr(base, "get_booster"):
        return []
    booster = base.get_booster()
    # xgboost's get_score keys are the classifier's own feature_names
    # (``f0``/``f1``/…) when we didn't pass feature_names at fit; we set
    # them here so the manifest carries meaningful names for operators.
    booster.feature_names = columns
    gain = booster.get_score(importance_type="gain")
    ranked = sorted(gain.items(), key=lambda kv: -kv[1])
    return [{"name": name, "gain": float(g)} for name, g in ranked[:k]]


def train_xgb_agent(
    schema: dict,
    train_patients: list[dict],
    val_patients: list[dict],
    columns: list[str],
    columns_idx: dict[str, int],
    n_classes: int,
    *,
    max_depth: int,
    n_estimators: int,
    lr: float,
    calibration: str,
    mask_policy: str,
    keep_lo: float,
    keep_hi: float,
    stop_thres: float,
    ig_smoothing: float,
    seed: int,
    class_balance: bool = False,
    n_targets: int | None = None,
    target_cost_multiplier: float = 1.0,
) -> XgbAgent:
    """Fit classifier + ev_marginals + optional calibration → :class:`XgbAgent`.

    Kept as a pure function (no CLI parsing / IO) so :mod:`test_train`
    can exercise it against synthetic fixtures without shelling out or
    touching DDXPlus data.

    When ``class_balance`` is True, per-row inverse-frequency sample
    weights are computed and passed through the classifier + calibration
    stages so every class contributes equal mass to the loss. Required
    for v3 subset training where target classes are ~5% of the population
    — without balancing the classifier defaults to Other and the IG policy
    stops at turn 0. ``n_targets`` + ``target_cost_multiplier`` further
    weight the first N classes (asymmetric target-vs-Other cost).
    """
    rng = np.random.default_rng(seed)
    x_train = encode_patient_batch(train_patients, schema, columns_idx)
    y_train = np.asarray([p["d"] for p in train_patients], dtype=np.int64)

    # Under ``mask_policy=random`` the classifier ALSO learns on partial
    # states — not just the ev_marginals regressor. Training only on full
    # signatures leaves the classifier dangerously over-confident on the
    # near-empty states it actually sees at turn 0 (which triggered the
    # IL=0 failure in the first real run). Fix: augment with THREE
    # random-masked replicas at different keep-rate bands so the
    # classifier sees the full spectrum from init-only through nearly-
    # complete state. 4× data, 4× fit cost — acceptable for a training
    # step measured in minutes; the alternative (over-confident classifier
    # → premature stop → zero-information eval) is worse.
    # Inverse-frequency sample weights (v3 subset training). Computed
    # once against the original unaugmented training labels so every
    # replica inherits the same per-row weight. Set to ``None`` when
    # class_balance is False (v2-compatible path).
    train_weights: np.ndarray | None = None
    if class_balance:
        train_weights = _build_sample_weights(
            y_train,
            n_classes=n_classes,
            target_cost_multiplier=target_cost_multiplier,
            n_targets=n_targets,
        )

    if mask_policy == "random":
        # Three masked replicas: very sparse (init-only-ish), mid, wide.
        # Ensures the calibration head + classifier see the near-empty
        # tail of the state distribution that turn 0 lives in.
        keep_bands = [
            (0.01, 0.15),
            (0.15, 0.50),
            (max(0.30, keep_lo), keep_hi),
        ]
        replicas = [x_train]
        replica_labels = [y_train]
        replica_weights: list[np.ndarray] | None = (
            [train_weights] if train_weights is not None else None
        )
        for i, (lo, hi) in enumerate(keep_bands):
            xm, _ = build_mask_pairs(
                x_train,
                keep_lo=lo,
                keep_hi=hi,
                rng=np.random.default_rng(seed + 100 + i),
            )
            replicas.append(xm)
            replica_labels.append(y_train)
            if replica_weights is not None:
                replica_weights.append(train_weights)
        x_clf = np.vstack(replicas)
        y_clf = np.concatenate(replica_labels)
        w_clf = np.concatenate(replica_weights) if replica_weights is not None else None
    else:
        x_clf = x_train
        y_clf = y_train
        w_clf = train_weights

    clf = _fit_classifier(
        x_clf,
        y_clf,
        n_classes=n_classes,
        max_depth=max_depth,
        n_estimators=n_estimators,
        lr=lr,
        seed=seed,
        sample_weight=w_clf,
    )
    if val_patients and calibration != "none":
        x_val = encode_patient_batch(val_patients, schema, columns_idx)
        y_val = np.asarray([p["d"] for p in val_patients], dtype=np.int64)
        # Val-set inverse-frequency weights match the classifier's own
        # balancing so the Platt sigmoid head fits the same distribution
        # the classifier saw. Computed on original val labels, then
        # replicated alongside the masked val augmentation below.
        val_weights: np.ndarray | None = None
        if class_balance:
            val_weights = _build_sample_weights(
                y_val,
                n_classes=n_classes,
                target_cost_multiplier=target_cost_multiplier,
                n_targets=n_targets,
            )
        # Calibration set: also augment with masked replicas so the
        # sigmoid head fits the same input distribution the classifier
        # will see at serve time. Mirrors the three keep-rate bands the
        # classifier training uses.
        if mask_policy == "random":
            xv_bands = [
                (0.01, 0.15),
                (0.15, 0.50),
                (max(0.30, keep_lo), keep_hi),
            ]
            val_replicas = [x_val]
            val_labels = [y_val]
            val_weight_replicas: list[np.ndarray] | None = (
                [val_weights] if val_weights is not None else None
            )
            for i, (lo, hi) in enumerate(xv_bands):
                xvm, _ = build_mask_pairs(
                    x_val,
                    keep_lo=lo,
                    keep_hi=hi,
                    rng=np.random.default_rng(seed + 200 + i),
                )
                val_replicas.append(xvm)
                val_labels.append(y_val)
                if val_weight_replicas is not None:
                    val_weight_replicas.append(val_weights)
            x_val = np.vstack(val_replicas)
            y_val = np.concatenate(val_labels)
            if val_weight_replicas is not None:
                val_weights = np.concatenate(val_weight_replicas)
        clf = _calibrate(
            clf, x_val, y_val, method=calibration, sample_weight=val_weights
        )

    ev_marginals: Any | None = None
    if mask_policy != "full":
        x_masked, x_true = build_mask_pairs(x_train, keep_lo, keep_hi, rng)
        ev_marginals = _fit_ev_marginals(
            x_masked,
            x_true,
            max_depth=max_depth,
            n_estimators=max(50, n_estimators // 4),  # cheaper regressor
            lr=lr,
            seed=seed,
        )

    global_marginals = x_train.mean(axis=0).astype(np.float32)
    return build_xgb_agent(
        classifier=clf,
        ev_marginals=ev_marginals,
        schema=schema,
        columns=columns,
        columns_idx=columns_idx,
        thres=stop_thres,
        ig_smoothing=ig_smoothing,
        global_marginals=global_marginals,
    )


def _write_manifest(
    out_dir: Path,
    *,
    model_id: str,
    weights_sha: str,
    train_params: dict[str, Any],
    diseases_trained: list[str],
    columns: list[str],
    feature_importance_top_k: list[dict[str, Any]],
    metrics: EvalMetrics,
    maxstep: int,
) -> Path:
    """Emit ``manifest.json`` with the algorithm-neutral + XGB-specific fields."""
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_id": "ddxplus",
        "model_id": model_id,
        "algorithm_module": "xgb",
        "algorithm_module_version": ALGORITHM_MODULE_VERSION,
        "training_commit": _git_commit(),
        "sha256": weights_sha,
        "train_params": train_params,
        "diseases_trained": diseases_trained,
        "training_target": "pathology",
        "feature_columns": columns,
        "feature_importance_top_k": feature_importance_top_k,
        "eval": {
            **dataclasses.asdict(metrics),
            "maxstep": maxstep,
        },
    }
    path = out_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Train the XGBoost symptom-prediction algorithm on DDXPlus.",
    )
    ap.add_argument("--data-dir", required=True, help="DDXPlus release dir.")
    ap.add_argument(
        "--out-subpath",
        default=None,
        help="Output path under CLARITYMED_HOME/models/symptoms/. "
        "Defaults to ddxplus/run/xgb_<slug>_<timestamp>.",
    )
    ap.add_argument(
        "--diseases",
        default=None,
        help="[v2 subset-conditional] Comma-separated disease names to "
        "WHITELIST — patients outside the list are dropped. Model becomes "
        "``P(disease | disease ∈ whitelist)``. Cannot be combined with "
        "``--targets``.",
    )
    ap.add_argument(
        "--targets",
        default=None,
        help="[v3 subset-parametric] Comma-separated target disease names. "
        "Training keeps ALL patients; every non-target disease is "
        "relabelled as an ``Other`` class. Output has ``N+1`` classes "
        "(the N targets in order, then Other). Model produces absolute "
        "probabilities ``{P(target_1), ..., P(target_N), P(Other)}``. "
        "Enables inverse-frequency class balancing so the initial prior "
        "is uniform over the ``N+1`` buckets.",
    )
    ap.add_argument(
        "--target-cost-multiplier",
        type=float,
        default=DEFAULT_TARGET_COST_MULTIPLIER,
        help="[v3 only] Extra multiplier applied on top of inverse-frequency "
        "balancing to the N target classes. Default 1.0 (pure balancing). "
        "Set >1 when missing a target is clinically costlier than missing "
        "an ``Other`` (e.g. Pneumonia recall matters more than URTI recall).",
    )
    ap.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    ap.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    ap.add_argument("--eval-n-val", type=int, default=DEFAULT_EVAL_N_VAL)
    ap.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    ap.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument(
        "--calibration",
        choices=["platt", "isotonic", "none"],
        default="platt",
        help="Posterior calibration method. Platt (sigmoid) is the "
        "default; isotonic overfits on small validate splits.",
    )
    ap.add_argument(
        "--mask-policy",
        choices=["random", "full"],
        default="random",
        help="'random' trains ev_marginals on masked ↔ full pairs "
        "(matches serving distribution). 'full' skips the marginals "
        "regressor and falls back to global column means at serve "
        "time — smaller checkpoint, less-informed IG picks.",
    )
    ap.add_argument("--keep-rate-lo", type=float, default=DEFAULT_KEEP_LO)
    ap.add_argument("--keep-rate-hi", type=float, default=DEFAULT_KEEP_HI)
    ap.add_argument("--stop-thres", type=float, default=DEFAULT_STOP_THRES)
    ap.add_argument("--ig-smoothing", type=float, default=DEFAULT_IG_SMOOTHING)
    ap.add_argument("--maxstep", type=int, default=DEFAULT_MAXSTEP)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run for sanity testing (2k train, 500 test, small trees).",
    )
    return ap


def main() -> None:
    """CLI: ``uv run --extra symptoms-server claritymed-symptoms-xgb-train-ddxplus ...``."""
    args = _build_argparser().parse_args()

    if args.diseases and args.targets:
        raise SystemExit(
            "--diseases (v2 whitelist) and --targets (v3 subset) are mutually "
            "exclusive. Pick one training mode."
        )

    whitelist: set[str] | None = None
    whitelist_sorted: list[str] = []
    if args.diseases:
        whitelist_sorted = sorted(
            d.strip() for d in args.diseases.split(",") if d.strip()
        )
        whitelist = set(whitelist_sorted)
        if len(whitelist) < 2:
            raise SystemExit(
                f"--diseases needs at least 2 names, got {whitelist_sorted!r}. "
                f"A 1-class model is not a classifier."
            )

    target_names: list[str] = []
    if args.targets:
        target_names = [d.strip() for d in args.targets.split(",") if d.strip()]
        if len(target_names) < 1:
            raise SystemExit(
                "--targets needs at least one name (the trailing Other class "
                "is added automatically)."
            )

    if args.out_subpath is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if target_names:
            slug = "_".join(n.replace(" ", "-").lower()[:6] for n in target_names[:3])
            args.out_subpath = f"ddxplus/run/xgb_v3_{slug}_{ts}"
        elif whitelist:
            slug = "_".join(
                n.replace(" ", "-").lower()[:6] for n in whitelist_sorted[:3]
            )
            args.out_subpath = f"ddxplus/run/xgb_subset_{slug}_{ts}"
        else:
            args.out_subpath = f"ddxplus/run/xgb_v1_{ts}"
    if args.smoke:
        args.n_train = min(args.n_train, 2_000)
        args.n_test = min(args.n_test, 500)
        args.eval_n_val = min(args.eval_n_val, 500)
        args.n_estimators = min(args.n_estimators, 40)
        args.max_depth = min(args.max_depth, 3)

    seed_everything(args.seed)
    t0 = time.time()
    schema = load_evidence_schema(args.data_dir)
    meta = load_evidence_meta(args.data_dir)
    # v2 whitelist mode filters pidx; v3 targets mode loads full pidx and
    # relabels each patient's ``d`` after load. Keep unfiltered pidx around
    # in target mode so we can build the subset mapping deterministically.
    pidx, sev = load_pidx(args.data_dir, whitelist=whitelist)
    subset_mapping: dict[int, int] | None = None
    n_targets: int | None = None
    if target_names:
        subset_mapping, n_dis = _resolve_subset_mapping(target_names, pidx)
        n_targets = len(target_names)
        # Rewrite sev to the N+1 subset space — placeholder value for
        # Other since it's a synthetic bucket, not a real disease.
        sev_subset = np.zeros(n_dis, dtype=np.float32)
        for full_name, full_idx in pidx.items():
            subset_idx = subset_mapping[full_idx]
            if subset_idx < n_targets:
                sev_subset[subset_idx] = sev[full_idx]
        sev_subset[n_targets] = _OTHER_CLASS_SEVERITY
        sev = sev_subset
    else:
        n_dis = len(pidx)
    columns, _labels, columns_idx = feature_columns_from_schema(schema, meta)
    print(
        f"[xgb-train] evidences={schema['n_ev']} diseases={n_dis} "
        f"features={len(columns)}"
        + (f"  [targets={target_names} +Other]" if target_names else "")
        + f"  [{time.time() - t0:.1f}s]"
    )

    train_pats = load_patients(args.data_dir, args.n_train, "train", schema, pidx)
    val_pats = load_patients(args.data_dir, args.eval_n_val, "validate", schema, pidx)
    test_pats = load_patients(args.data_dir, args.n_test, "test", schema, pidx)

    if subset_mapping is not None:
        # v3 relabel: rewrites p["d"] + p["diff"] in-place so downstream
        # `interactive_eval` and per-row weight construction see the N+1
        # class layout, not the original 49-class labels.
        _relabel_patients(train_pats, subset_mapping)
        _relabel_patients(val_pats, subset_mapping)
        _relabel_patients(test_pats, subset_mapping)
        y_train_dbg = np.asarray([p["d"] for p in train_pats])
        counts = np.bincount(y_train_dbg, minlength=n_dis).tolist()
        print(
            "[xgb-train] subset class counts (post-relabel): "
            + ", ".join(
                f"{name}={counts[k]}" for k, name in enumerate(target_names + ["Other"])
            )
        )

    if target_names:
        diseases_trained = target_names + ["Other"]
    else:
        diseases_trained = sorted(pidx, key=lambda n: pidx[n])
    train_params = {
        "n_train_kept": len(train_pats),
        "n_val_kept": len(val_pats),
        "n_test_kept": len(test_pats),
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "lr": args.lr,
        "calibration": args.calibration,
        "mask_policy": args.mask_policy,
        "keep_rate_lo": args.keep_rate_lo,
        "keep_rate_hi": args.keep_rate_hi,
        "stop_thres": args.stop_thres,
        "ig_smoothing": args.ig_smoothing,
        "maxstep": args.maxstep,
        "seed": args.seed,
        "whitelist_active": whitelist is not None,
        "targets": target_names or None,
        "target_cost_multiplier": args.target_cost_multiplier if target_names else None,
        "class_balance": bool(target_names),
    }

    out_dir = _models_dir() / args.out_subpath
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.pkl"
    model_id = Path(args.out_subpath).name

    with symptom_run(
        "ddxplus", run_name=model_id, run_type="xgb_train", params=train_params
    ):
        agent = train_xgb_agent(
            schema=schema,
            train_patients=train_pats,
            val_patients=val_pats,
            columns=columns,
            columns_idx=columns_idx,
            n_classes=n_dis,
            max_depth=args.max_depth,
            n_estimators=args.n_estimators,
            lr=args.lr,
            calibration=args.calibration,
            mask_policy=args.mask_policy,
            keep_lo=args.keep_rate_lo,
            keep_hi=args.keep_rate_hi,
            stop_thres=args.stop_thres,
            ig_smoothing=args.ig_smoothing,
            seed=args.seed,
            class_balance=bool(target_names),
            n_targets=n_targets,
            target_cost_multiplier=args.target_cost_multiplier,
        )
        agent.save(weights_path)
        print(f"[xgb-train] wrote {weights_path}  [{time.time() - t0:.1f}s]")

        print(
            f"[xgb-train] evaluating on {len(test_pats)} test patients "
            f"(maxstep={args.maxstep})..."
        )
        t_eval = time.time()
        metrics: EvalMetrics = interactive_eval(
            TypedEnv(list(test_pats), schema, n_dis),
            agent,
            maxstep=args.maxstep,
            games=len(test_pats),
            severity=sev,
        )
        print(
            f"[xgb-train] TEST (maxstep={args.maxstep}) "
            f"IL={metrics.IL:.2f} ACC={metrics.ACC:.2f} "
            f"DDF1={metrics.DDF1:.2f} DSR={metrics.DSR:.2f}  "
            f"[eval {time.time() - t_eval:.1f}s]"
        )
        log_eval_metrics(metrics, prefix="test/")

        weights_sha = _sha256_file(weights_path)
        feature_importance = _feature_importance_top_k(
            agent.classifier, columns, FEATURE_IMPORTANCE_TOP_K
        )
        manifest_path = _write_manifest(
            out_dir,
            model_id=model_id,
            weights_sha=weights_sha,
            train_params=train_params,
            diseases_trained=diseases_trained,
            columns=columns,
            feature_importance_top_k=feature_importance,
            metrics=metrics,
            maxstep=args.maxstep,
        )
        import mlflow

        mlflow.log_artifact(str(weights_path))
        mlflow.log_artifact(str(manifest_path))
        print(f"[xgb-train] wrote {manifest_path}")


if __name__ == "__main__":
    main()
