"""Inference-time hyperparameter tuning for the promoted BUSI checkpoint.

This phase runs **after** training and reads the staging directory
(``weights.pt`` + ``manifest.json``) the train phase produced. It does
**not** retrain — instead it caches per-image forward-pass outputs
(both vanilla and TTA-averaged) and runs an Optuna study over the
inference-time parameters that materially affect production
performance:

* ``temperature`` — calibration logit scaling (0.5–3.0 log-uniform)
* ``malignant_threshold`` — classification cutoff for the malignant
  top1 selection (0.20–0.70 uniform)
* ``seg_threshold`` — sigmoid cutoff for mask binarization
  (0.20–0.80 uniform)
* ``confidence_low_max`` / ``confidence_medium_max`` — confidence-tier
  boundaries (constraint: ``low_max < medium_max``)
* ``tta_default`` — boolean; when ``True`` the catalog advertises TTA
  as the default and inference uses the averaged logits

The objective is **constraint-aware**: a trial scores ``1 + composite``
(in ``(1, 2]``) when it meets every deploy floor (malignant recall
≥ 0.85, accuracy ≥ 0.85, dice ≥ 0.70), or a negative
"distance-to-feasibility" penalty when it does not. Feasible always
strictly beats infeasible, so Optuna's maximize naturally picks the
best feasible trial when one exists, while infeasible trials still
provide a gradient toward feasibility. The composite itself stays
``0.6 * malignant_recall + 0.4 * dice`` — same as the search phase.

Floors live in :mod:`~claritymed.ingest.vision.busi.deploy` so the gate
and the optimizer share one source of truth. Tune mirrors the constants
locally to avoid a circular import (deploy imports ``_latest_staging_dir``
from this module); a test asserts the two are in sync.

Since per-image logits are cached, each Optuna trial is a cheap numpy
pass — 30+ trials run in well under a minute even on CPU.

After Optuna picks the best trial, the tuned params are written into
``manifest.json::tuned_inference`` and the test-split breakdown is
re-evaluated under those params so the deploy phase has an unbiased
metric for the regression gate. When **no** trial reached feasibility
in the budget, tune still writes the best-available block (so deploy's
gate produces a clean failure with the same floors) and logs a warning
naming the worst-violated floor — the operator's signal that the
checkpoint cannot be tuned into compliance and retraining is needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claritymed import config as _cfg
from claritymed.core.vision.schemas import (
    ConfidenceThresholds,
    Manifest,
    TunedInferenceParams,
)
from claritymed.ingest.mlflow_utils import (
    experiment_name,
    mlflow_run,
    optuna_storage_uri,
    study_name,
    tracking_uri,
)
from claritymed.ingest.vision.busi.dataset import (
    BUSI_LABELS,
    build_dataset,
    discover,
    stratified_split,
)
from claritymed.ingest.vision.busi.download import DATASET_SUBDIR, busi_data_root
from claritymed.ingest.vision.busi.train import DATASET_ID, MODEL_ID

logger = logging.getLogger(__name__)

PHASE = "tune"

# Composite weights + floors + feasibility scoring all live in
# :mod:`scoring` so this module, ``train.py``, ``hparam.py``, and
# ``deploy.py`` share one definition. Anything below that talks about
# "feasible" / "the floor" is parameterised by those constants.
from claritymed.ingest.vision.busi.scoring import (  # noqa: E402
    COMPOSITE_DICE_WEIGHT,
    COMPOSITE_RECALL_WEIGHT,
    FEASIBLE_OFFSET,
    TUNE_FLOORS,
    feasibility_aware_score,
    study_feasibility_summary,
)


def study_id(task_id: str) -> str:
    """Canonical Optuna study name for one pipeline run's tune phase.

    Same per-task naming as :func:`hparam.study_id` so each pipeline
    run gets its own study scoped by ``task_id``; resuming a tune
    requires passing the original task_id.
    """
    return f"{study_name('vision', DATASET_ID, PHASE)}-{task_id}"


def run_tune(*, staging_dir: Path, trials: int, smoke: bool) -> Path:
    """Tune the inference-time params for the model in ``staging_dir``.

    Reads the lineage ``task_id`` from ``staging_dir/provenance.json``
    so MLflow + every Optuna tune trial gets tagged with the same id
    the train phase used. Updates ``staging_dir/manifest.json`` in
    place with the chosen :class:`TunedInferenceParams`, refreshes
    ``staging_dir/eval_metrics.json`` with the tuned test breakdown,
    and appends the tune phase to ``staging_dir/provenance.json``.

    Returns the staging dir for chaining into the deploy phase.
    """
    manifest_path = staging_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} missing — run train.py first or pass a valid "
            f"--staging-dir."
        )

    manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = Manifest.model_validate(manifest_dict)

    task_id = _read_task_id(staging_dir)

    if smoke:
        best, tuned_val, tuned_test = _smoke_tune_result()
        mlflow_info = {"experiment_name": "smoke", "run_id": "smoke"}
    else:
        cache_val = _build_logit_cache(manifest, staging_dir, split="val")
        cache_test = _build_logit_cache(manifest, staging_dir, split="test")
        best, tuned_val, tuned_test, mlflow_info = _run_optuna_study(
            cache_val=cache_val,
            cache_test=cache_test,
            trials=trials,
            task_id=task_id,
        )

    tuned_block = TunedInferenceParams(
        temperature=best["temperature"],
        classification_thresholds={"malignant": best["malignant_threshold"]},
        seg_threshold=best["seg_threshold"],
        confidence_thresholds=ConfidenceThresholds(
            low_max=best["confidence_low_max"],
            medium_max=best["confidence_medium_max"],
        ),
        tta_default=best["tta_default"],
    )

    # Persist the tuned block onto the manifest — schema-only (no
    # eval_metrics splat; those land in eval_metrics.json so manifest.json
    # always roundtrips through Manifest.model_validate).
    updated = _with_tuned_inference(manifest, tuned_block)
    manifest_path.write_text(
        json.dumps(updated.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    eval_path = staging_dir / "eval_metrics.json"
    existing_eval = (
        json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    )
    existing_eval.update(
        {
            "tuned_params": best,
            "tuned_val_breakdown": tuned_val,
            "tuned_test_breakdown": tuned_test,
            "tuned_test_score": tuned_test["composite"],
        }
    )
    eval_path.write_text(json.dumps(existing_eval, indent=2), encoding="utf-8")

    _append_provenance(staging_dir, best=best, mlflow_info=mlflow_info, task_id=task_id)
    return staging_dir


def _read_task_id(staging_dir: Path) -> str:
    """Read the lineage task_id from the staging dir's provenance file.

    Errors are fatal — without a task_id we'd silently break the
    lineage chain, defeating the purpose of having one.
    """
    provenance_path = staging_dir / "provenance.json"
    if not provenance_path.exists():
        raise SystemExit(
            f"{provenance_path} missing — staging dir was not produced by "
            f"a recent train phase. Re-run train.py."
        )
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    task_id = payload.get("task_id")
    if not task_id:
        raise SystemExit(
            f"{provenance_path} has no 'task_id' field — staging dir was "
            f"produced by an older train phase. Re-train with the current "
            f"pipeline."
        )
    return str(task_id)


@dataclass
class _LogitCache:
    """Per-image forward-pass outputs, cached so trials run on numpy alone."""

    labels: list[int]  # ground-truth class index per image
    masks: Any  # numpy uint8 [N, H, W] — binary GT mask
    cls_logits_plain: Any  # numpy float32 [N, num_classes]
    cls_logits_tta: Any  # numpy float32 [N, num_classes]
    seg_probs_plain: Any  # numpy float32 [N, H, W] — sigmoid output
    seg_probs_tta: Any  # numpy float32 [N, H, W]


def _build_logit_cache(
    manifest: Manifest, staging_dir: Path, *, split: str
) -> _LogitCache:
    """Run the trained model once over ``split`` and cache logits + masks.

    Built lazily so smoke paths can skip torch entirely.
    """
    import numpy as np
    import torch

    weights = staging_dir / "weights.pt"
    if not weights.exists():
        raise SystemExit(
            f"{weights} missing — train phase did not persist a checkpoint"
        )

    root = busi_data_root() / DATASET_SUBDIR
    if not root.is_dir():
        raise SystemExit(
            f"BUSI not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.busi.download` first."
        )

    samples = discover(root)
    splits = stratified_split(samples)
    ds = build_dataset(splits[split])

    device = _select_device(torch)
    model = _load_model_from_staging(manifest, weights, device)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=16, shuffle=False, num_workers=2
    )

    cls_plain: list[Any] = []
    cls_tta: list[Any] = []
    seg_plain: list[Any] = []
    seg_tta: list[Any] = []
    labels_out: list[int] = []
    masks_out: list[Any] = []

    with torch.no_grad():
        for imgs, masks, labels in loader:
            imgs = imgs.to(device)
            cls_logits, seg_logits = model(imgs)
            cls_plain.append(cls_logits.cpu().numpy())
            seg_plain.append(seg_logits.sigmoid().cpu().numpy()[:, 0])

            # TTA = horizontal flip; cheap, captures laterality robustness.
            cls_logits_flip, seg_logits_flip = model(torch.flip(imgs, dims=[3]))
            cls_avg = (cls_logits + cls_logits_flip) / 2.0
            seg_avg = (
                seg_logits.sigmoid() + torch.flip(seg_logits_flip.sigmoid(), dims=[3])
            ) / 2.0
            cls_tta.append(cls_avg.cpu().numpy())
            seg_tta.append(seg_avg.cpu().numpy()[:, 0])

            labels_out.extend(int(lbl) for lbl in labels.tolist())
            masks_out.append(masks.cpu().numpy()[:, 0])

    return _LogitCache(
        labels=labels_out,
        masks=np.concatenate(masks_out, axis=0),
        cls_logits_plain=np.concatenate(cls_plain, axis=0),
        cls_logits_tta=np.concatenate(cls_tta, axis=0),
        seg_probs_plain=np.concatenate(seg_plain, axis=0),
        seg_probs_tta=np.concatenate(seg_tta, axis=0),
    )


def _evaluate_cache(cache: _LogitCache, params: dict[str, float]) -> dict[str, float]:
    """Compute the composite + breakdown for one trial's params.

    Pure numpy — no torch, no model forward. Called by every Optuna
    trial; profile-bound by ``len(cache.labels)`` not by trial count.
    """
    import numpy as np

    use_tta = bool(params["tta_default"])
    cls_logits = cache.cls_logits_tta if use_tta else cache.cls_logits_plain
    seg_probs = cache.seg_probs_tta if use_tta else cache.seg_probs_plain

    scaled = cls_logits / float(params["temperature"])
    probs = _softmax(scaled)

    malig_idx = BUSI_LABELS.index("malignant")
    malig_thresh = float(params["malignant_threshold"])
    seg_thresh = float(params["seg_threshold"])

    preds = _argmax_with_threshold(
        probs, malig_idx=malig_idx, malig_threshold=malig_thresh
    )
    labels = np.asarray(cache.labels)

    tp = int(((preds == malig_idx) & (labels == malig_idx)).sum())
    fn = int(((preds != malig_idx) & (labels == malig_idx)).sum())
    correct = int((preds == labels).sum())
    total = int(labels.shape[0])

    recall = tp / max(tp + fn, 1)
    accuracy = correct / max(total, 1)
    pred_masks = (seg_probs > seg_thresh).astype(np.float32)
    dice = _mean_dice(pred_masks, cache.masks)
    composite = COMPOSITE_RECALL_WEIGHT * recall + COMPOSITE_DICE_WEIGHT * dice
    return {
        "composite": composite,
        "malignant_recall": recall,
        "dice": dice,
        "accuracy": accuracy,
    }


def _run_optuna_study(
    *,
    cache_val: _LogitCache,
    cache_test: _LogitCache,
    trials: int,
    task_id: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, str]]:
    """Run Optuna over the inference-param space; return best + breakdowns."""
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna not installed — `uv sync --extra vision-server`"
        ) from exc

    from claritymed.ingest.mlflow_utils import TASK_ID_TAG

    storage = optuna_storage_uri()
    name = study_id(task_id)

    def _objective(trial) -> float:
        trial.set_user_attr("task_id", task_id)
        params = _suggest_params(trial)
        if params["confidence_low_max"] >= params["confidence_medium_max"]:
            # Hard constraint: low_max < medium_max. Pruning is cheaper
            # than catching the validator post-hoc and lets Optuna's
            # acquisition function steer away from infeasible cells.
            raise optuna.TrialPruned()
        breakdown = _evaluate_cache(cache_val, params)
        # Stash the breakdown on the trial so study_feasibility_summary
        # can reconstruct feasibility / deficits without re-evaluating.
        trial.set_user_attr("breakdown", breakdown)
        return feasibility_aware_score(breakdown, TUNE_FLOORS)

    with mlflow_run(
        "vision",
        DATASET_ID,
        run_name=f"tune-{MODEL_ID}",
        run_type="tune",
        params={"trials": trials},
        tags={TASK_ID_TAG: task_id},
    ) as mlflow_handle:
        study = optuna.create_study(
            study_name=name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(_objective, n_trials=trials)
        best_params = dict(study.best_trial.params)
        # Recompute breakdowns once with the winning params (Optuna only
        # stores the composite; we want the full breakdown for the
        # deploy log + eval_metrics).
        tuned_val = _evaluate_cache(cache_val, best_params)
        tuned_test = _evaluate_cache(cache_test, best_params)

        summary = study_feasibility_summary(study, TUNE_FLOORS)
        if study.best_trial.value < FEASIBLE_OFFSET:
            # Every trial violated at least one floor. Tune still writes
            # the best-available block so deploy's gate produces a
            # consistent error with the same floors — but the operator
            # needs to know retuning won't help.
            logger.warning(
                "tune: no feasible trial in %d trials "
                "(feasible=%d, infeasible=%d). Best trial still violates "
                "floors; deploy gate will reject. Worst deficits: %s. "
                "Retrain with stronger model / more epochs.",
                trials,
                summary["feasible_trials"],
                summary["infeasible_trials"],
                summary["worst_deficits"],
            )
        else:
            logger.info(
                "tune: %d feasible trial(s) of %d completed; best val composite=%.4f",
                summary["feasible_trials"],
                summary["feasible_trials"] + summary["infeasible_trials"],
                tuned_val["composite"],
            )
        try:
            import mlflow

            mlflow.log_metrics({f"val/{k}": v for k, v in tuned_val.items()})
            mlflow.log_metrics({f"test/{k}": v for k, v in tuned_test.items()})
            for key, value in best_params.items():
                mlflow.log_param(f"tuned_{key}", value)
        except ImportError:
            pass
        mlflow_info = {
            "experiment_name": experiment_name("vision", DATASET_ID),
            "run_id": mlflow_handle.info.run_id,
            "study_name": name,
            "storage_uri": storage,
        }
    return best_params, tuned_val, tuned_test, mlflow_info


def _suggest_params(trial) -> dict[str, float]:
    """Build one trial's params from the Optuna trial object."""
    return {
        "temperature": trial.suggest_float("temperature", 0.5, 3.0, log=True),
        "malignant_threshold": trial.suggest_float("malignant_threshold", 0.20, 0.70),
        "seg_threshold": trial.suggest_float("seg_threshold", 0.20, 0.80),
        "confidence_low_max": trial.suggest_float("confidence_low_max", 0.40, 0.75),
        "confidence_medium_max": trial.suggest_float(
            "confidence_medium_max", 0.55, 0.95
        ),
        "tta_default": trial.suggest_categorical("tta_default", [False, True]),
    }


def _smoke_tune_result() -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Synthetic tune result so the wiring can be exercised without torch."""
    best = {
        "temperature": 1.2,
        "malignant_threshold": 0.45,
        "seg_threshold": 0.5,
        "confidence_low_max": 0.55,
        "confidence_medium_max": 0.80,
        "tta_default": True,
    }
    breakdown = {
        "composite": 0.60,
        "malignant_recall": 0.88,
        "dice": 0.72,
        "accuracy": 0.85,
    }
    return best, breakdown, breakdown


# --- helpers -------------------------------------------------------------


def _with_tuned_inference(manifest: Manifest, tuned: TunedInferenceParams) -> Manifest:
    """Return a new Manifest with ``tuned_inference`` set.

    Uses ``model_validate`` (not ``model_copy``) so the cross-field
    validator that rejects ``classification_thresholds`` against unknown
    labels actually fires — see CLAUDE.md "Pydantic model_copy vs
    model_validate".
    """
    data = manifest.model_dump(mode="python")
    data["tuned_inference"] = tuned.model_dump(mode="python")
    return Manifest.model_validate(data)


def _softmax(logits):
    import numpy as np

    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _argmax_with_threshold(probs, *, malig_idx: int, malig_threshold: float):
    """Pick argmax but require malignant to clear its threshold to win.

    Mirrors :class:`~claritymed.servers.vision.adapters.busi_unet.BUSIUnetAdapter._top1_under_thresholds`
    so the tune phase's score reflects what the server will actually do
    at inference time.
    """

    raw_argmax = probs.argmax(axis=1)
    # Where raw argmax is malignant but malignant prob < threshold, fall
    # back to the next-best non-malignant class.
    mask = (raw_argmax == malig_idx) & (probs[:, malig_idx] < malig_threshold)
    if not mask.any():
        return raw_argmax
    alt = probs.copy()
    alt[mask, malig_idx] = -1.0
    fallback = alt.argmax(axis=1)
    final = raw_argmax.copy()
    final[mask] = fallback[mask]
    return final


def _mean_dice(pred_masks, gt_masks, eps: float = 1e-6) -> float:
    """Mean per-image dice over the batch.

    Empty-vs-empty (e.g. ``normal`` images) score as 1.0 so the
    aggregate isn't pulled down by classes without lesions.
    """
    import numpy as np

    intersection = (pred_masks * gt_masks).sum(axis=(1, 2))
    union = pred_masks.sum(axis=(1, 2)) + gt_masks.sum(axis=(1, 2))
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(np.mean(dice))


def _select_device(torch):
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model_from_staging(manifest: Manifest, weights: Path, device):
    """Rebuild the trained model architecture and load the checkpoint."""
    import torch

    from claritymed.servers.vision.adapters.busi_unet import build_busi_model

    # The manifest is the authoritative record of the trained backbone.
    # eval_metrics.json is kept as a fallback for legacy staging dirs
    # written before the manifest.backbone field existed.
    backbone = manifest.backbone
    if backbone == "custom_unet":
        eval_path = weights.parent / "eval_metrics.json"
        if eval_path.exists():
            try:
                backbone = (
                    json.loads(eval_path.read_text(encoding="utf-8"))
                    .get("params", {})
                    .get("backbone", backbone)
                )
            except (json.JSONDecodeError, KeyError):
                pass

    model = build_busi_model(
        backbone=backbone, num_classes=len(manifest.labels), pretrained=False
    ).to(device)
    state = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _append_provenance(
    staging_dir: Path,
    *,
    best: dict[str, float],
    mlflow_info: dict[str, str],
    task_id: str,
) -> None:
    """Splice the tune-phase breadcrumb into ``provenance.json``.

    The lineage ``task_id`` is set once at train time; tune writes it
    onto its own block as a check that the value matches what train
    recorded. Mismatch should be impossible because tune reads it
    straight from train's provenance — but recording it again means an
    operator inspecting only the ``tune`` block can still see the id.
    """
    path = staging_dir / "provenance.json"
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    payload["tune"] = {
        "task_id": task_id,
        "params": best,
        "mlflow": {
            "tracking_uri": tracking_uri(),
            "experiment_name": mlflow_info.get("experiment_name"),
            "tune_run_id": mlflow_info.get("run_id"),
        },
        "optuna": {
            "storage_uri": optuna_storage_uri(),
            "tune_study_name": study_id(task_id),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _latest_staging_dir() -> Path:
    """Pick the most recent timestamped staging dir under the disease run dir."""
    run_dir = _cfg.CLARITYMED_HOME / "models" / "vision" / DATASET_ID / "run"
    candidates = sorted(
        (p for p in run_dir.glob(f"{MODEL_ID}_*") if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            f"no staging dirs under {run_dir} — run train.py first or pass --staging-dir."
        )
    return candidates[0]


# --- CLI -----------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Path to the train-phase staging dir. Defaults to the most recent.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Optuna trial count for the inference-param sweep.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Skip torch + Optuna; synthesize tuned params so wiring can be checked.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    staging = args.staging_dir or _latest_staging_dir()
    out = run_tune(staging_dir=staging, trials=args.trials, smoke=args.smoke)
    print(f"tuned {out}")
    return 0


def cli() -> None:  # pragma: no cover
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    cli()
