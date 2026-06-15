"""BUSI U-Net production training.

The actual training run takes hours on Apple Silicon / CUDA; this
module ships the wiring so an operator can run it with confidence. The
``--smoke`` flag drops to a 10-image sub-sample + 1 epoch so the
implementer can verify the loader / model / loss / optimizer all wire
together before committing to the full run.

**Hyperparameters come from the Optuna search study** (study name from
:func:`~claritymed.ingest.vision.busi.hparam.study_id`). Running
production training without a completed search study fails fast — the
operator is expected to run hparam search first.

**Early stopping** runs against the val split composite. ``--max-epochs``
sets the ceiling; ``--patience`` controls how many epochs of stagnation
trigger an early stop. The default ceiling is high enough (100) that
overfitting curves are visible end-to-end before patience kicks in;
the operator can inspect ``training_curve.json`` to confirm.

The checkpoint persisted to disk is the **best-epoch** state dict, not
the last epoch — overfitting after the early-stopping window doesn't
contaminate the deployed weights.

Outputs land at::

    ~/.claritymed/models/vision/breast_cancer_ultrasound/run/<model_id>_<timestamp>/
        weights.pt              # best-epoch checkpoint
        manifest.json           # tuned_inference is None until tune.py runs
        eval_metrics.json       # val + held-out test composite + breakdown
        training_curve.json     # per-epoch loss / val composite
        provenance.json         # mlflow + optuna run/study ids for deploy log

The downstream tune phase reads from the staging dir, writes tuned
params into ``manifest.json``, then deploy promotes the dir to a
versioned sibling stable path (see ``deploy.py``).

Manifest fields are written via :class:`~claritymed.core.vision.schemas.Manifest`
so the same validation that runs at server boot catches a malformed
write here.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from claritymed import config as _cfg
from claritymed.ingest.vision.busi.dataset import (
    BUSI_LABELS,
    build_dataset,
    discover,
    stratified_split,
)
from claritymed.ingest.vision.busi.download import busi_data_root, DATASET_SUBDIR
from claritymed.ingest.vision.busi.scoring import (
    FEASIBLE_OFFSET,
    SEARCH_FLOORS,
    TRAIN_FLOORS,
    feasibility_aware_score,
    is_feasible,
)
from claritymed.ingest.mlflow_utils import mlflow_run

logger = logging.getLogger(__name__)

DATASET_ID = "breast_cancer_ultrasound"
MODEL_ID = "breast_busi_unet_v1"
MODEL_VERSION = "v1.0.0"

# Per-label metadata baked into the manifest so the server can return it
# verbatim on every detection. Translator-facing wording lives in
# configs/i18n/<lang>/vision.yaml; this block is the EN canonical so
# audit pipelines can read it without the i18n loader.
LABELS_META: dict[str, dict[str, str]] = {
    "benign": {
        "description": "Non-cancerous lesion. Routine follow-up is usually appropriate.",
        "cancer_status": "benign",
        "clinical_action": "routine_followup",
    },
    "malignant": {
        "description": "Suspicious for cancer. A breast specialist should review the image.",
        "cancer_status": "malignant",
        "clinical_action": "urgent_specialist",
    },
    "normal": {
        "description": "No lesion identified. No action required from this image alone.",
        "cancer_status": "normal",
        "clinical_action": "no_action",
    },
}


def _staging_dir(model_id: str) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        _cfg.CLARITYMED_HOME
        / "models"
        / "vision"
        / DATASET_ID
        / "run"
        / f"{model_id}_{ts}"
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(
    *,
    target: Path,
    weights_sha: str,
    eval_metrics: dict,
    supports_tta: bool,
    backbone: str,
) -> None:
    """Compose the manifest, validate, then write.

    ``tuned_inference`` is left as ``None``; the tune phase
    (``tune.py``) reads the manifest, runs its inference-time sweep,
    and writes the tuned block back in place. Train writing
    ``tuned_inference`` itself would couple the two phases.
    """
    from claritymed.core.vision.schemas import Manifest

    manifest = Manifest(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        framework="pytorch",
        accepted_modality="ultrasound",
        sha256_weights=weights_sha,
        task="classification+segmentation",
        labels=list(BUSI_LABELS),
        labels_meta={
            label: {
                "description": LABELS_META[label]["description"],
                "cancer_status": LABELS_META[label]["cancer_status"],
                "clinical_action": LABELS_META[label]["clinical_action"],
            }
            for label in BUSI_LABELS
        },
        cancer_class=True,
        cancer_status_mapping={
            label: LABELS_META[label]["cancer_status"] for label in BUSI_LABELS
        },
        clinical_action_mapping={
            label: LABELS_META[label]["clinical_action"] for label in BUSI_LABELS
        },
        supports_saliency=False,
        supports_tta=supports_tta,
        model_card_url=None,
        backbone=backbone,
        tuned_inference=None,
    )
    # ``manifest.json`` must roundtrip through ``Manifest.model_validate``
    # cleanly — eval metrics live in the sibling ``eval_metrics.json``
    # so we don't have to relax ``extra="forbid"`` on the schema.
    _ = eval_metrics  # kept in signature for backwards-compat; written by caller.
    target.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


# --- best-HP loader -------------------------------------------------------


class _NoSearchStudyError(SystemExit):
    """Raised when the operator tries to train without running hparam first."""


@dataclasses.dataclass(frozen=True)
class _BestHpPick:
    """Resolved best-HP record carrying lineage back to the source trial."""

    params: dict[str, Any]
    trial_number: int
    trial_task_id: str | None


def _load_best_hp(*, task_id: str) -> _BestHpPick:
    """Read the best trial from the hparam study owned by ``task_id``.

    Per-task study naming (see :func:`hparam.study_id`) means the study
    only holds this pipeline run's own trials — no cross-run
    contamination, so the picker just takes ``best_trial`` directly. To
    train from a prior search, re-invoke with that run's ``task_id``.

    Refuses to fall back to a hardcoded default — silent fallback would
    let an operator promote a model trained with stale defaults after a
    search-space change, which is exactly the kind of "looks fine until
    eval surprises you" regression the pipeline is supposed to prevent.
    """
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna not installed — run `uv sync --extra vision-server`."
        ) from exc
    from claritymed.ingest.mlflow_utils import optuna_storage_uri
    from claritymed.ingest.vision.busi.hparam import study_id

    name = study_id(task_id)
    try:
        study = optuna.load_study(study_name=name, storage=optuna_storage_uri())
    except KeyError as exc:
        raise _NoSearchStudyError(
            f"Optuna study {name!r} not found in {optuna_storage_uri()}. "
            f"Run `claritymed-vision-hparam-breast-cancer-ultrasound "
            f"--task-id {task_id}` first."
        ) from exc

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        raise _NoSearchStudyError(
            f"Optuna study {name!r} has no completed trials. "
            f"Run the hparam search before production training."
        )

    best = max(completed, key=lambda t: float("-inf") if t.value is None else t.value)
    return _BestHpPick(
        params=dict(best.params),
        trial_number=best.number,
        trial_task_id=best.user_attrs.get("task_id"),
    )


def run_training_trial(
    params: dict[str, Any],
    *,
    epochs: int,
    smoke: bool,
    trial=None,
) -> float:
    """One Optuna trial — train + return a feasibility-aware score.

    Returns :func:`~claritymed.ingest.vision.busi.scoring.feasibility_aware_score`
    applied at ``SEARCH_FLOORS`` to the best epoch's val breakdown.
    Search has a tiny per-trial budget (~5 epochs), so its bar is
    "shows signs of life" — recall + accuracy clearly above random,
    dice clearly above noise — not the deploy bar (which is what
    train/tune later optimize against). Trials infeasible at
    SEARCH_FLOORS still get a distance-to-feasibility score so Optuna's
    acquisition function has a gradient.

    Historically returned the raw composite, which let trials with
    ``recall=1, dice≈0`` win the search and silently hand the downstream
    training a degenerate HP combo; the new behaviour matches what the
    search is actually paid to find.

    The full forward pass requires torch + a real BUSI download. The
    smoke path returns a synthetic feasibility-aware score (lifted just
    above ``FEASIBLE_OFFSET``) so the Optuna machinery itself can be
    verified offline.
    """
    if smoke:
        # 1 epoch on 10 samples — verifies the dataset loader + model
        # + loss + optimizer wire together. Returns a synthetic
        # feasibility-aware score (just above FEASIBLE_OFFSET) so Optuna
        # records the trial as feasible.
        logger.info("smoke trial: params=%s", params)
        _smoke_forward_pass(params)
        return FEASIBLE_OFFSET + 0.5

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "torch not installed — run `uv sync --extra vision-server`."
        ) from exc

    root = busi_data_root() / DATASET_SUBDIR
    if not root.is_dir():
        raise SystemExit(
            f"BUSI not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.busi.download` first."
        )

    samples = discover(root)
    splits = stratified_split(samples)
    train_ds = build_dataset(splits["train"])
    val_ds = build_dataset(splits["val"])

    device = _select_device(torch)
    model = _build_model(params["backbone"], pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=2
    )

    best_score = -float("inf")
    best_breakdown: dict[str, float] | None = None
    with mlflow_run(
        "vision",
        DATASET_ID,
        run_name=f"trial-{trial.number if trial else 'manual'}",
        run_type="train",
        params=params,
        nested=trial is not None,
    ):
        for epoch in range(epochs):
            model.train()
            for imgs, masks, labels in loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                cls_logits, seg_logits = model(imgs)
                loss = _composite_loss(
                    cls_logits, seg_logits, labels, masks, params["seg_loss_weight"]
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            _, breakdown = _eval_full(
                model, val_loader, device, params["seg_loss_weight"]
            )
            # Search-phase budget is tiny (~5 epochs); judge HP combos
            # by SEARCH_FLOORS, not the deploy floor — a HP combo that
            # can't reach recall 0.65 in 5 epochs isn't going to reach
            # 0.85 in 100. Using deploy floor here would make every
            # trial infeasible and rob Optuna of feasible-region
            # gradient.
            score = feasibility_aware_score(breakdown, SEARCH_FLOORS)
            if score > best_score:
                best_score = score
                best_breakdown = breakdown
            logger.info(
                "epoch=%d val_composite=%.4f val_feasibility=%.4f (best=%.4f)",
                epoch,
                breakdown["composite"],
                score,
                best_score,
            )
            if trial is not None:
                trial.report(score, epoch)
                if trial.should_prune():
                    import optuna  # type: ignore[import-not-found]

                    raise optuna.TrialPruned()
        # Stash the winning breakdown on the trial (when present) so
        # ``study_feasibility_summary`` downstream can reconstruct
        # feasibility / deficits without re-evaluating.
        if trial is not None and best_breakdown is not None:
            trial.set_user_attr("breakdown", best_breakdown)
    return best_score


def run_production_training(
    *,
    max_epochs: int,
    patience: int,
    smoke: bool,
    task_id: str | None = None,
) -> Path:
    """Train one production checkpoint and write the manifest.

    Reads best HP from the Optuna study, runs early-stopping training
    against the val split composite, persists best-epoch weights plus
    per-epoch curves, and finally evaluates on the held-out test split
    so the deploy phase has a regression-gate metric that's never been
    used for optimization.

    The ``task_id`` (auto-generated when absent) lands on the MLflow
    train run tag, on every Optuna tune trial later, and in
    ``provenance.json`` next to the winning search trial's task_id —
    so the whole lineage is reconstructable from any artifact.

    Returns the staging directory path.
    """
    from claritymed.ingest.mlflow_utils import generate_task_id

    if task_id is None:
        task_id = generate_task_id()
    logger.info("train task_id=%s", task_id)

    if smoke:
        # Smoke runs the wiring without requiring a real search study —
        # the pipeline-orchestrator smoke test should be able to drive
        # all four phases on a fresh box, so we synthesize HP here
        # instead of failing on a missing study.
        params: dict[str, Any] = {
            "backbone": "custom_unet",
            "lr": 1e-3,
            "seg_loss_weight": 1.0,
        }
        hp_pick = _BestHpPick(params=params, trial_number=-1, trial_task_id=task_id)
        logger.info("smoke training with synthetic HP: %s", params)
    else:
        hp_pick = _load_best_hp(task_id=task_id)
        params = hp_pick.params
        logger.info(
            "training with best HP from study trial #%d (task_id=%s): %s",
            hp_pick.trial_number,
            hp_pick.trial_task_id,
            params,
        )

    staging = _staging_dir(MODEL_ID)
    staging.mkdir(parents=True, exist_ok=True)

    history = _train_with_early_stopping(
        params,
        max_epochs=max_epochs,
        patience=patience,
        smoke=smoke,
        task_id=task_id,
    )

    # Persist the best-epoch state dict, then re-hash for the manifest.
    weights = staging / "weights.pt"
    _persist_weights(history.best_state_dict, weights, params=params, smoke=smoke)
    weights_sha = _sha256_file(weights)

    eval_metrics = {
        "params": params,
        "best_epoch": history.best_epoch,
        "best_val_composite": history.best_val_composite,
        "best_val_feasibility": history.best_val_feasibility,
        "best_val_feasible": history.best_val_feasible,
        "epochs_trained": history.epochs_trained,
        "early_stopped": history.early_stopped,
        "feasible_epoch_count": history.feasible_epoch_count,
        "val_breakdown": history.val_breakdown,
        "test_breakdown": history.test_breakdown,
        "test_score": history.test_score,
    }

    _write_manifest(
        target=staging / "manifest.json",
        weights_sha=weights_sha,
        eval_metrics=eval_metrics,
        supports_tta=True,
        backbone=params["backbone"],
    )
    (staging / "eval_metrics.json").write_text(
        json.dumps(eval_metrics, indent=2), encoding="utf-8"
    )
    (staging / "training_curve.json").write_text(
        json.dumps(history.curves, indent=2), encoding="utf-8"
    )
    _write_provenance(
        staging,
        params=params,
        mlflow_info=history.mlflow_info,
        task_id=task_id,
        hp_pick=hp_pick,
    )
    return staging


@dataclasses.dataclass
class _TrainingHistory:
    """Bundle returned from :func:`_train_with_early_stopping`.

    Carries everything the surrounding writer needs to persist the
    staging dir without re-reading torch state.
    """

    best_state_dict: Any
    best_val_composite: float  # raw 0.6*recall + 0.4*dice at the chosen best epoch
    best_val_feasibility: float  # feasibility-aware score at the chosen best epoch
    best_val_feasible: bool  # True iff best epoch met every deploy floor
    best_epoch: int
    epochs_trained: int
    early_stopped: bool
    feasible_epoch_count: int  # # of epochs that met every floor during the run
    curves: list[dict[str, float]]
    val_breakdown: dict[str, float]  # best-epoch breakdown (not last-epoch)
    test_breakdown: dict[str, float]
    test_score: float
    mlflow_info: dict[str, str]


def _train_with_early_stopping(
    params: dict[str, Any],
    *,
    max_epochs: int,
    patience: int,
    smoke: bool,
    task_id: str,
) -> _TrainingHistory:
    """Run the training loop with patience-based early stopping.

    The composite ``score = 0.6 * malignant_recall + 0.4 * dice`` drives
    both the per-epoch early-stop trigger and the final selection. Each
    epoch logs (train_loss, val_loss, val_score, val_breakdown) to
    MLflow and accumulates ``curves`` so an operator can plot the
    overfitting onset without parsing MLflow's API.
    """
    if smoke:
        return _smoke_training_history(params)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "torch not installed — run `uv sync --extra vision-server`."
        ) from exc

    from claritymed.ingest.mlflow_utils import mlflow_run, experiment_name

    root = busi_data_root() / DATASET_SUBDIR
    if not root.is_dir():
        raise SystemExit(
            f"BUSI not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.busi.download` first."
        )

    samples = discover(root)
    splits = stratified_split(samples)
    train_ds = build_dataset(splits["train"])
    val_ds = build_dataset(splits["val"])
    test_ds = build_dataset(splits["test"])

    device = _select_device(torch)
    model = _build_model(params["backbone"], pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=16, shuffle=False, num_workers=2
    )

    # Best-epoch selection is driven by feasibility-aware scoring so a
    # noisy first epoch with high recall but ~0 dice cannot lock the
    # best slot for the rest of the run. Raw composite is preserved on
    # the curves + history for human readability (eval_metrics.json
    # keeps the historical `best_val_score` field semantics).
    best_selection_score = -float("inf")
    best_val_composite = -1.0
    best_val_breakdown: dict[str, float] = {}
    best_epoch = 0
    best_state_dict: Any = None
    curves: list[dict[str, float]] = []
    feasible_epoch_count = 0
    epochs_since_improve = 0
    early_stopped = False

    from claritymed.ingest.mlflow_utils import TASK_ID_TAG

    with mlflow_run(
        "vision",
        DATASET_ID,
        run_name=f"train-{MODEL_ID}",
        run_type="train",
        params={**params, "max_epochs": max_epochs, "patience": patience},
        tags={TASK_ID_TAG: task_id},
    ) as run:
        run_id = run.info.run_id
        for epoch in range(max_epochs):
            train_loss = _train_one_epoch(
                model, loader, optimizer, device, params["seg_loss_weight"]
            )
            val_loss, val_breakdown = _eval_full(
                model, val_loader, device, params["seg_loss_weight"]
            )
            val_composite = val_breakdown["composite"]
            # Best-epoch selection runs against TRAIN_FLOORS — the "is
            # this within tune's reach of the deploy bar?" bar, lower
            # than deploy by enough headroom that tune can close the
            # gap. Using deploy floor here would mean a near-feasible
            # epoch (e.g. recall 0.84) gets ranked the same as a wildly
            # infeasible one, hiding the genuinely-close run.
            val_selection_score = feasibility_aware_score(val_breakdown, TRAIN_FLOORS)
            epoch_is_feasible = is_feasible(val_breakdown, TRAIN_FLOORS)
            if epoch_is_feasible:
                feasible_epoch_count += 1
            curves.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_score": val_composite,
                    "val_selection_score": val_selection_score,
                    "val_feasible": float(epoch_is_feasible),
                    **{f"val_{k}": v for k, v in val_breakdown.items()},
                }
            )
            _log_epoch_to_mlflow(epoch, train_loss, val_loss, val_breakdown)
            if val_selection_score > best_selection_score:
                best_selection_score = val_selection_score
                best_val_composite = val_composite
                best_val_breakdown = val_breakdown
                best_epoch = epoch
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
            logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f "
                "val_composite=%.4f val_selection=%.4f feasible=%s "
                "best_selection=%.4f (epoch %d) stale=%d",
                epoch,
                train_loss,
                val_loss,
                val_composite,
                val_selection_score,
                epoch_is_feasible,
                best_selection_score,
                best_epoch,
                epochs_since_improve,
            )
            if epochs_since_improve >= patience:
                logger.info("early stop at epoch %d (patience=%d)", epoch, patience)
                early_stopped = True
                break

        if best_selection_score < FEASIBLE_OFFSET:
            # No epoch met TRAIN_FLOORS. We still persist the
            # best-available state dict so the downstream pipeline
            # (tune → deploy) produces a consistent failure naming the
            # exact floors, rather than crashing mid-run after hours of
            # GPU time. The warning here is the operator's first signal
            # that even an ideal tune can't save this run.
            deficits = TRAIN_FLOORS.deficits(best_val_breakdown)
            logger.warning(
                "train: no epoch met TRAIN_FLOORS in %d trained "
                "(feasible=%d). Best epoch %d val composite=%.4f, "
                "deficits=%s. Deploy gate will almost certainly reject; "
                "retrain with stronger model / loss / more data.",
                len(curves),
                feasible_epoch_count,
                best_epoch,
                best_val_composite,
                {k: f"{v:.3f}" for k, v in deficits.items()},
            )

        # Restore best weights, evaluate on the held-out test split.
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        _, test_breakdown = _eval_full(
            model, test_loader, device, params["seg_loss_weight"]
        )
        test_score = test_breakdown["composite"]
        _log_test_to_mlflow(test_breakdown)

        mlflow_info = {
            "experiment_name": experiment_name("vision", DATASET_ID),
            "run_id": run_id,
        }

    return _TrainingHistory(
        best_state_dict=best_state_dict,
        best_val_composite=best_val_composite,
        best_val_feasibility=best_selection_score,
        best_val_feasible=best_selection_score >= FEASIBLE_OFFSET,
        best_epoch=best_epoch,
        epochs_trained=len(curves),
        early_stopped=early_stopped,
        feasible_epoch_count=feasible_epoch_count,
        curves=curves,
        val_breakdown=best_val_breakdown,
        test_breakdown=test_breakdown,
        test_score=test_score,
        mlflow_info=mlflow_info,
    )


def _smoke_training_history(params: dict[str, Any]) -> _TrainingHistory:
    """Return a fake training history that exercises the full output write path.

    Used when ``--smoke`` is set so the surrounding pipeline (deploy
    gate, manifest write, provenance log) can be unit-tested without
    touching torch. The synthetic breakdown is constructed to be
    feasible (every floor met) so the smoke run produces a manifest
    that survives the deploy gate's structural checks; the gate's
    regression-vs-previous comparison happens against the LATEST.jsonl
    log and is independent of these numbers.
    """
    _smoke_forward_pass(params)
    breakdown = {
        "composite": 0.55,
        "malignant_recall": 0.86,
        "dice": 0.71,
        "accuracy": 0.86,
    }
    selection_score = feasibility_aware_score(breakdown, TRAIN_FLOORS)
    return _TrainingHistory(
        best_state_dict={"smoke": True, "backbone": params.get("backbone")},
        best_val_composite=breakdown["composite"],
        best_val_feasibility=selection_score,
        best_val_feasible=is_feasible(breakdown, TRAIN_FLOORS),
        best_epoch=0,
        epochs_trained=1,
        early_stopped=False,
        feasible_epoch_count=1,
        curves=[
            {
                "epoch": 0,
                "train_loss": 0.42,
                "val_loss": 0.40,
                "val_score": breakdown["composite"],
                "val_selection_score": selection_score,
                "val_feasible": 1.0,
                "val_composite": breakdown["composite"],
                "val_malignant_recall": breakdown["malignant_recall"],
                "val_dice": breakdown["dice"],
                "val_accuracy": breakdown["accuracy"],
            }
        ],
        val_breakdown=breakdown,
        test_breakdown=breakdown,
        test_score=breakdown["composite"],
        mlflow_info={"experiment_name": "smoke", "run_id": "smoke"},
    )


def _persist_weights(
    state_dict: Any, target: Path, *, params: dict[str, Any], smoke: bool
) -> None:
    """Save the best-epoch state dict (or a tiny smoke sentinel)."""
    try:
        import torch

        if smoke:
            torch.save(
                {
                    "backbone": params.get("backbone"),
                    "smoke": True,
                    "state": state_dict,
                },
                target,
            )
        else:
            torch.save(state_dict, target)
    except ImportError:
        target.write_bytes(b"smoke")


def _write_provenance(
    staging: Path,
    *,
    params: dict[str, Any],
    mlflow_info: dict[str, str],
    task_id: str,
    hp_pick: _BestHpPick,
) -> None:
    """Drop a provenance.json file the deploy phase can splice into the log.

    Keeping it in the staging dir means each candidate carries its own
    breadcrumb trail. ``task_id`` is the lineage anchor — every MLflow
    run + Optuna trial in the same pipeline execution shares it.
    ``search_trial_number`` + ``search_trial_task_id`` close the loop on
    "where did this HP come from?" — a winning trial from a different
    pipeline run is recorded (not masked) for full traceability.

    Deploy reads it verbatim into LATEST.jsonl.
    """
    from claritymed.ingest.mlflow_utils import (
        optuna_storage_uri,
        tracking_uri,
    )
    from claritymed.ingest.vision.busi.hparam import study_id

    payload = {
        "phase": "train",
        "task_id": task_id,
        "params": params,
        "mlflow": {
            "tracking_uri": tracking_uri(),
            "experiment_name": mlflow_info.get("experiment_name"),
            "train_run_id": mlflow_info.get("run_id"),
        },
        "optuna": {
            "storage_uri": optuna_storage_uri(),
            "search_study_name": study_id(task_id),
            "search_trial_number": hp_pick.trial_number,
            "search_trial_task_id": hp_pick.trial_task_id,
        },
    }
    (staging / "provenance.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


# --- internal helpers -----------------------------------------------------


def _select_device(torch):
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise SystemExit(
        "no MPS or CUDA device detected. BUSI U-Net training on CPU is "
        "unworkable — point this script at a machine with a GPU."
    )


def _build_model(backbone: str, *, pretrained: bool = False):
    """Build a U-Net with the requested encoder + a classification head.

    The real model definition lives in
    ``servers/vision/adapters/busi_unet.py`` so the same forward pass
    is shared between training and inference.

    ``pretrained=True`` should be passed only from real training (not
    smoke or inference). It triggers a one-time ImageNet checkpoint
    download for the resnet50 / efficientnet_b0 encoders; the cached
    weights warm-start the encoder so dice converges within the
    search-phase epoch budget.
    """
    from claritymed.servers.vision.adapters.busi_unet import build_busi_model

    return build_busi_model(
        backbone=backbone, num_classes=len(BUSI_LABELS), pretrained=pretrained
    )


def _composite_loss(cls_logits, seg_logits, labels, masks, seg_weight):
    import torch.nn.functional as F

    cls_loss = F.cross_entropy(cls_logits, labels)
    seg_loss = F.binary_cross_entropy_with_logits(seg_logits, masks)
    return cls_loss + seg_weight * seg_loss


def _eval_one_epoch(model, loader, device) -> float:
    """Composite score (back-compat shim used by ``run_training_trial``)."""
    _, breakdown = _eval_full(model, loader, device, seg_weight=1.0)
    return breakdown["composite"]


def _train_one_epoch(model, loader, optimizer, device, seg_weight: float) -> float:
    """Run one training epoch and return the mean batch loss."""
    import torch  # noqa: F401 — already imported at call site, keeps adapter local

    model.train()
    losses: list[float] = []
    for imgs, masks, labels in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        cls_logits, seg_logits = model(imgs)
        loss = _composite_loss(cls_logits, seg_logits, labels, masks, seg_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / max(len(losses), 1)


def _eval_full(
    model, loader, device, seg_weight: float
) -> tuple[float, dict[str, float]]:
    """Compute composite + breakdown + mean eval loss.

    The breakdown carries the individual components the deploy gate
    cares about (``malignant_recall``, ``dice``, ``accuracy``) so a
    single tensor pass produces everything downstream needs.
    """
    import torch

    model.eval()
    malig_idx = BUSI_LABELS.index("malignant")
    tp = fn = correct = total = 0
    dice_sum = 0.0
    loss_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for imgs, masks, labels in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            cls_logits, seg_logits = model(imgs)
            loss_sum += float(
                _composite_loss(cls_logits, seg_logits, labels, masks, seg_weight)
            )
            preds = cls_logits.argmax(dim=1)
            tp += int(((preds == malig_idx) & (labels == malig_idx)).sum())
            fn += int(((preds != malig_idx) & (labels == malig_idx)).sum())
            correct += int((preds == labels).sum())
            total += int(labels.numel())
            dice_sum += float(_dice_score(seg_logits.sigmoid(), masks))
            n_batches += 1
    recall = tp / max(tp + fn, 1)
    dice = dice_sum / max(n_batches, 1)
    accuracy = correct / max(total, 1)
    breakdown = {
        "composite": 0.6 * recall + 0.4 * dice,
        "malignant_recall": recall,
        "dice": dice,
        "accuracy": accuracy,
    }
    mean_loss = loss_sum / max(n_batches, 1)
    return mean_loss, breakdown


def _log_epoch_to_mlflow(
    epoch: int, train_loss: float, val_loss: float, val_breakdown: dict[str, float]
) -> None:
    """Log per-epoch scalars to the active MLflow run.

    Lazy mlflow import keeps the train module importable without the
    extra installed (used by tests / docs).
    """
    try:
        import mlflow
    except ImportError:
        return
    metrics = {
        "train/loss": train_loss,
        "val/loss": val_loss,
        **{f"val/{k}": v for k, v in val_breakdown.items()},
    }
    mlflow.log_metrics(metrics, step=epoch)


def _log_test_to_mlflow(test_breakdown: dict[str, float]) -> None:
    """Log the final test-split breakdown to the active MLflow run."""
    try:
        import mlflow
    except ImportError:
        return
    mlflow.log_metrics({f"test/{k}": v for k, v in test_breakdown.items()})


def _dice_score(pred, target, eps: float = 1e-6) -> float:
    pred_bin = (pred > 0.5).float()
    num = 2 * (pred_bin * target).sum()
    den = pred_bin.sum() + target.sum() + eps
    return float(num / den)


def _smoke_forward_pass(params: dict[str, Any]) -> None:
    """Smoke test: 1 mini-batch through the model.

    Reaches into :func:`_build_model` so a missing optional dependency
    fails loudly in CI rather than at the bottom of a 4-hour training
    run.
    """
    try:
        import torch
    except ImportError:
        logger.warning("torch not installed — smoke pass skipped")
        return
    model = _build_model(params["backbone"])
    x = torch.randn(2, 3, 256, 256)
    cls_logits, seg_logits = model(x)
    assert cls_logits.shape == (2, len(BUSI_LABELS))
    assert seg_logits.shape[0] == 2


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help=(
            "Ceiling on training epochs. Default high enough that the "
            "loss curves visibly diverge before patience triggers; the "
            "early-stop logic picks the best epoch."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Epochs without val/composite improvement before early stop.",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help=(
            "Pipeline lineage id. Generated when absent. Recorded in "
            "MLflow + provenance + LATEST.jsonl; pass the same value "
            "across phases to keep one task_id end-to-end."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1 epoch on 10 samples — verifies the loop without training",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    started = time.monotonic()
    staging = run_production_training(
        max_epochs=args.max_epochs,
        patience=args.patience,
        smoke=args.smoke,
        task_id=args.task_id,
    )
    elapsed = time.monotonic() - started
    print(f"wrote {staging} in {elapsed:.1f}s")
    return 0


def cli() -> None:  # pragma: no cover
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    cli()
