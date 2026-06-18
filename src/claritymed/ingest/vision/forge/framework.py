"""Forge phases — task-polymorphic search / train / tune / deploy / pipeline.

Every public function in this module accepts a
:class:`~claritymed.ingest.vision.forge.spec.ModelSpec` and reaches the
task / dataset / floors through that one handle. No dataset-specific
logic lives here.

Phase contracts (mirror the BUSI pipeline this replaces):

* :func:`run_hparam` — Optuna search over ``spec.hparam_space``.
  Each trial gets a tiny per-trial budget (``epochs``) so we judge
  HP combos against ``spec.task.phase_floors("search")`` — the
  "shows signs of life" bar, not the deploy bar.
* :func:`run_train` — production training with patience-based early
  stop. Reads the best HP from the search study. Writes
  ``weights.pt`` + ``manifest.json`` (tuned_inference=None) +
  ``eval_metrics.json`` + ``training_curve.json`` + ``provenance.json``
  into a fresh staging dir.
* :func:`run_tune` — Optuna over inference-time params (no retrain).
  Caches model forward outputs once, then iterates trials over the
  cache in pure numpy. Writes the chosen
  :class:`~claritymed.core.vision.schemas.TunedInferenceParams` back
  into the staging manifest.
* :func:`run_deploy` — floor + regression gate + versioned promote +
  surgical ``configs/vision.yaml`` edit + ``LATEST.jsonl`` append.
* :func:`run_pipeline` — orchestrates all four phases in order with
  inter-phase floor gates. A single ``task_id`` threads through every
  artifact (MLflow tags / Optuna user_attrs / provenance.json /
  LATEST.jsonl row) so the lineage is reconstructable from any one of
  them.

Smoke mode (``smoke=True``) wires every phase end-to-end on synthetic
data so the pipeline can be CI-tested without GPU + real Kaggle
download. The Task subclasses provide feasible-by-construction smoke
breakdowns so deploy clears the floor gate. Smoke deploy intentionally
**stops at the floor check** — it does not promote the staging dir to
the disease's stable path, update the ``<model_id>`` stable symlink,
append to ``LATEST.jsonl``, patch ``configs/vision.yaml``, or run the
regression gate. The whole point is that a wiring check must not leave
fake state the vision-server would later try to load, nor poison the
next smoke's regression gate with its own previous output.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from claritymed.core.vision.schemas import Manifest
from claritymed.ingest.mlflow_utils import (
    TASK_ID_TAG,
    experiment_name,
    generate_task_id,
    mlflow_run,
    optuna_storage_uri,
    study_name,
    tracking_uri,
)
from claritymed.ingest.vision.forge.common import (
    ALL_PHASES,
    FEATURE,
    append_latest_entry,
    check_floors,
    assert_vision_yaml_has_model,
    disease_root,
    latest_jsonl_path,
    latest_staging_dir,
    log_metrics,
    patch_vision_yaml,
    read_latest_entry,
    select_device,
    sha256_file,
    staging_dir,
    version_tag,
)
from claritymed.ingest.vision.forge.scoring import (
    FEASIBLE_OFFSET,
    feasibility_aware_score,
    gate_or_raise,
    is_feasible,
    study_feasibility_summary,
)
from claritymed.ingest.vision.forge.spec import ModelSpec

logger = logging.getLogger(__name__)


# === Lineage / study naming =============================================


def _study_id(spec: ModelSpec, phase: str, task_id: str) -> str:
    """Canonical Optuna study name for ``(spec, phase)`` in one pipeline run."""
    return f"{study_name(FEATURE, spec.dataset.disease_id, phase)}-{task_id}"


# === HPARAM phase =======================================================


def run_hparam(
    spec: ModelSpec,
    *,
    trials: int,
    epochs: int,
    smoke: bool,
    task_id: str,
) -> str:
    """Optuna search over ``spec.hparam_space``. Returns the lineage id."""
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna not installed — `uv sync --extra vision-server`."
        ) from exc

    study = optuna.create_study(
        study_name=_study_id(spec, "hparam", task_id),
        storage=optuna_storage_uri(),
        direction="maximize",
        load_if_exists=True,
    )
    search_floors = spec.task.phase_floors("search")
    logger.info(
        "forge.search: trials=%d epochs=%d floors=%s composite_weights=%s",
        trials,
        epochs,
        search_floors.floors,
        spec.task.composite_weights,
    )
    study.optimize(
        _build_search_objective(spec, epochs=epochs, smoke=smoke, task_id=task_id),
        n_trials=trials,
    )
    return task_id


def _build_search_objective(spec: ModelSpec, *, epochs: int, smoke: bool, task_id: str):
    """Construct the per-trial objective for the search phase."""

    def _objective(trial) -> float:
        trial.set_user_attr("task_id", task_id)
        params = spec.suggest_hparams(trial)
        return _run_search_trial(spec, params, epochs=epochs, smoke=smoke, trial=trial)

    return _objective


def _run_search_trial(
    spec: ModelSpec,
    params: dict[str, Any],
    *,
    epochs: int,
    smoke: bool,
    trial,
) -> float:
    """One Optuna trial — train briefly + return feasibility-aware score."""
    if smoke:
        logger.info("smoke trial: params=%s", params)
        return FEASIBLE_OFFSET + 0.5

    task = spec.task
    floors = task.phase_floors("search")
    splits = spec.dataset.build_splits()
    device = select_device(_torch(), require_gpu=True)
    model = task.build_model(
        backbone=params["backbone"],
        num_classes=len(spec.dataset.labels),
        pretrained=True,
    ).to(device)
    optimizer = _torch().optim.AdamW(model.parameters(), lr=params["lr"])

    train_loader = _torch().utils.data.DataLoader(
        splits.train, batch_size=16, shuffle=True, num_workers=0
    )
    val_loader = _torch().utils.data.DataLoader(
        splits.val, batch_size=16, shuffle=False, num_workers=0
    )

    best_score = -float("inf")
    best_breakdown: dict[str, float] | None = None
    with mlflow_run(
        FEATURE,
        spec.dataset.disease_id,
        run_name=f"trial-{trial.number}",
        run_type="train",
        params=params,
        nested=True,
        tags={TASK_ID_TAG: task_id_of(trial)},
    ):
        for epoch in range(epochs):
            _train_one_epoch(task, model, train_loader, optimizer, device, params)
            _, breakdown = task.evaluate(
                model, val_loader, device, labels=spec.dataset.labels, hp=params
            )
            score = feasibility_aware_score(breakdown, floors, task.composite_weights)
            if score > best_score:
                best_score = score
                best_breakdown = breakdown
            log_metrics({f"val/{k}": v for k, v in breakdown.items()}, step=epoch)
            trial.report(score, epoch)
            if trial.should_prune():
                import optuna as _optuna

                raise _optuna.TrialPruned()
        if best_breakdown is not None:
            trial.set_user_attr("breakdown", best_breakdown)
    del train_loader, val_loader
    feasible = best_score >= FEASIBLE_OFFSET
    logger.info(
        "forge.search trial=%d score=%.4f feasible=%s breakdown=%s",
        trial.number,
        best_score,
        feasible,
        best_breakdown or {},
    )
    return best_score


def _train_one_epoch(
    task, model, loader, optimizer, device, hp: dict[str, Any]
) -> float:
    """Run one training epoch and return the mean batch loss."""
    model.train()
    losses: list[float] = []
    for batch in loader:
        inputs, targets = task.unpack_batch(batch)
        inputs = inputs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        outputs = model(inputs)
        loss = task.compute_loss(outputs, targets, hp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / max(len(losses), 1)


def task_id_of(trial) -> str:
    """Return the trial's task_id, falling back to a placeholder."""
    return trial.user_attrs.get("task_id", "unknown")


def _load_best_hp(spec: ModelSpec, task_id: str) -> dict[str, Any]:
    """Load the best trial's params from the search study for ``task_id``.

    Refuses to fall back to defaults — silent fallback would let an
    operator promote a model trained with stale defaults after a
    search-space change.
    """
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna not installed — `uv sync --extra vision-server`."
        ) from exc

    name = _study_id(spec, "hparam", task_id)
    try:
        study = optuna.load_study(study_name=name, storage=optuna_storage_uri())
    except KeyError as exc:
        raise SystemExit(
            f"Optuna study {name!r} not found in {optuna_storage_uri()}. "
            f"Run the search phase first."
        ) from exc
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        raise SystemExit(
            f"Optuna study {name!r} has no completed trials. "
            f"Run the hparam search before production training."
        )
    best = max(completed, key=lambda t: float("-inf") if t.value is None else t.value)
    return dict(best.params)


# === TRAIN phase ========================================================


@dataclasses.dataclass
class _TrainingHistory:
    best_state_dict: Any
    best_val_composite: float
    best_val_feasibility: float
    best_val_feasible: bool
    best_epoch: int
    epochs_trained: int
    early_stopped: bool
    feasible_epoch_count: int
    curves: list[dict[str, float]]
    val_breakdown: dict[str, float]
    test_breakdown: dict[str, float]
    test_score: float
    mlflow_info: dict[str, str]


def run_train(
    spec: ModelSpec,
    *,
    max_epochs: int,
    patience: int,
    smoke: bool,
    task_id: str | None = None,
) -> Path:
    """Train one production checkpoint and write the staging dir.

    Returns the staging dir path so the caller can chain into tune.
    """
    if task_id is None:
        task_id = generate_task_id()
    logger.info("train task_id=%s model_id=%s", task_id, spec.model_id)

    if smoke:
        params = _smoke_hp(spec)
        logger.info("smoke training with synthetic HP: %s", params)
        history = _smoke_training_history(spec, params)
    else:
        params = _load_best_hp(spec, task_id)
        history = _train_with_early_stopping(
            spec, params, max_epochs=max_epochs, patience=patience, task_id=task_id
        )

    staging = staging_dir(dataset_id=spec.dataset.disease_id, model_id=spec.model_id)
    staging.mkdir(parents=True, exist_ok=True)

    weights = staging / "weights.pt"
    _persist_weights(history.best_state_dict, weights, params=params, smoke=smoke)
    weights_sha = sha256_file(weights)

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
        spec,
        target=staging / "manifest.json",
        weights_sha=weights_sha,
        backbone=params["backbone"],
    )
    (staging / "eval_metrics.json").write_text(
        json.dumps(eval_metrics, indent=2), encoding="utf-8"
    )
    (staging / "training_curve.json").write_text(
        json.dumps(history.curves, indent=2), encoding="utf-8"
    )
    _write_provenance(
        spec, staging, params=params, mlflow_info=history.mlflow_info, task_id=task_id
    )
    return staging


def _train_with_early_stopping(
    spec: ModelSpec,
    params: dict[str, Any],
    *,
    max_epochs: int,
    patience: int,
    task_id: str,
) -> _TrainingHistory:
    """Run the production training loop with feasibility-aware best-epoch pick."""
    torch = _torch()
    task = spec.task
    train_floors = task.phase_floors("train")
    splits = spec.dataset.build_splits()
    device = select_device(torch, require_gpu=True)
    model = task.build_model(
        backbone=params["backbone"],
        num_classes=len(spec.dataset.labels),
        pretrained=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    train_loader = torch.utils.data.DataLoader(
        splits.train, batch_size=16, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        splits.val, batch_size=16, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        splits.test, batch_size=16, shuffle=False, num_workers=2
    )

    state = _LoopState()
    with mlflow_run(
        FEATURE,
        spec.dataset.disease_id,
        run_name=f"train-{spec.model_id}",
        run_type="train",
        params={**params, "max_epochs": max_epochs, "patience": patience},
        tags={TASK_ID_TAG: task_id},
    ) as run:
        run_id = run.info.run_id
        for epoch in range(max_epochs):
            train_loss = _train_one_epoch(
                task, model, train_loader, optimizer, device, params
            )
            val_loss, val_breakdown = task.evaluate(
                model, val_loader, device, labels=spec.dataset.labels, hp=params
            )
            state.record(
                epoch,
                train_loss,
                val_loss,
                val_breakdown,
                train_floors,
                task,
                model,
                patience,
            )
            if state.early_stopped:
                logger.info("early stop at epoch %d (patience=%d)", epoch, patience)
                break

        if state.best_selection_score < FEASIBLE_OFFSET:
            deficits = train_floors.deficits(state.best_val_breakdown)
            logger.warning(
                "train: no epoch met TRAIN_FLOORS (best epoch %d, deficits=%s). "
                "Deploy gate will almost certainly reject.",
                state.best_epoch,
                {k: f"{v:.3f}" for k, v in deficits.items()},
            )
        if state.best_state_dict is not None:
            model.load_state_dict(state.best_state_dict)
        _, test_breakdown = task.evaluate(
            model, test_loader, device, labels=spec.dataset.labels, hp=params
        )
        log_metrics({f"test/{k}": v for k, v in test_breakdown.items()})
        mlflow_info = {
            "experiment_name": experiment_name(FEATURE, spec.dataset.disease_id),
            "run_id": run_id,
        }

    return _TrainingHistory(
        best_state_dict=state.best_state_dict,
        best_val_composite=state.best_val_composite,
        best_val_feasibility=state.best_selection_score,
        best_val_feasible=state.best_selection_score >= FEASIBLE_OFFSET,
        best_epoch=state.best_epoch,
        epochs_trained=len(state.curves),
        early_stopped=state.early_stopped,
        feasible_epoch_count=state.feasible_epoch_count,
        curves=state.curves,
        val_breakdown=state.best_val_breakdown,
        test_breakdown=test_breakdown,
        test_score=test_breakdown.get("composite", 0.0),
        mlflow_info=mlflow_info,
    )


class _LoopState:
    """Per-epoch state for the early-stopping training loop.

    Pulled into a class because the loop's selection logic touches
    ~10 variables; threading them through ``_train_with_early_stopping``
    pushed it over the 100-line function ceiling.
    """

    def __init__(self) -> None:
        self.best_selection_score = -float("inf")
        self.best_val_composite = -1.0
        self.best_val_breakdown: dict[str, float] = {}
        self.best_epoch = 0
        self.best_state_dict: Any = None
        self.curves: list[dict[str, float]] = []
        self.feasible_epoch_count = 0
        self.epochs_since_improve = 0
        self.early_stopped = False

    def record(
        self,
        epoch,
        train_loss,
        val_loss,
        val_breakdown,
        train_floors,
        task,
        model,
        patience,
    ):
        val_composite = val_breakdown.get("composite", 0.0)
        score = feasibility_aware_score(
            val_breakdown, train_floors, task.composite_weights
        )
        epoch_is_feasible = is_feasible(val_breakdown, train_floors)
        if epoch_is_feasible:
            self.feasible_epoch_count += 1
        self.curves.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_score": val_composite,
                "val_selection_score": score,
                "val_feasible": float(epoch_is_feasible),
                **{f"val_{k}": v for k, v in val_breakdown.items()},
            }
        )
        log_metrics(
            {
                "train/loss": train_loss,
                "val/loss": val_loss,
                **{f"val/{k}": v for k, v in val_breakdown.items()},
            },
            step=epoch,
        )
        if score > self.best_selection_score:
            self.best_selection_score = score
            self.best_val_composite = val_composite
            self.best_val_breakdown = val_breakdown
            self.best_epoch = epoch
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self.epochs_since_improve = 0
        else:
            self.epochs_since_improve += 1
        if self.epochs_since_improve >= patience:
            self.early_stopped = True


def _smoke_hp(spec: ModelSpec) -> dict[str, Any]:
    """Synthesize an HP dict for ``--smoke``: first backbone + zero LR.

    Backbone is the first choice in ``hparam_space["backbone"]`` so the
    smoke path doesn't have to know which dataset it's driving.
    """
    backbone_space = spec.hparam_space.get("backbone")
    backbone = backbone_space.choices[0] if backbone_space else "custom_unet"
    params = {"backbone": backbone, "lr": 1e-3}
    if "seg_loss_weight" in spec.hparam_space:
        params["seg_loss_weight"] = 1.0
    return params


def _smoke_training_history(
    spec: ModelSpec, params: dict[str, Any]
) -> _TrainingHistory:
    """Return a fake training history exercising the staging-write path."""
    breakdown = spec.task.smoke_breakdown()
    selection_score = feasibility_aware_score(
        breakdown, spec.task.phase_floors("train"), spec.task.composite_weights
    )
    smoke_curve = {
        "epoch": 0,
        "train_loss": 0.42,
        "val_loss": 0.40,
        "val_score": breakdown.get("composite", 0.0),
        "val_selection_score": selection_score,
        "val_feasible": 1.0,
        **{f"val_{k}": v for k, v in breakdown.items()},
    }
    return _TrainingHistory(
        best_state_dict={"smoke": True, "backbone": params.get("backbone")},
        best_val_composite=breakdown.get("composite", 0.0),
        best_val_feasibility=selection_score,
        best_val_feasible=selection_score >= FEASIBLE_OFFSET,
        best_epoch=0,
        epochs_trained=1,
        early_stopped=False,
        feasible_epoch_count=1,
        curves=[smoke_curve],
        val_breakdown=breakdown,
        test_breakdown=breakdown,
        test_score=breakdown.get("composite", 0.0),
        mlflow_info={"experiment_name": "smoke", "run_id": "smoke"},
    )


def _persist_weights(
    state_dict, target: Path, *, params: dict[str, Any], smoke: bool
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


def _write_manifest(
    spec: ModelSpec, *, target: Path, weights_sha: str, backbone: str
) -> None:
    """Compose, validate, and write ``manifest.json``."""
    manifest = Manifest(
        model_id=spec.model_id,
        model_version=spec.model_version,
        framework=spec.framework,
        accepted_modality=spec.dataset.accepted_modality,
        sha256_weights=weights_sha,
        task=spec.task.name,
        labels=list(spec.dataset.labels),
        labels_meta=dict(spec.dataset.labels_meta),
        cancer_class=spec.dataset.cancer_class,
        cancer_status_mapping=spec.dataset.cancer_status_mapping(),
        clinical_action_mapping=spec.dataset.clinical_action_mapping(),
        supports_saliency=spec.task.supports_saliency,
        supports_tta=spec.task.supports_tta,
        model_card_url=None,
        backbone=backbone,
        tuned_inference=None,
    )
    target.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_provenance(
    spec: ModelSpec,
    staging: Path,
    *,
    params: dict[str, Any],
    mlflow_info: dict[str, str],
    task_id: str,
) -> None:
    """Drop a provenance.json file the deploy phase splices into LATEST.jsonl."""
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
            "search_study_name": _study_id(spec, "hparam", task_id),
            "search_trial_task_id": task_id,
        },
    }
    (staging / "provenance.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


# === TUNE phase =========================================================


def run_tune(
    spec: ModelSpec,
    *,
    staging_dir: Path,
    trials: int,
    smoke: bool,
) -> Path:
    """Optuna over inference params; update staging dir's manifest in place."""
    manifest_path = staging_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} missing — run train first.")
    manifest = Manifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    task_id = _read_task_id(staging_dir)

    if smoke:
        best = spec.task.smoke_tuned_params()
        tuned_val = spec.task.smoke_breakdown()
        tuned_test = tuned_val
        mlflow_info = {"experiment_name": "smoke", "run_id": "smoke"}
    else:
        cache_val = _cache_split(spec, manifest, staging_dir, split="val")
        cache_test = _cache_split(spec, manifest, staging_dir, split="test")
        best, tuned_val, tuned_test, mlflow_info = _run_inference_study(
            spec,
            cache_val=cache_val,
            cache_test=cache_test,
            trials=trials,
            task_id=task_id,
        )

    tuned_block = spec.task.build_tuned_inference_block(
        best, labels=spec.dataset.labels
    )
    updated = _with_tuned_inference(manifest, tuned_block)
    manifest_path.write_text(
        json.dumps(updated.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _update_eval_metrics(
        staging_dir, best=best, tuned_val=tuned_val, tuned_test=tuned_test
    )
    _append_provenance(
        spec, staging_dir, best=best, mlflow_info=mlflow_info, task_id=task_id
    )
    return staging_dir


def _cache_split(spec: ModelSpec, manifest: Manifest, staging_dir: Path, *, split: str):
    """Build the per-split forward-output cache for the tune phase."""
    torch = _torch()
    weights = staging_dir / "weights.pt"
    if not weights.exists():
        raise SystemExit(
            f"{weights} missing — train phase did not persist a checkpoint"
        )

    splits = spec.dataset.build_splits()
    ds = getattr(splits, split)
    device = select_device(torch)
    model = _load_model_from_staging(spec, manifest, weights, device)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=16, shuffle=False, num_workers=2
    )
    return spec.task.cache_outputs(model, loader, device)


def _run_inference_study(
    spec: ModelSpec,
    *,
    cache_val,
    cache_test,
    trials: int,
    task_id: str,
):
    """Run Optuna over the inference-param space; return best + breakdowns."""
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna not installed — `uv sync --extra vision-server`"
        ) from exc

    task = spec.task
    floors = task.phase_floors("tune")
    name = _study_id(spec, "tune", task_id)

    def _objective(trial) -> float:
        trial.set_user_attr("task_id", task_id)
        params = spec.suggest_inference_params(trial)
        # Constraint: low_max < medium_max. Pruning surfaces it to the
        # acquisition function as an infeasible cell.
        if params.get("confidence_low_max", 0) >= params.get(
            "confidence_medium_max", 1
        ):
            raise optuna.TrialPruned()
        breakdown = task.evaluate_cache(cache_val, params, labels=spec.dataset.labels)
        trial.set_user_attr("breakdown", breakdown)
        return feasibility_aware_score(breakdown, floors, task.composite_weights)

    with mlflow_run(
        FEATURE,
        spec.dataset.disease_id,
        run_name=f"tune-{spec.model_id}",
        run_type="tune",
        params={"trials": trials},
        tags={TASK_ID_TAG: task_id},
    ) as mlflow_handle:
        study = optuna.create_study(
            study_name=name,
            storage=optuna_storage_uri(),
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(_objective, n_trials=trials)
        best = dict(study.best_trial.params)
        tuned_val = task.evaluate_cache(cache_val, best, labels=spec.dataset.labels)
        tuned_test = task.evaluate_cache(cache_test, best, labels=spec.dataset.labels)

        summary = study_feasibility_summary(study, floors)
        if study.best_trial.value < FEASIBLE_OFFSET:
            logger.warning(
                "tune: no feasible trial in %d (worst deficits=%s). "
                "Deploy gate will reject. Retrain with stronger model / more data.",
                trials,
                summary["worst_deficits"],
            )
        log_metrics({f"val/{k}": v for k, v in tuned_val.items()})
        log_metrics({f"test/{k}": v for k, v in tuned_test.items()})
        mlflow_info = {
            "experiment_name": experiment_name(FEATURE, spec.dataset.disease_id),
            "run_id": mlflow_handle.info.run_id,
            "study_name": name,
            "storage_uri": optuna_storage_uri(),
        }
    return best, tuned_val, tuned_test, mlflow_info


def _with_tuned_inference(manifest: Manifest, tuned) -> Manifest:
    """Set ``tuned_inference`` via ``model_validate`` so cross-field rules fire."""
    data = manifest.model_dump(mode="python")
    data["tuned_inference"] = tuned.model_dump(mode="python")
    return Manifest.model_validate(data)


def _update_eval_metrics(staging_dir: Path, *, best, tuned_val, tuned_test) -> None:
    """Splice the tuned breakdowns into ``eval_metrics.json``."""
    eval_path = staging_dir / "eval_metrics.json"
    existing = (
        json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    )
    existing.update(
        {
            "tuned_params": best,
            "tuned_val_breakdown": tuned_val,
            "tuned_test_breakdown": tuned_test,
            "tuned_test_score": tuned_test.get("composite", 0.0),
        }
    )
    eval_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _append_provenance(
    spec: ModelSpec,
    staging_dir: Path,
    *,
    best: dict[str, Any],
    mlflow_info: dict[str, str],
    task_id: str,
) -> None:
    """Add the tune-phase breadcrumb to provenance.json."""
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
            "tune_study_name": _study_id(spec, "tune", task_id),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_task_id(staging_dir: Path) -> str:
    """Read the lineage task_id from staging dir's provenance.json."""
    provenance_path = staging_dir / "provenance.json"
    if not provenance_path.exists():
        raise SystemExit(f"{provenance_path} missing — staging dir is incomplete.")
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    task_id = payload.get("task_id")
    if not task_id:
        raise SystemExit(
            f"{provenance_path} has no task_id — re-train with current pipeline."
        )
    return str(task_id)


def _load_model_from_staging(
    spec: ModelSpec, manifest: Manifest, weights: Path, device
):
    """Rebuild the trained model architecture and load the checkpoint."""
    import torch

    model = spec.task.build_model(
        backbone=manifest.backbone,
        num_classes=len(manifest.labels),
        pretrained=False,
    ).to(device)
    state = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# === DEPLOY phase ========================================================


def run_deploy(
    spec: ModelSpec,
    *,
    staging_dir: Path,
    smoke: bool = False,
    force: bool = False,
) -> Path:
    """Promote ``staging_dir`` to a versioned stable path."""
    manifest_path = staging_dir / "manifest.json"
    eval_path = staging_dir / "eval_metrics.json"
    provenance_path = staging_dir / "provenance.json"
    for path in (manifest_path, eval_path, provenance_path):
        if not path.exists():
            raise SystemExit(f"{path} missing — staging dir is incomplete")

    manifest = Manifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if manifest.tuned_inference is None:
        raise SystemExit(f"{manifest_path} has tuned_inference=None — run tune first.")
    eval_metrics = json.loads(eval_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    tuned_test = eval_metrics.get("tuned_test_breakdown") or {}
    if not tuned_test:
        raise SystemExit(f"{eval_path} missing tuned_test_breakdown — run tune first.")

    floors_map = spec.task.deploy_floors_map()
    if force:
        logger.warning(
            "deploy --force: skipping floor gate. floors=%s actual=%s. Do NOT use in production.",
            floors_map,
            {k: tuned_test.get(k) for k in floors_map},
        )
    else:
        check_floors(tuned_test, floors_map)

    # Smoke deploy verifies floor wiring against Task.smoke_breakdown()
    # but stops there: it must not touch the disease's stable directory,
    # the <model_id> stable symlink, LATEST.jsonl, configs/vision.yaml,
    # nor the regression gate. Otherwise a sub-second wiring check
    # leaves a fake "deployed model" the vision-server would try to load
    # at boot — and the regression gate would block the next smoke run
    # against its own previous output.
    if smoke:
        logger.info(
            "smoke deploy: floors cleared on synthetic breakdown; "
            "skipped promote / LATEST.jsonl / symlink / vision.yaml / "
            "regression-gate. Staging dir at %s.",
            staging_dir,
        )
        return staging_dir

    previous = read_latest_entry(
        latest_jsonl_path(spec.dataset.disease_id), model_id=spec.model_id
    )
    if previous is not None:
        _check_regression(previous, tuned_test)

    # Validate registry-side wiring up-front, so a missing
    # configs/vision.yaml entry fails BEFORE copytree / symlink — not
    # after, like the original ordering did, which would leave a
    # dangling stable-symlink pointing at a half-promoted artifact.
    assert_vision_yaml_has_model(spec.model_id)

    tag = version_tag()
    stable_dirname = f"{spec.model_id}__{tag}"
    stable_path = disease_root(spec.dataset.disease_id) / stable_dirname
    if stable_path.exists():
        raise SystemExit(
            f"refusing to overwrite {stable_path} — pick a different version_tag or remove it manually."
        )
    logger.info("promoting %s → %s", staging_dir, stable_path)
    shutil.copytree(staging_dir, stable_path)
    _update_stable_symlink(spec, stable_dirname)

    new_manifest_sha = sha256_file(stable_path / "manifest.json")
    new_weights_subpath = f"vision/{spec.dataset.disease_id}/{stable_dirname}"
    patch_vision_yaml(
        model_id=spec.model_id,
        new_weights_subpath=new_weights_subpath,
        new_manifest_sha=new_manifest_sha,
    )
    entry = _build_latest_entry(
        spec,
        version_tag=tag,
        weights_subpath=new_weights_subpath,
        manifest_path=stable_path / "manifest.json",
        manifest_sha=new_manifest_sha,
        provenance=provenance,
        eval_metrics=eval_metrics,
        tuned_test=tuned_test,
        previous_score=(
            previous["metrics"]["tuned_test_composite"] if previous else None
        ),
    )
    append_latest_entry(latest_jsonl_path(spec.dataset.disease_id), entry)
    logger.info("appended LATEST.jsonl entry for version %s", tag)
    return stable_path


def _check_regression(previous: dict[str, Any], tuned_test: dict[str, float]) -> None:
    """Candidate must beat the previously-deployed composite for the same model_id."""
    prev_score = float(previous["metrics"]["tuned_test_composite"])
    candidate_score = float(tuned_test["composite"])
    if candidate_score <= prev_score:
        raise SystemExit(
            f"regression gate failed: candidate tuned_test_composite="
            f"{candidate_score:.4f} ≤ active {prev_score:.4f} "
            f"(version {previous['version_tag']}). Re-tune or retrain."
        )
    logger.info(
        "regression gate ok: %.4f > active %.4f (Δ=%+.4f)",
        candidate_score,
        prev_score,
        candidate_score - prev_score,
    )


def _update_stable_symlink(spec: ModelSpec, stable_dirname: str) -> None:
    """Atomically point ``<model_id>`` symlink at the new versioned dir."""
    root = disease_root(spec.dataset.disease_id)
    stable_link = root / spec.model_id
    tmp_link = root / f"{spec.model_id}.tmp"
    tmp_link.unlink(missing_ok=True)
    tmp_link.symlink_to(stable_dirname)
    tmp_link.rename(stable_link)
    logger.info("updated stable symlink %s → %s", stable_link.name, stable_dirname)


def _build_latest_entry(
    spec: ModelSpec,
    *,
    version_tag: str,
    weights_subpath: str,
    manifest_path: Path,
    manifest_sha: str,
    provenance: dict[str, Any],
    eval_metrics: dict[str, Any],
    tuned_test: dict[str, float],
    previous_score: float | None,
) -> dict[str, Any]:
    """Compose the JSONL row carrying the full deploy audit trail."""
    from datetime import UTC, datetime

    candidate_score = float(tuned_test["composite"])
    delta = (
        candidate_score - float(previous_score) if previous_score is not None else None
    )
    floors_map = spec.task.deploy_floors_map()
    return {
        "version_tag": version_tag,
        "deployed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task_id": provenance.get("task_id"),
        "model_id": spec.model_id,
        "disease_id": spec.dataset.disease_id,
        "weights_subpath": weights_subpath,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "mlflow": {
            "tracking_uri": provenance.get("mlflow", {}).get("tracking_uri"),
            "experiment_name": provenance.get("mlflow", {}).get("experiment_name"),
            "train_run_id": provenance.get("mlflow", {}).get("train_run_id"),
            "tune_run_id": provenance.get("tune", {})
            .get("mlflow", {})
            .get("tune_run_id"),
        },
        "optuna": {
            "storage_uri": provenance.get("optuna", {}).get("storage_uri"),
            "search_study_name": provenance.get("optuna", {}).get("search_study_name"),
            "search_trial_task_id": provenance.get("optuna", {}).get(
                "search_trial_task_id"
            ),
            "tune_study_name": provenance.get("tune", {})
            .get("optuna", {})
            .get("tune_study_name"),
        },
        "best_hp": provenance.get("params"),
        "best_inference_params": provenance.get("tune", {}).get("params"),
        "metrics": {
            "best_val_score": eval_metrics.get("best_val_score"),
            "test_score": eval_metrics.get("test_score"),
            "tuned_val_breakdown": eval_metrics.get("tuned_val_breakdown"),
            "tuned_test_breakdown": tuned_test,
            "tuned_test_composite": candidate_score,
            "best_epoch": eval_metrics.get("best_epoch"),
            "early_stopped": eval_metrics.get("early_stopped"),
            "epochs_trained": eval_metrics.get("epochs_trained"),
        },
        "previous_tuned_test_composite": previous_score,
        "delta": delta,
        "floors_passed": {
            name: tuned_test.get(name, 0.0) >= floor
            for name, floor in floors_map.items()
        },
    }


# === PIPELINE orchestrator =============================================


def run_pipeline(
    spec: ModelSpec,
    *,
    phases: tuple[str, ...] = ALL_PHASES,
    trials: int = 20,
    search_epochs: int = 15,
    max_epochs: int = 100,
    patience: int = 15,
    tune_trials: int = 30,
    smoke: bool = False,
    staging_dir: Path | None = None,
    task_id: str | None = None,
    force: bool = False,
    deploy_force: bool = False,
) -> Path | None:
    """Run the requested phases in order. Returns the final stable path or None."""
    if task_id is None:
        task_id = generate_task_id()
    logger.info("pipeline task_id=%s model=%s force=%s", task_id, spec.model_id, force)

    staging = staging_dir
    stable_path: Path | None = None

    if "search" in phases:
        logger.info("=== phase: search — task_id=%s ===", task_id)
        if not smoke:
            run_hparam(
                spec, trials=trials, epochs=search_epochs, smoke=False, task_id=task_id
            )
            breakdown = _read_search_winner_breakdown(spec, task_id)
            if breakdown is not None:
                gate_or_raise(
                    phase_label="search",
                    breakdown=breakdown,
                    floors=spec.task.phase_floors("search"),
                    force=force,
                )

    if "train" in phases:
        logger.info("=== phase: train — task_id=%s ===", task_id)
        staging = run_train(
            spec, max_epochs=max_epochs, patience=patience, smoke=smoke, task_id=task_id
        )
        logger.info("staging dir: %s", staging)
        if not smoke:
            breakdown = _read_staging_breakdown(staging, "val_breakdown")
            if breakdown is not None:
                gate_or_raise(
                    phase_label="train",
                    breakdown=breakdown,
                    floors=spec.task.phase_floors("train"),
                    force=force,
                )

    if "tune" in phases:
        logger.info("=== phase: tune — task_id=%s ===", task_id)
        target = staging or latest_staging_dir(
            dataset_id=spec.dataset.disease_id, model_id=spec.model_id
        )
        run_tune(spec, staging_dir=target, trials=tune_trials, smoke=smoke)
        staging = target
        if not smoke:
            breakdown = _read_staging_breakdown(staging, "tuned_test_breakdown")
            if breakdown is not None:
                gate_or_raise(
                    phase_label="tune",
                    breakdown=breakdown,
                    floors=spec.task.phase_floors("tune"),
                    force=force,
                )

    if "deploy" in phases:
        logger.info("=== phase: deploy — task_id=%s ===", task_id)
        target = staging or latest_staging_dir(
            dataset_id=spec.dataset.disease_id, model_id=spec.model_id
        )
        stable_path = run_deploy(
            spec, staging_dir=target, smoke=smoke, force=deploy_force
        )
        logger.info("deployed to: %s", stable_path)

    return stable_path


def _read_search_winner_breakdown(
    spec: ModelSpec, task_id: str
) -> dict[str, float] | None:
    """Load the best Optuna trial for ``task_id`` and return its breakdown."""
    try:
        import optuna
    except ImportError:
        return None
    name = _study_id(spec, "hparam", task_id)
    try:
        study = optuna.load_study(study_name=name, storage=optuna_storage_uri())
    except KeyError:
        return None
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        return None
    best = max(completed, key=lambda t: float("-inf") if t.value is None else t.value)
    return dict(best.user_attrs.get("breakdown") or {}) or None


def _read_staging_breakdown(staging_dir: Path, key: str) -> dict[str, float] | None:
    """Read a specific breakdown dict out of ``eval_metrics.json``."""
    path = staging_dir / "eval_metrics.json"
    if not path.exists():
        return None
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    block = payload.get(key)
    if not isinstance(block, dict):
        return None
    return {k: float(v) for k, v in block.items() if isinstance(v, (int, float))}


# === lazy torch handle =================================================


def _torch():
    try:
        import torch  # noqa: F401

        return torch
    except ImportError as exc:
        raise SystemExit(
            "torch not installed — `uv sync --extra vision-server`."
        ) from exc


__all__ = [
    "run_deploy",
    "run_hparam",
    "run_pipeline",
    "run_train",
    "run_tune",
]
