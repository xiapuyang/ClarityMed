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
import gc
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
    class_weight_tensor_from_splits,
    dataset_stats_from_splits,
    disease_root,
    latest_jsonl_path,
    latest_staging_dir,
    log_metrics,
    patch_vision_yaml,
    read_latest_entry,
    scalar_only,
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
    # Resume handoff — when ``--task-id`` reuses an existing study, log how
    # many trials are already in storage so the operator can see at a
    # glance that TPE has a prior to work from (and that ``trials=N`` is
    # additional, not total).
    existing = study.trials
    if existing:
        by_state: dict[str, int] = {}
        for t in existing:
            by_state[t.state.name] = by_state.get(t.state.name, 0) + 1
        logger.info(
            "forge.search resume: existing_trials=%d by_state=%s",
            len(existing),
            by_state,
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


def _shutdown_dataloaders(*loaders) -> None:
    """Synchronously kill ``persistent_workers=True`` DataLoader workers.

    ``del loader`` only schedules ``__del__`` — under Optuna sweeps that
    means trial N's workers may still be alive when N+1 spawns its own,
    leaking pipe FDs until ``os.pipe()`` returns ``Errno 24``. Pruned
    trials make it worse: ``TrialPruned`` raises out of the epoch loop,
    bypassing any ``del`` at the function tail. We poke the private
    ``_iterator._shutdown_workers`` because that's exactly what torch's
    own ``__del__`` calls — no public hook for it.
    """
    for loader in loaders:
        if loader is None:
            continue
        it = getattr(loader, "_iterator", None)
        if it is None:
            continue
        shutdown = getattr(it, "_shutdown_workers", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception:
                pass
        loader._iterator = None


def _fd_breakdown() -> dict[str, int]:
    """Categorize current-process FDs into pipes/files/sockets/anon_inode/other.

    On Linux reads ``/proc/self/fd`` readlink targets — pinpoint
    categorization (``pipe:[N]``, ``socket:[N]``, ``anon_inode:[…]``,
    or a real path). On macOS falls back to ``psutil.Process()``
    interfaces (``open_files``, ``net_connections``); the remainder
    of ``num_fds()`` minus what we could attribute lands in ``other``
    since BSD-style ``proc_pidinfo`` doesn't surface the pipe vs
    anon_inode split without ``lsof``-equivalent privileges.

    Returns an empty dict when psutil is not importable (e.g. minimal
    CI image) so the caller can short-circuit without an extra import
    guard.
    """
    import os
    import sys

    try:
        import psutil
    except ImportError:
        return {}

    out = {
        "total": 0,
        "files": 0,
        "sockets": 0,
        "pipes": 0,
        "anon_inode": 0,
        "other": 0,
    }
    if sys.platform.startswith("linux") and os.path.isdir("/proc/self/fd"):
        for entry in os.listdir("/proc/self/fd"):
            out["total"] += 1
            try:
                tgt = os.readlink(f"/proc/self/fd/{entry}")
            except OSError:
                out["other"] += 1
                continue
            if tgt.startswith("pipe:"):
                out["pipes"] += 1
            elif tgt.startswith("socket:"):
                out["sockets"] += 1
            elif tgt.startswith("anon_inode:"):
                out["anon_inode"] += 1
            elif tgt.startswith("/"):
                out["files"] += 1
            else:
                out["other"] += 1
        return out

    proc = psutil.Process()
    try:
        out["files"] = len(proc.open_files())
    except Exception:  # noqa: BLE001
        pass
    try:
        out["sockets"] = len(proc.net_connections(kind="all"))
    except Exception:  # noqa: BLE001
        pass
    try:
        out["total"] = proc.num_fds()
    except Exception:  # noqa: BLE001
        out["total"] = out["files"] + out["sockets"]
    attributed = out["files"] + out["sockets"]
    if out["total"] > attributed:
        # macOS collapses pipes + anon_inode + uncategorized into "other"
        # — direction of growth still distinguishes "system FDs" from
        # files/sockets, which is enough to pick the next investigation.
        out["other"] = out["total"] - attributed
    return out


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

    # Per-trial FD-count heartbeat, categorized. Pure num_fds() only
    # tells us "something leaks"; the type breakdown points at the
    # suspect: pipes → DataLoader/multiprocessing, files → MLflow /
    # tempfile, sockets → httpx / wandb, anon_inode → asyncio
    # epoll/eventfd/inotify. With persistent_workers=False the pipes
    # column should hold flat — drift in any other column is a
    # non-DataLoader leak.
    try:
        fdb = _fd_breakdown()
        if fdb:
            logger.info(
                "forge.search trial=%d fd total=%d files=%d sockets=%d "
                "pipes=%d anon_inode=%d other=%d",
                trial.number,
                fdb["total"],
                fdb["files"],
                fdb["sockets"],
                fdb["pipes"],
                fdb["anon_inode"],
                fdb["other"],
            )
    except Exception:  # noqa: BLE001 — heartbeat must never break a trial
        logger.debug("forge.search fd heartbeat failed", exc_info=True)
    # Per-trial Optuna picks — surfaces ``class_weight`` and friends so
    # the operator can correlate per-epoch behaviour with the scheme TPE
    # chose, without digging into the MLflow run.
    logger.info("forge.search trial=%d params=%s", trial.number, params)

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
    # ``hp`` carries the (per-batch) loss-time inputs: Optuna's suggested
    # params plus any derived tensors that aren't JSON-serialisable.
    # ``params`` stays clean so Optuna trial.params, mlflow params, and
    # provenance.json don't get a torch tensor stuffed in them. Underscore
    # prefix marks the derived key as internal.
    hp = _build_hp_with_class_weight(params, splits, spec, device)

    # Worker counts come from the dataset spec — tiny datasets (BUSI:
    # ~780 images, ~40 batches/epoch) override the (8, 4) default down
    # to (2, 2) because macOS ``spawn`` re-imports torch/PIL/claritymed
    # in each worker (~1s each) and that cost dwarfs the actual
    # forward/backward pass at this scale.
    #
    # ``persistent_workers=False`` in search — the per-epoch respawn
    # bill (~num_workers seconds on macOS ``spawn``) is small for the
    # 3-15 epoch search budgets, and ``persistent_workers=True`` was
    # leaking FDs across trials (mp Pipe/SemLock objects whose Python
    # GC doesn't run on the heartbeat-to-heartbeat boundary even with
    # explicit ``_shutdown_workers`` + ``gc.collect()``). Hit macOS's
    # 256 launchd soft limit by trial ~30 and crashed train's test_loader
    # spawn with ``EMFILE``. Train phase keeps ``persistent_workers=True``
    # because its 100-epoch budget makes the respawn cost matter.
    train_workers, val_workers = spec.dataset.search_num_workers
    train_loader = _torch().utils.data.DataLoader(
        splits.train,
        batch_size=16,
        shuffle=True,
        num_workers=train_workers,
        persistent_workers=False,
    )
    val_loader = _torch().utils.data.DataLoader(
        splits.val,
        batch_size=16,
        shuffle=False,
        num_workers=val_workers,
        persistent_workers=False,
    )

    best_score = -float("inf")
    best_breakdown: dict[str, float] | None = None
    best_epoch = -1
    try:
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
                train_loss = _train_one_epoch(
                    task, model, train_loader, optimizer, device, hp
                )
                _, breakdown = task.evaluate(
                    model, val_loader, device, labels=spec.dataset.labels, hp=hp
                )
                score = feasibility_aware_score(
                    breakdown, floors, task.composite_weights
                )
                improved = score > best_score
                if improved:
                    best_score = score
                    best_breakdown = breakdown
                    best_epoch = epoch
                log_metrics(
                    {
                        "train/loss": train_loss,
                        "val/score": score,
                        **{f"val/{k}": v for k, v in scalar_only(breakdown).items()},
                    },
                    step=epoch,
                )
                # Mirror the running winner into ``val/best_*`` *every*
                # epoch — single-point metrics get rendered as bar charts
                # in MLflow, multi-point as a proper line. Carrying the
                # running best forward also preserves Overview tab
                # semantics (last logged value = selected epoch).
                best_scalar = scalar_only(best_breakdown or breakdown)
                log_metrics(
                    {
                        **{f"val/best_{k}": v for k, v in best_scalar.items()},
                        "val/best_score": best_score,
                        "val/best_epoch": float(best_epoch),
                    },
                    step=epoch,
                )
                # Per-epoch heartbeat — long search trials (15-25 min/epoch
                # on RSNA pre-optimisations) need progress visible mid-run,
                # not just the once-per-trial summary that fires at the end.
                # ``best`` carries the running winner so the operator can
                # see at a glance whether this epoch improved on the
                # trial's best-so-far without scrolling back.
                logger.info(
                    "forge.search trial=%d epoch=%d train_loss=%.4f "
                    "score=%.4f best=%.4f best_epoch=%d breakdown=%s",
                    trial.number,
                    epoch,
                    train_loss,
                    score,
                    best_score,
                    best_epoch,
                    {k: f"{v:.3f}" for k, v in scalar_only(breakdown).items()},
                )
                trial.report(score, epoch)
                if trial.should_prune():
                    import optuna as _optuna

                    raise _optuna.TrialPruned()
            if best_breakdown is not None:
                trial.set_user_attr("breakdown", best_breakdown)
    finally:
        _shutdown_dataloaders(train_loader, val_loader)
        del model, optimizer, train_loader, val_loader
        gc.collect()
    feasible = best_score >= FEASIBLE_OFFSET
    logger.info(
        "forge.search trial=%d score=%.4f best_epoch=%d feasible=%s breakdown=%s",
        trial.number,
        best_score,
        best_epoch,
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
    logger.info(
        "forge.search winner: trial=%d score=%.4f params=%s breakdown=%s",
        best.number,
        float(best.value) if best.value is not None else float("nan"),
        dict(best.params),
        best.user_attrs.get("breakdown") or {},
    )
    return dict(best.params)


# === TRAIN phase ========================================================


@dataclasses.dataclass
class _TrainingHistory:
    best_state_dict: Any
    best_val_composite: float
    # ``best_val_score`` is the feasibility-aware selection score:
    # ``feasibility_aware_score(val_breakdown, train_floors, weights)`` =
    # what early-stop tracks = what Optuna would have ranked. Distinct
    # from ``best_val_composite`` (raw weighted sum) when an epoch
    # missed a floor.
    best_val_score: float
    best_val_feasible: bool
    best_epoch: int
    epochs_trained: int
    early_stopped: bool
    feasible_epoch_count: int
    curves: list[dict[str, float]]
    val_breakdown: dict[str, float]
    test_breakdown: dict[str, float]
    # ``test_composite`` is the raw weighted-sum metric; ``test_score``
    # is the feasibility-aware optimisation target computed against
    # ``train_floors`` (matches the value Optuna would have ranked).
    test_composite: float
    test_score: float
    mlflow_info: dict[str, str]
    # ``dataset_stats`` captures the train/val/test class distribution
    # the model actually saw at training time, so any future re-eval can
    # reproduce or interpret reported breakdowns. ``None`` in smoke mode
    # (smoke skips ``build_splits()`` to stay fast).
    dataset_stats: dict[str, Any] | None = None


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

    eval_metrics: dict[str, Any] = {
        "params": params,
        "best_epoch": history.best_epoch,
        "best_val_composite": history.best_val_composite,
        # ``best_val_score`` is the canonical name for the
        # feasibility-aware selection score. ``best_val_feasible`` is
        # the boolean derived from the same number against the floor
        # offset.
        "best_val_score": history.best_val_score,
        "best_val_feasible": history.best_val_feasible,
        "epochs_trained": history.epochs_trained,
        "early_stopped": history.early_stopped,
        "feasible_epoch_count": history.feasible_epoch_count,
        "val_breakdown": history.val_breakdown,
        "test_breakdown": history.test_breakdown,
        "test_composite": history.test_composite,
        "test_score": history.test_score,
    }
    if history.dataset_stats is not None:
        eval_metrics["dataset_stats"] = history.dataset_stats
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
    dataset_stats = dataset_stats_from_splits(splits, spec.dataset.labels)
    device = select_device(torch, require_gpu=True)
    model = task.build_model(
        backbone=params["backbone"],
        num_classes=len(spec.dataset.labels),
        pretrained=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    # See ``_run_search_trial`` for the ``hp`` vs ``params`` rationale.
    hp = _build_hp_with_class_weight(params, splits, spec, device)
    # ``num_workers=8`` + ``persistent_workers=True`` — see the search
    # phase comment above; same reasoning, RSNA's epoch count is much
    # higher here (default 100) so the per-epoch worker spawn savings
    # compound. ``test_loader`` keeps ``persistent_workers=True`` even
    # though it's iterated once: the spawn happens lazily on the first
    # ``iter(...)`` and we'd rather pay it overlapped with model load
    # than serially on the final eval.
    train_loader = torch.utils.data.DataLoader(
        splits.train,
        batch_size=16,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        splits.val,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        splits.test,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
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
                task, model, train_loader, optimizer, device, hp
            )
            val_loss, val_breakdown = task.evaluate(
                model, val_loader, device, labels=spec.dataset.labels, hp=hp
            )
            score = state.record(
                epoch,
                train_loss,
                val_loss,
                val_breakdown,
                train_floors,
                task,
                model,
                patience,
            )
            # Per-epoch heartbeat — train phase runs up to ``max_epochs``
            # (default 100) and silent epochs hide whether the loss is
            # actually moving. Mirrors the search-phase line: score is
            # the feasibility-aware selection score (the thing
            # early-stop tracks), best tracks the running winner.
            logger.info(
                "forge.train epoch=%d train_loss=%.4f val_loss=%.4f "
                "score=%.4f best=%.4f best_epoch=%d breakdown=%s",
                epoch,
                train_loss,
                val_loss,
                score,
                state.best_selection_score,
                state.best_epoch,
                {k: f"{v:.3f}" for k, v in scalar_only(val_breakdown).items()},
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
            model, test_loader, device, labels=spec.dataset.labels, hp=hp
        )
        # Test score on ``train_floors`` — same formula Optuna optimised
        # in search and ``state.best_selection_score`` tracked across
        # epochs. ``composite`` is the raw weighted sum; ``score`` is the
        # feasibility-aware optimisation target and the right thing to
        # eyeball against the search winner and val/score curve.
        test_score = feasibility_aware_score(
            test_breakdown, train_floors, task.composite_weights
        )
        test_composite = test_breakdown.get("composite", 0.0)
        log_metrics({f"test/{k}": v for k, v in scalar_only(test_breakdown).items()})
        log_metrics({"test/score": test_score})
        logger.info(
            "forge.train test eval: score=%.4f composite=%.4f breakdown=%s",
            test_score,
            test_composite,
            {k: f"{v:.3f}" for k, v in scalar_only(test_breakdown).items()},
        )
        mlflow_info = {
            "experiment_name": experiment_name(FEATURE, spec.dataset.disease_id),
            "run_id": run_id,
        }

    return _TrainingHistory(
        best_state_dict=state.best_state_dict,
        best_val_composite=state.best_val_composite,
        best_val_score=state.best_selection_score,
        best_val_feasible=state.best_selection_score >= FEASIBLE_OFFSET,
        best_epoch=state.best_epoch,
        epochs_trained=len(state.curves),
        early_stopped=state.early_stopped,
        feasible_epoch_count=state.feasible_epoch_count,
        curves=state.curves,
        val_breakdown=state.best_val_breakdown,
        test_breakdown=test_breakdown,
        test_composite=test_composite,
        test_score=test_score,
        mlflow_info=mlflow_info,
        dataset_stats=dataset_stats,
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
        val_scalar = scalar_only(val_breakdown)
        # ``training_curve.json`` is a per-epoch timeline of scalars;
        # the nested ``per_class`` block in val_breakdown is dropped
        # here (only the final test/val breakdowns in eval_metrics.json
        # keep it).
        self.curves.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                # ``val_composite`` = raw weighted sum (the breakdown
                # number). ``val_score`` = feasibility-aware score =
                # what early-stop tracks = what Optuna would have
                # ranked. They differ when an epoch misses a floor.
                "val_composite": val_composite,
                "val_score": score,
                "val_feasible": float(epoch_is_feasible),
                **{f"val_{k}": v for k, v in val_scalar.items()},
            }
        )
        log_metrics(
            {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/score": score,
                **{f"val/{k}": v for k, v in val_scalar.items()},
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
        # Mirror the running winner into ``val/best_*`` *every* epoch
        # (not just on improvement) so MLflow renders a proper monotone
        # step-up time series instead of a single-point bar — single-
        # point metrics flip its chart kind from line to bar. The last
        # logged value is still the selected epoch's value, so Overview
        # tab semantics are preserved.
        best_val_scalar = scalar_only(self.best_val_breakdown)
        log_metrics(
            {
                "val/best_score": self.best_selection_score,
                "val/best_epoch": float(self.best_epoch),
                **{f"val/best_{k}": v for k, v in best_val_scalar.items()},
            },
            step=epoch,
        )
        if self.epochs_since_improve >= patience:
            self.early_stopped = True
        return score


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
        "val_composite": breakdown.get("composite", 0.0),
        "val_score": selection_score,
        "val_feasible": 1.0,
        **{f"val_{k}": v for k, v in breakdown.items()},
    }
    return _TrainingHistory(
        best_state_dict={"smoke": True, "backbone": params.get("backbone")},
        best_val_composite=breakdown.get("composite", 0.0),
        best_val_score=selection_score,
        best_val_feasible=selection_score >= FEASIBLE_OFFSET,
        best_epoch=0,
        epochs_trained=1,
        early_stopped=False,
        feasible_epoch_count=1,
        curves=[smoke_curve],
        val_breakdown=breakdown,
        test_breakdown=breakdown,
        test_composite=breakdown.get("composite", 0.0),
        test_score=selection_score,
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
    # Handoff from train — what the previous phase produced that we're
    # tuning against. ``val_breakdown`` is what the train floor gate saw;
    # logging it here lets you sanity-check that tune is starting from
    # the right checkpoint without diffing eval_metrics.json by hand.
    train_val = _read_staging_breakdown(staging_dir, "val_breakdown") or {}
    train_test = _read_staging_breakdown(staging_dir, "test_breakdown") or {}
    logger.info(
        "forge.tune handoff: staging=%s task_id=%s train_val=%s train_test=%s",
        staging_dir,
        task_id,
        train_val,
        train_test,
    )

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
    # Tune phase floors == deploy floors (see ``FloorBundle.for_phase``),
    # so persisted ``tuned_*_score`` is computed once here and is the
    # exact value the deploy regression gate will compare against.
    tune_floors = spec.task.phase_floors("tune")
    tuned_val_score = feasibility_aware_score(
        tuned_val, tune_floors, spec.task.composite_weights
    )
    tuned_test_score_persist = feasibility_aware_score(
        tuned_test, tune_floors, spec.task.composite_weights
    )
    _update_eval_metrics(
        staging_dir,
        best=best,
        tuned_val=tuned_val,
        tuned_test=tuned_test,
        tuned_val_score=tuned_val_score,
        tuned_test_score=tuned_test_score_persist,
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
        # Mirror the search/train pattern: ``score`` is the actual
        # optimisation target (``study.best_trial.value``), ``composite``
        # is just the weighted-sum component. Logging both makes
        # tune↔search↔train numbers directly comparable.
        tuned_val_score = feasibility_aware_score(
            tuned_val, floors, task.composite_weights
        )
        tuned_test_score = feasibility_aware_score(
            tuned_test, floors, task.composite_weights
        )
        log_metrics({f"val/{k}": v for k, v in scalar_only(tuned_val).items()})
        log_metrics({f"test/{k}": v for k, v in scalar_only(tuned_test).items()})
        log_metrics({"val/score": tuned_val_score, "test/score": tuned_test_score})
        logger.info(
            "forge.tune winner: trial=%d val_score=%.4f test_score=%.4f "
            "val_composite=%.4f test_composite=%.4f params=%s "
            "val_breakdown=%s test_breakdown=%s",
            study.best_trial.number,
            tuned_val_score,
            tuned_test_score,
            tuned_val.get("composite", 0.0),
            tuned_test.get("composite", 0.0),
            best,
            {k: f"{v:.3f}" for k, v in scalar_only(tuned_val).items()},
            {k: f"{v:.3f}" for k, v in scalar_only(tuned_test).items()},
        )
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


def _update_eval_metrics(
    staging_dir: Path,
    *,
    best,
    tuned_val,
    tuned_test,
    tuned_val_score: float,
    tuned_test_score: float,
) -> None:
    """Splice the tuned breakdowns into ``eval_metrics.json``.

    ``tuned_*_score`` is the canonical feasibility-aware score computed
    against deploy floors; ``tuned_*_composite`` is the raw weighted
    sum. Both persisted so the regression gate uses score while the
    composite remains visible for analysis.
    """
    eval_path = staging_dir / "eval_metrics.json"
    existing = (
        json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    )
    existing.update(
        {
            "tuned_params": best,
            "tuned_val_breakdown": tuned_val,
            "tuned_test_breakdown": tuned_test,
            "tuned_val_composite": tuned_val.get("composite", 0.0),
            "tuned_test_composite": tuned_test.get("composite", 0.0),
            "tuned_val_score": tuned_val_score,
            "tuned_test_score": tuned_test_score,
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
    # Handoff from tune — what's actually about to be floor-gated. Logging
    # both the breakdown and the floor map together makes a failed deploy
    # gate self-explanatory: the next log line is the gate decision.
    logger.info(
        "forge.deploy handoff: staging=%s tuned_test=%s floors=%s tuned_inference=%s",
        staging_dir,
        tuned_test,
        floors_map,
        manifest.tuned_inference.model_dump(mode="json"),
    )
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

    # ``tuned_test_score`` is the canonical feasibility-aware score
    # persisted by tune. The regression gate compares this against the
    # previously-deployed model's score (not raw composite) so the gate
    # honours the same feasibility cliff Optuna optimised across.
    candidate_score = float(eval_metrics["tuned_test_score"])
    candidate_composite = float(eval_metrics.get("tuned_test_composite", 0.0))
    previous = read_latest_entry(
        latest_jsonl_path(spec.dataset.disease_id), model_id=spec.model_id
    )
    # Resolved once so the regression gate and the audit entry both see
    # the same baseline value — legacy rows are reconstructed from the
    # preserved breakdown so the comparison stays on a single scale.
    previous_score = (
        _previous_tuned_test_score(previous, spec) if previous is not None else None
    )
    if previous is not None:
        assert previous_score is not None
        if force:
            logger.warning(
                "deploy --force: skipping regression gate. candidate "
                "tuned_test_score=%.4f vs active %.4f (Δ=%+.4f, version %s). "
                "Do NOT use in production.",
                candidate_score,
                previous_score,
                candidate_score - previous_score,
                previous["version_tag"],
            )
        else:
            _check_regression(previous["version_tag"], previous_score, candidate_score)

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
        candidate_score=candidate_score,
        candidate_composite=candidate_composite,
        previous_score=previous_score,
    )
    append_latest_entry(latest_jsonl_path(spec.dataset.disease_id), entry)
    logger.info("appended LATEST.jsonl entry for version %s", tag)
    return stable_path


def _previous_tuned_test_score(previous: dict[str, Any], spec: ModelSpec) -> float:
    """Read the previous entry's feasibility-aware test score.

    New entries persist ``tuned_test_score`` directly. Older entries
    (deployed before the rename) only have ``tuned_test_breakdown`` plus
    a now-defunct ``test_score`` key that stored the raw composite —
    a different scale from today's feasibility-aware score. Comparing
    them directly would mix ``[0, 1]`` against ``(1, 2] ∪ (-inf, 0]``
    and silently mis-rank.

    For those legacy rows we recompute the score from the preserved
    breakdown using today's tune-phase floors and composite weights —
    the same formula the candidate was just scored with — so candidate
    and baseline land on the same scale. ``tuned_test_breakdown`` is
    preserved verbatim across the schema change, so the metric inputs
    themselves are stable.

    Missing both keys means the entry pre-dates breakdown persistence;
    fail loud rather than silently skipping the regression gate.
    """
    metrics = previous["metrics"]
    if "tuned_test_score" in metrics:
        return float(metrics["tuned_test_score"])
    breakdown = metrics.get("tuned_test_breakdown")
    if not breakdown:
        raise SystemExit(
            "regression gate: previous LATEST.jsonl entry "
            f"{previous.get('version_tag')!r} has no ``tuned_test_score`` "
            "and no ``tuned_test_breakdown`` to reconstruct from — entry "
            "pre-dates the current scoring schema. Re-deploy or remove "
            "the stale entry."
        )
    floors = spec.task.phase_floors("tune")
    return feasibility_aware_score(breakdown, floors, spec.task.composite_weights)


def _check_regression(
    previous_version: str, prev_score: float, candidate_score: float
) -> None:
    """Candidate must beat the previously-deployed score for the same model_id.

    Compares feasibility-aware scores (not raw composites): a candidate
    with higher composite that misses a deploy floor loses to a
    previously-deployed feasible model, matching what search / train /
    tune optimised for. ``FEASIBLE_OFFSET`` puts every feasible score in
    ``(1, 2]`` and every infeasible one in ``(-inf, 0]``, so the
    comparison naturally honours the floor cliff.
    """
    if candidate_score <= prev_score:
        raise SystemExit(
            f"regression gate failed: candidate tuned_test_score="
            f"{candidate_score:.4f} ≤ active {prev_score:.4f} "
            f"(version {previous_version}). Re-tune or retrain."
        )
    logger.info(
        "regression gate ok: tuned_test_score=%.4f > active %.4f (Δ=%+.4f)",
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
    candidate_score: float,
    candidate_composite: float,
    previous_score: float | None,
) -> dict[str, Any]:
    """Compose the JSONL row carrying the full deploy audit trail."""
    from datetime import UTC, datetime

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
            "best_val_composite": eval_metrics.get("best_val_composite"),
            "test_score": eval_metrics.get("test_score"),
            "test_composite": eval_metrics.get("test_composite"),
            "tuned_val_breakdown": eval_metrics.get("tuned_val_breakdown"),
            "tuned_test_breakdown": tuned_test,
            "tuned_test_score": candidate_score,
            "tuned_test_composite": candidate_composite,
            "best_epoch": eval_metrics.get("best_epoch"),
            "early_stopped": eval_metrics.get("early_stopped"),
            "epochs_trained": eval_metrics.get("epochs_trained"),
        },
        "previous_tuned_test_score": previous_score,
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


# === hp helpers ==========================================================


def _build_hp_with_class_weight(
    params: dict[str, Any], splits, spec: ModelSpec, device
) -> dict[str, Any]:
    """Materialise the per-batch ``hp`` dict, including the weight tensor.

    Reads ``params["class_weight"]`` (Optuna's pick: ``"none"`` /
    ``"inverse_freq"`` / ``"sqrt_inv_freq"``). When the chosen scheme
    requires per-class weights, builds the tensor once from
    ``splits.train`` label counts and stashes it under
    ``_class_weight_tensor`` (underscore prefix = derived, not a real
    hparam — keeps it out of JSON dumps that read ``params`` directly).

    No-op when ``class_weight`` is absent (e.g. legacy ModelSpec without
    the new hparam, or smoke runs that synthesise minimal params).
    """
    hp = dict(params)
    scheme = params.get("class_weight", "none")
    if scheme == "none":
        return hp
    tensor = class_weight_tensor_from_splits(
        splits, num_classes=len(spec.dataset.labels), scheme=scheme, device=device
    )
    if tensor is not None:
        hp["_class_weight_tensor"] = tensor
    return hp


# === lazy torch handle =================================================


_TORCH_MP_CONFIGURED = False


def _torch():
    global _TORCH_MP_CONFIGURED
    try:
        import torch  # noqa: F401

        if not _TORCH_MP_CONFIGURED:
            # macOS default ``file_descriptor`` strategy routes every
            # worker→main tensor through a Unix-socket FD; under Optuna
            # sweeps those FDs accumulate across trials and trip the
            # per-process ceiling (``kern.maxfilesperproc``). ``file_system``
            # uses /tmp-backed shm instead, dropping the leak vector.
            import torch.multiprocessing as _tmp

            _tmp.set_sharing_strategy("file_system")
            _TORCH_MP_CONFIGURED = True
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
