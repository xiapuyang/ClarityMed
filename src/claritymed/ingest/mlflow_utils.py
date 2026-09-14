"""Shared MLflow helpers for model training and tuning.

Hoisted from ``ingest/symptoms/mlflow_utils.py`` so the vision feature
can reuse the same experiment-naming + tracking-URI conventions
without a cross-feature dependency. Symptoms callers updated; vision
training pipeline (Unit 6) imports the same helpers.

mlflow is imported lazily so this module is safe to import without the
extras installed.  Callers that actually invoke the public functions
need ``uv sync --extra symptoms-server`` (or the equivalent vision
extra once one lands).

Experiment naming convention: ``claritymed-<feature>-<dataset_id>``.
Runs are tagged with ``run_type`` (``train`` / ``tune``) and
``feature`` so the MLflow UI can filter them.

Tracking DB is **shared across every (feature, dataset)** at
``CLARITYMED_HOME/tracking/mlflow.db``. One DB, many experiments —
disambiguation lives in the experiment name. Start the UI with::

    mlflow ui --backend-store-uri sqlite:///$HOME/.claritymed/tracking/mlflow.db --port 5000

Historic per-dataset databases under ``models/<feature>/<dataset>/run/mlflow.db``
remain on disk but are no longer written to; either ``mlflow ui`` them
separately or migrate runs by hand if the history matters.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generator

from claritymed import config as _cfg

if TYPE_CHECKING:
    import mlflow as _mlflow_t

TASK_ID_TAG = "claritymed.task_id"


def generate_task_id() -> str:
    """Return a fresh task_id (one per pipeline execution).

    Format: ``YYYYMMDDTHHMMSSZ-<8 hex>``. The timestamp prefix makes the
    id naturally sortable; the random suffix avoids collisions when two
    pipeline runs start in the same second. Stored verbatim in:

    * MLflow run tags (``claritymed.task_id``) — one tag per search /
      train / tune run that shares this lineage.
    * Optuna trial ``user_attrs`` — set inside the objective function
      so every trial carries the task_id of whoever launched it.
    * Staging ``provenance.json`` and the deploy ``LATEST.jsonl`` row.

    With these four touch points an operator can pivot in any direction
    from a single task_id and reach every artifact the pipeline produced.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(4)
    return f"{stamp}-{suffix}"


def experiment_name(feature: str, dataset_id: str) -> str:
    """Return the canonical MLflow experiment name for a (feature, dataset)."""
    return f"claritymed-{feature}-{dataset_id}"


def tracking_uri() -> str:
    """Return the shared SQLite tracking URI for every feature + dataset.

    A single DB keeps the MLflow UI one-click ("open this URI, filter by
    experiment") and avoids the per-feature ramp-up cost of pointing the
    UI at N different files. SQLite handles concurrent writers across
    studies fine — Optuna's storage at :func:`optuna_storage_uri` shares
    the same locking model and the same posture.
    """
    db = _cfg.CLARITYMED_HOME / "tracking" / "mlflow.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db}"


def optuna_storage_uri() -> str:
    """Return the shared Optuna storage URI for every study.

    Studies are disambiguated by ``study_name`` (e.g.
    ``claritymed-vision-busi-hparam``, ``claritymed-vision-busi-tune``)
    so one DB carries every search/tune across every feature. Same
    rationale as :func:`tracking_uri` — one place to point ``optuna``
    CLI, one place to back up.
    """
    db = _cfg.CLARITYMED_HOME / "tracking" / "optuna.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db}"


def study_name(feature: str, dataset_id: str, phase: str) -> str:
    """Return the canonical Optuna study name.

    Args:
        feature: Feature id (``"vision"`` / ``"symptoms"``).
        dataset_id: Dataset id (``"busi"`` / ``"ddxplus"``).
        phase: ``"hparam"`` for HP search, ``"tune"`` for inference
            param tuning. Any new phase (e.g. ``"calibration"``) is one
            string here and one matching caller — no schema change.
    """
    return f"claritymed-{feature}-{dataset_id}-{phase}"


def _tracking_uri(feature: str, dataset_id: str) -> str:  # noqa: ARG001
    """Deprecated — kept so legacy imports keep working.

    The (feature, dataset_id) signature is preserved but ignored; the
    shared :func:`tracking_uri` is returned. Remove once all call sites
    migrate.
    """
    return tracking_uri()


def _mlflow() -> "_mlflow_t":
    try:
        import mlflow

        return mlflow
    except ImportError as exc:
        raise SystemExit(
            "mlflow not installed — run `uv sync --extra symptoms-server` "
            "(or the equivalent vision extra)."
        ) from exc


def _finite(v: float) -> float | None:
    """Return v if finite, None otherwise (MLflow rejects NaN/inf)."""
    return v if math.isfinite(v) else None


@contextlib.contextmanager
def mlflow_run(
    feature: str,
    dataset_id: str,
    run_name: str | None = None,
    run_type: str = "train",
    params: dict | None = None,
    nested: bool = False,
    tags: dict[str, str] | None = None,
) -> Generator["_mlflow_t.ActiveRun", None, None]:
    """Start an MLflow run scoped to a (feature, dataset) experiment.

    Args:
        feature: Feature identifier (``"symptoms"`` / ``"vision"``).
        dataset_id: Dataset identifier, e.g. ``"ddxplus"`` / ``"busi"``.
        run_name: Human-readable run name shown in the UI.
        run_type: Tag value for ``run_type`` (``"train"`` or ``"tune"``).
        params: Hyperparameters to log at run start.
        nested: Pass ``True`` for child runs inside a parent run.
        tags: Extra tags merged on top of the defaults — used to attach
            the lineage ``claritymed.task_id`` to every run in one
            pipeline execution.

    Yields:
        The active ``mlflow.ActiveRun``.
    """
    mlflow = _mlflow()
    mlflow.set_tracking_uri(_tracking_uri(feature, dataset_id))
    mlflow.set_experiment(f"claritymed-{feature}-{dataset_id}")
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tag("run_type", run_type)
        mlflow.set_tag("dataset_id", dataset_id)
        mlflow.set_tag("feature", feature)
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        if params:
            mlflow.log_params(params)
        yield run


@contextlib.contextmanager
def symptom_run(
    dataset_id: str,
    run_name: str | None = None,
    run_type: str = "train",
    params: dict | None = None,
    nested: bool = False,
) -> Generator["_mlflow_t.ActiveRun", None, None]:
    """Backwards-compatible alias for ``mlflow_run(feature='symptoms', …)``.

    Kept so the existing symptoms call sites compile without churn.
    Prefer :func:`mlflow_run` in new code.
    """
    with mlflow_run(
        "symptoms",
        dataset_id,
        run_name=run_name,
        run_type=run_type,
        params=params,
        nested=nested,
    ) as run:
        yield run


def log_eval_metrics(
    metrics: Any,
    prefix: str = "",
    step: int | None = None,
) -> None:
    """Log all finite dataclass fields of ``metrics`` to the active MLflow run.

    Args:
        metrics: Any dataclass whose fields are numeric.
        prefix: Optional prefix, e.g. ``"test/"`` or ``"val/"``.
        step: Step index (epoch number for per-epoch val metrics).
    """
    mlflow = _mlflow()
    logged = {}
    for field in dataclasses.fields(metrics):
        v = getattr(metrics, field.name)
        fv = _finite(float(v))
        if fv is not None:
            logged[f"{prefix}{field.name}"] = fv
    mlflow.log_metrics(logged, step=step)


def log_epoch(
    epoch: int,
    sym_loss: float,
    pat_loss: float,
    stop_loss: float,
    val_metrics: Any,
    score: float,
) -> None:
    """Log per-epoch training and validation metrics (symptoms-shaped).

    Vision training uses :func:`log_eval_metrics` directly with its own
    per-epoch loss dict; this helper stays symptoms-specific because
    the symptom / pathology / stop loss decomposition has no vision
    analogue.
    """
    mlflow = _mlflow()
    train_metrics = {
        "train/sym_loss": sym_loss,
        "train/pat_loss": pat_loss,
        "train/stop_loss": stop_loss,
        "train/total_loss": sym_loss + pat_loss + stop_loss,
    }
    if (s := _finite(score)) is not None:
        train_metrics["val/score"] = s
    mlflow.log_metrics(train_metrics, step=epoch)
    log_eval_metrics(val_metrics, prefix="val/", step=epoch)
