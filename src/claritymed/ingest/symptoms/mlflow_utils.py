"""Shared MLflow helpers for symptoms model training and tuning.

mlflow is imported lazily so this module is safe to import without the
``symptoms-server`` extra installed.  Callers that actually invoke the
public functions need ``uv sync --extra symptoms-server``.

Experiment naming convention: ``claritymed-symptoms-<dataset_id>``.
Runs are tagged with ``run_type`` (``train`` / ``tune``) so the MLflow
UI can filter them without needing separate experiments.

Tracking DB per dataset: ``CLARITYMED_HOME/models/symptoms/<dataset_id>/run/mlflow.db``.
Start the UI with::

    mlflow ui --backend-store-uri sqlite:///$HOME/.claritymed/models/symptoms/ddxplus/run/mlflow.db --port 5000
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from typing import TYPE_CHECKING, Generator

from claritymed import config as _cfg

if TYPE_CHECKING:
    import mlflow as _mlflow_t

    from claritymed.ingest.symptoms.typed_basd import EvalMetrics

EXPERIMENT_PREFIX = "claritymed-symptoms"


def _tracking_uri(dataset_id: str) -> str:
    """SQLite DB under each dataset's run dir, CWD-independent."""
    db = _cfg.CLARITYMED_HOME / "models" / "symptoms" / dataset_id / "run" / "mlflow.db"
    return f"sqlite:///{db}"


def _mlflow() -> "_mlflow_t":
    try:
        import mlflow

        return mlflow
    except ImportError as exc:
        raise SystemExit(
            "mlflow not installed — run `uv sync --extra symptoms-server`."
        ) from exc


def _finite(v: float) -> float | None:
    """Return v if finite, None otherwise (MLflow rejects NaN/inf)."""
    return v if math.isfinite(v) else None


@contextlib.contextmanager
def symptom_run(
    dataset_id: str,
    run_name: str | None = None,
    run_type: str = "train",
    params: dict | None = None,
    nested: bool = False,
) -> Generator["_mlflow_t.ActiveRun", None, None]:
    """Start an MLflow run scoped to a dataset experiment.

    Args:
        dataset_id: Dataset identifier, e.g. ``"ddxplus"``.
        run_name: Human-readable run name shown in the UI.
        run_type: Tag value for ``run_type`` (``"train"`` or ``"tune"``).
        params: Hyperparameters to log at run start.
        nested: Pass ``True`` for child runs inside a parent run.

    Yields:
        The active ``mlflow.ActiveRun``.
    """
    mlflow = _mlflow()
    mlflow.set_tracking_uri(_tracking_uri(dataset_id))
    mlflow.set_experiment(f"{EXPERIMENT_PREFIX}-{dataset_id}")
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tag("run_type", run_type)
        mlflow.set_tag("dataset_id", dataset_id)
        if params:
            mlflow.log_params(params)
        yield run


def log_eval_metrics(
    metrics: "EvalMetrics",
    prefix: str = "",
    step: int | None = None,
) -> None:
    """Log all finite EvalMetrics fields to the active MLflow run.

    Args:
        metrics: Eval result from ``interactive_eval``.
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
    val_metrics: "EvalMetrics",
    score: float,
) -> None:
    """Log per-epoch training and validation metrics.

    Args:
        epoch: 0-based epoch index (used as MLflow step).
        sym_loss: Average symptom-head loss for this epoch.
        pat_loss: Average pathology-head loss for this epoch.
        stop_loss: Average stop-head loss for this epoch.
        val_metrics: Validation split eval metrics.
        score: Early-stopping composite score (DDF1 or DSR-penalty).
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
