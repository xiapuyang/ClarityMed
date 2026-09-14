"""Symptoms-side compatibility shim — re-exports from ``ingest/mlflow_utils.py``.

The helpers moved up one level (Unit 6 — see
``docs/plans/2026-06-14-001-feat-vision-detection-plan.md``) so the
vision training pipeline can reuse them without a cross-feature
dependency. This file remains so existing symptoms call sites keep
working; new code should import from ``claritymed.ingest.mlflow_utils``
directly.
"""

from claritymed.ingest.mlflow_utils import (
    log_epoch,
    log_eval_metrics,
    mlflow_run,
    symptom_run,
)

EXPERIMENT_PREFIX = "claritymed-symptoms"

__all__ = [
    "EXPERIMENT_PREFIX",
    "log_epoch",
    "log_eval_metrics",
    "mlflow_run",
    "symptom_run",
]
