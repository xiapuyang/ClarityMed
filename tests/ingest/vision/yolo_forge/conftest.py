"""Shared yolo_forge test fixtures.

Stubs out :func:`mlflow_phase_run` by default so framework-level tests
don't hit the real MLflow tracking DB at
``~/.claritymed/tracking/mlflow.db`` (or wherever ``CLARITYMED_HOME``
points). Tests that specifically exercise the mlflow shim opt out by
re-patching the attribute themselves.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from claritymed.ingest.vision.yolo_forge import framework


@contextmanager
def _noop_mlflow_run(**kwargs):  # noqa: ARG001 — interface match
    yield None


@pytest.fixture(autouse=True)
def _stub_mlflow_phase_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default stub. mlflow-specific tests overwrite this with their own patch."""
    monkeypatch.setattr(framework, "mlflow_phase_run", _noop_mlflow_run)
