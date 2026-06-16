"""Smoke pipeline end-to-end on the chest CT classifier ModelSpec.

Mirrors the BUSI smoke pipeline test (``tests/ingest/vision/busi/
test_pipeline.py``) so the forge framework's task polymorphism is
exercised against the cls-only path too — without this, every code
path in forge would only ever be hit on the cls+seg shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claritymed.ingest.vision.forge.common import ALL_PHASES


def _reload_forge_for_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    import importlib

    from claritymed import config as _cfg

    importlib.reload(_cfg)
    for mod_name in (
        "claritymed.ingest.mlflow_utils",
        "claritymed.ingest.vision.forge.common",
        "claritymed.ingest.vision.forge.framework",
        "claritymed.ingest.vision.chest_ct.dataset_spec",
        "claritymed.ingest.vision.chest_ct.models.resnet50_v1",
    ):
        importlib.reload(__import__(mod_name, fromlist=["_"]))


def test_smoke_pipeline_stops_at_staging_for_chest_ct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same smoke-no-promote contract as BUSI, exercised on the cls-only path."""
    _reload_forge_for_home(monkeypatch, tmp_path)
    from claritymed.ingest.vision.chest_ct.models.resnet50_v1 import RESNET50_V1
    from claritymed.ingest.vision.forge.framework import run_pipeline

    staging = run_pipeline(
        RESNET50_V1,
        phases=ALL_PHASES,
        trials=1,
        search_epochs=1,
        max_epochs=1,
        patience=1,
        tune_trials=2,
        smoke=True,
        staging_dir=None,
    )
    assert staging is not None
    assert staging.exists()
    assert staging.parent.name == "run"

    # The cls-only manifest is the per-task artifact that proves the
    # classification path was exercised (not the cls+seg one).
    manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task"] == "classification"

    disease_root = staging.parent.parent
    assert not list(disease_root.glob("lung_chest_ct_resnet50_v1__*"))
    assert not (disease_root / "lung_chest_ct_resnet50_v1").exists()
    assert not (disease_root / "LATEST.jsonl").exists()


def test_chest_ct_modelspec_has_classification_task_shape() -> None:
    """Surface checks on the ModelSpec: task type, modality, critical labels."""
    from claritymed.ingest.vision.chest_ct.models.resnet50_v1 import RESNET50_V1

    assert RESNET50_V1.task.name == "classification"
    assert RESNET50_V1.dataset.accepted_modality == "ct"
    assert set(RESNET50_V1.task.critical_labels) == {
        "adenocarcinoma",
        "large_cell_carcinoma",
        "squamous_cell_carcinoma",
    }
    # Composite weights / floors carry the chest_ct metric names.
    assert "cancer_recall" in RESNET50_V1.task.composite_weights
    assert "cancer_recall" in RESNET50_V1.task.floors.deploy
    assert "dice" not in RESNET50_V1.task.floors.deploy
