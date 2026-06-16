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


def test_smoke_pipeline_writes_versioned_deploy_for_chest_ct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full pipeline against the ResNet-50 chest CT spec."""
    _reload_forge_for_home(monkeypatch, tmp_path)
    from claritymed.ingest.vision.chest_ct.models.resnet50_v1 import RESNET50_V1
    from claritymed.ingest.vision.forge.framework import run_pipeline

    stable = run_pipeline(
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
    assert stable is not None
    assert stable.exists()
    assert stable.name.startswith("lung_chest_ct_resnet50_v1__")

    latest = stable.parent / "LATEST.jsonl"
    rows = [
        json.loads(line)
        for line in latest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    entry = rows[0]
    assert entry["model_id"] == "lung_chest_ct_resnet50_v1"
    assert entry["disease_id"] == "lung_cancer_chest_ct"
    # Floors map mentions cancer_recall, not malignant_recall (task-specific).
    assert "cancer_recall" in entry["floors_passed"]
    assert "dice" not in entry["floors_passed"]
    assert entry["floors_passed"]["cancer_recall"] is True
    assert entry["floors_passed"]["accuracy"] is True


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
