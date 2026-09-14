"""task_id lineage: format, threading through every artifact."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from claritymed.ingest.mlflow_utils import TASK_ID_TAG, generate_task_id


def test_generate_task_id_format_is_pinned() -> None:
    """Format: YYYYMMDDTHHMMSSZ-<8 hex>."""
    a = generate_task_id()
    b = generate_task_id()
    assert re.match(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$", a), a
    assert a != b, "secrets.token_hex collision — astronomically unlikely"
    assert a[:16] <= b[:16]


def test_task_id_tag_constant() -> None:
    assert TASK_ID_TAG == "claritymed.task_id"


def _reload_forge_for_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    import importlib

    from claritymed import config as _cfg

    importlib.reload(_cfg)
    for mod_name in (
        "claritymed.ingest.mlflow_utils",
        "claritymed.ingest.vision.forge.common",
        "claritymed.ingest.vision.forge.framework",
        "claritymed.ingest.vision.busi.dataset_spec",
        "claritymed.ingest.vision.busi.models.unet_resnet50",
    ):
        importlib.reload(__import__(mod_name, fromlist=["_"]))


def test_pipeline_threads_single_task_id_through_every_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance.json on the staging dir carries the same task_id through
    every phase. (Smoke doesn't write LATEST.jsonl, so the lineage check
    is against the staging dir — for real runs, the same task_id also
    lands in LATEST.jsonl from the deploy phase.)"""
    _reload_forge_for_home(monkeypatch, tmp_path)
    from claritymed.ingest.vision.busi.models.unet_resnet50 import UNET_RESNET50
    from claritymed.ingest.vision.forge.common import ALL_PHASES
    from claritymed.ingest.vision.forge.framework import run_pipeline

    pinned_task_id = "20260101T120000Z-abcdef12"
    staging = run_pipeline(
        UNET_RESNET50,
        phases=ALL_PHASES,
        trials=1,
        search_epochs=1,
        max_epochs=1,
        patience=1,
        tune_trials=2,
        smoke=True,
        staging_dir=None,
        task_id=pinned_task_id,
    )
    assert staging is not None

    provenance = json.loads((staging / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["task_id"] == pinned_task_id
    assert provenance["tune"]["task_id"] == pinned_task_id
    assert provenance["optuna"]["search_trial_task_id"] == pinned_task_id


def test_pipeline_generates_task_id_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _reload_forge_for_home(monkeypatch, tmp_path)
    from claritymed.ingest.vision.busi.models.unet_resnet50 import UNET_RESNET50
    from claritymed.ingest.vision.forge.common import ALL_PHASES
    from claritymed.ingest.vision.forge.framework import run_pipeline

    staging = run_pipeline(
        UNET_RESNET50,
        phases=ALL_PHASES,
        trials=1,
        search_epochs=1,
        max_epochs=1,
        patience=1,
        tune_trials=2,
        smoke=True,
        staging_dir=None,
        task_id=None,
    )
    assert staging is not None

    provenance = json.loads((staging / "provenance.json").read_text(encoding="utf-8"))
    task_id = provenance["task_id"]
    assert re.match(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$", task_id), task_id
    assert provenance["tune"]["task_id"] == task_id
