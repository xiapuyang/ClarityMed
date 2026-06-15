"""task_id lineage: format, threading through every artifact."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from claritymed.ingest.mlflow_utils import TASK_ID_TAG, generate_task_id


# --- generate_task_id format -----------------------------------------------


def test_generate_task_id_format_is_pinned() -> None:
    """Format: YYYYMMDDTHHMMSSZ-<8 hex>. Pinned because LATEST.jsonl + docs
    rely on the shape for grep / sort recipes.
    """
    a = generate_task_id()
    b = generate_task_id()
    assert re.match(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$", a), a
    assert a != b, "secrets.token_hex collision — astronomically unlikely"
    # Timestamp prefix sorts chronologically across seconds. Within the
    # same second the random suffix breaks the tie, which is fine — the
    # uniqueness story is the load-bearing one, not strict ordering.
    assert a[:16] <= b[:16]


def test_task_id_tag_constant() -> None:
    """The MLflow tag key is part of the public contract — pin it."""
    assert TASK_ID_TAG == "claritymed.task_id"


# --- pipeline threads a single task_id end-to-end --------------------------


def _reload_pipeline_modules() -> None:
    """Reload the pipeline modules so they pick up CLARITYMED_HOME from env."""
    import importlib

    from claritymed import config as _cfg

    importlib.reload(_cfg)
    for mod_name in (
        "claritymed.ingest.mlflow_utils",
        "claritymed.ingest.vision.busi.train",
        "claritymed.ingest.vision.busi.hparam",
        "claritymed.ingest.vision.busi.tune",
        "claritymed.ingest.vision.busi.deploy",
        "claritymed.ingest.vision.busi.pipeline",
    ):
        importlib.reload(__import__(mod_name, fromlist=["_"]))


def test_pipeline_threads_single_task_id_through_every_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One task_id appears in provenance (top + tune block) + LATEST.jsonl."""
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    _reload_pipeline_modules()

    from claritymed.ingest.vision.busi.pipeline import ALL_PHASES, run_pipeline

    pinned_task_id = "20260101T120000Z-abcdef12"
    stable = run_pipeline(
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
    assert stable is not None

    provenance = json.loads((stable / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["task_id"] == pinned_task_id
    assert provenance["tune"]["task_id"] == pinned_task_id
    # search_trial_task_id matches: smoke train invents a synthetic trial
    # owned by the same task_id, mimicking the strict-task happy path.
    assert provenance["optuna"]["search_trial_task_id"] == pinned_task_id

    latest = stable.parent / "LATEST.jsonl"
    entry = json.loads(latest.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["task_id"] == pinned_task_id
    assert entry["optuna"]["search_trial_task_id"] == pinned_task_id


def test_pipeline_generates_task_id_when_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --task-id the orchestrator mints one and uses it everywhere."""
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    _reload_pipeline_modules()

    from claritymed.ingest.vision.busi.pipeline import ALL_PHASES, run_pipeline

    stable = run_pipeline(
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
    assert stable is not None
    entry = json.loads(
        (stable.parent / "LATEST.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    task_id = entry["task_id"]
    assert re.match(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$", task_id), task_id

    provenance = json.loads((stable / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["task_id"] == task_id
    assert provenance["tune"]["task_id"] == task_id
