"""Pipeline orchestrator: phase parsing + smoke end-to-end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claritymed.ingest.vision.busi.pipeline import ALL_PHASES, _parse_phases


def test_parse_phases_reorders_to_canonical_order() -> None:
    """An operator typing "deploy,tune" gets a tune→deploy run, not a failure."""
    assert _parse_phases("deploy,tune") == ("tune", "deploy")


def test_parse_phases_drops_duplicates() -> None:
    """The pipeline should be idempotent under repeated phase names."""
    assert _parse_phases("train,train,tune") == ("train", "tune")


def test_parse_phases_rejects_typos() -> None:
    """A typo must surface as a clear error, not a silent skip."""
    with pytest.raises(SystemExit) as exc:
        _parse_phases("trian")
    assert "trian" in str(exc.value)


def test_parse_phases_default_is_full_chain() -> None:
    assert _parse_phases(",".join(ALL_PHASES)) == ALL_PHASES


def test_smoke_pipeline_writes_versioned_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end smoke: search→train→tune→deploy lands in a versioned dir."""
    # Redirect CLARITYMED_HOME so the smoke run doesn't pollute the dev box.
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    # Re-import so module-level _cfg.CLARITYMED_HOME picks up the env.
    import importlib

    from claritymed import config as _cfg

    importlib.reload(_cfg)

    # Reload every module that captured CLARITYMED_HOME at import time.
    for mod_name in (
        "claritymed.ingest.mlflow_utils",
        "claritymed.ingest.vision.busi.train",
        "claritymed.ingest.vision.busi.hparam",
        "claritymed.ingest.vision.busi.tune",
        "claritymed.ingest.vision.busi.deploy",
        "claritymed.ingest.vision.busi.pipeline",
    ):
        importlib.reload(__import__(mod_name, fromlist=["_"]))

    from claritymed.ingest.vision.busi.pipeline import run_pipeline as _run_pipeline

    stable = _run_pipeline(
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
    # Versioned naming: "<model_id>__<timestamp>"
    assert "__" in stable.name
    assert stable.name.startswith("breast_busi_unet_v1__")
    # LATEST.jsonl has exactly one entry after the first deploy.
    latest = stable.parent / "LATEST.jsonl"
    assert latest.exists()
    rows = [
        json.loads(line)
        for line in latest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    entry = rows[0]
    # Provenance carries every join key needed for observability.
    for required in ("mlflow", "optuna", "best_hp", "best_inference_params", "metrics"):
        assert required in entry, f"LATEST.jsonl entry missing {required!r}"
    assert entry["mlflow"]["tracking_uri"]
    assert entry["optuna"]["storage_uri"]


def test_smoke_pipeline_second_deploy_blocked_by_regression_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke values are deterministic; second deploy with same score must fail."""
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
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

    from claritymed.ingest.vision.busi.pipeline import run_pipeline as _run_pipeline

    _run_pipeline(
        phases=ALL_PHASES,
        trials=1,
        search_epochs=1,
        max_epochs=1,
        patience=1,
        tune_trials=2,
        smoke=True,
        staging_dir=None,
    )
    with pytest.raises(SystemExit) as exc:
        _run_pipeline(
            phases=ALL_PHASES,
            trials=1,
            search_epochs=1,
            max_epochs=1,
            patience=1,
            tune_trials=2,
            smoke=True,
            staging_dir=None,
        )
    assert "regression gate failed" in str(exc.value)
