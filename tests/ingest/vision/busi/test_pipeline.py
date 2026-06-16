"""Pipeline orchestrator: phase parsing + smoke end-to-end (BUSI U-Net).

Forge's :func:`run_pipeline` is the orchestrator; this file drives it
with the BUSI U-Net :class:`ModelSpec`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claritymed.ingest.vision.forge.common import ALL_PHASES, parse_phases


def test_parse_phases_reorders_to_canonical_order() -> None:
    """An operator typing "deploy,tune" gets a tune→deploy run, not a failure."""
    assert parse_phases("deploy,tune") == ("tune", "deploy")


def test_parse_phases_drops_duplicates() -> None:
    assert parse_phases("train,train,tune") == ("train", "tune")


def test_parse_phases_rejects_typos() -> None:
    with pytest.raises(SystemExit) as exc:
        parse_phases("trian")
    assert "trian" in str(exc.value)


def test_parse_phases_default_is_full_chain() -> None:
    assert parse_phases(",".join(ALL_PHASES)) == ALL_PHASES


def _reload_forge_for_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reload forge + busi spec modules so they pick up CLARITYMED_HOME."""
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


def test_smoke_pipeline_writes_versioned_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end smoke: search→train→tune→deploy lands in a versioned dir."""
    _reload_forge_for_home(monkeypatch, tmp_path)
    from claritymed.ingest.vision.busi.models.unet_resnet50 import UNET_RESNET50
    from claritymed.ingest.vision.forge.framework import run_pipeline

    stable = run_pipeline(
        UNET_RESNET50,
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
    assert "__" in stable.name
    assert stable.name.startswith("breast_busi_unet_v1__")
    latest = stable.parent / "LATEST.jsonl"
    assert latest.exists()
    rows = [
        json.loads(line)
        for line in latest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    entry = rows[0]
    for required in ("mlflow", "optuna", "best_hp", "best_inference_params", "metrics"):
        assert required in entry, f"LATEST.jsonl entry missing {required!r}"
    assert entry["mlflow"]["tracking_uri"]
    assert entry["optuna"]["storage_uri"]


def test_smoke_pipeline_second_deploy_blocked_by_regression_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke values are deterministic; second deploy with same score must fail."""
    _reload_forge_for_home(monkeypatch, tmp_path)
    from claritymed.ingest.vision.busi.models.unet_resnet50 import UNET_RESNET50
    from claritymed.ingest.vision.forge.framework import run_pipeline

    run_pipeline(
        UNET_RESNET50,
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
        run_pipeline(
            UNET_RESNET50,
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
