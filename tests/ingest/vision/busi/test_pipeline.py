"""Pipeline orchestrator: phase parsing + smoke end-to-end (BUSI U-Net).

Forge's :func:`run_pipeline` is the orchestrator; this file drives it
with the BUSI U-Net :class:`ModelSpec`.
"""

from __future__ import annotations

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


def test_smoke_pipeline_stops_at_staging_without_touching_stable_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Smoke must verify wiring without leaving fake state for vision-server."""
    _reload_forge_for_home(monkeypatch, tmp_path)
    from claritymed.ingest.vision.busi.models.unet_resnet50 import UNET_RESNET50
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
    )
    # Smoke deploy returns the staging dir (not a stable path).
    assert staging is not None
    assert staging.exists()
    assert staging.parent.name == "run"
    # The staging dir itself still has every artifact a real run would write.
    for required in (
        "manifest.json",
        "weights.pt",
        "eval_metrics.json",
        "provenance.json",
    ):
        assert (staging / required).exists(), f"staging missing {required!r}"

    disease_root = staging.parent.parent
    # No versioned stable directory.
    assert not list(disease_root.glob("breast_busi_unet_v1__*"))
    # No <model_id> stable symlink.
    assert not (disease_root / "breast_busi_unet_v1").exists()
    # No LATEST.jsonl.
    assert not (disease_root / "LATEST.jsonl").exists()


def test_smoke_pipeline_is_idempotent_does_not_self_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two smoke runs in a row must both succeed — smoke writes no LATEST.jsonl,
    so the regression gate has nothing to compare against."""
    _reload_forge_for_home(monkeypatch, tmp_path)
    from claritymed.ingest.vision.busi.models.unet_resnet50 import UNET_RESNET50
    from claritymed.ingest.vision.forge.framework import run_pipeline

    kwargs = dict(
        phases=ALL_PHASES,
        trials=1,
        search_epochs=1,
        max_epochs=1,
        patience=1,
        tune_trials=2,
        smoke=True,
        staging_dir=None,
    )
    first = run_pipeline(UNET_RESNET50, **kwargs)
    second = run_pipeline(UNET_RESNET50, **kwargs)
    # Both runs land somewhere under run/, both succeed, and no
    # LATEST.jsonl exists to cross-block them. (Same-second runs may
    # share a staging dir; the wiring contract is "no self-block", not
    # "distinct paths".)
    assert first is not None and first.exists()
    assert second is not None and second.exists()
    assert not (first.parent.parent / "LATEST.jsonl").exists()
