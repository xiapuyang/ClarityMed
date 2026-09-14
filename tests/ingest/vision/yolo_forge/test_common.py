"""Tests for :mod:`claritymed.ingest.vision.yolo_forge.common` helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.ingest.vision.yolo_forge import common
from claritymed.ingest.vision.yolo_forge.spec import (
    YoloModelSpec,
)


# --- parse_phases -------------------------------------------------------


def test_parse_phases_default_is_all_in_order() -> None:
    assert common.parse_phases(None) == list(common.ALL_PHASES)
    assert common.parse_phases("") == list(common.ALL_PHASES)


def test_parse_phases_reorders_to_canonical() -> None:
    """`deploy,prepare,tune` typed in random order → canonical sequence."""
    assert common.parse_phases("deploy,prepare,tune") == ["prepare", "tune", "deploy"]


def test_parse_phases_rejects_unknown_token() -> None:
    with pytest.raises(SystemExit, match="unknown phase"):
        common.parse_phases("prepare,refrigerate,train")


# --- resolve_model_spec --------------------------------------------------


def test_resolve_model_spec_happy_path() -> None:
    spec = common.resolve_model_spec(
        "claritymed.ingest.vision.rsna_pneumonia_yolo.models.yolov8n_v1:RSNA_YOLOV8N_V1"
    )
    assert isinstance(spec, YoloModelSpec)
    assert spec.model_id == "rsna_pneumonia_yolov8n_v1"


def test_resolve_model_spec_missing_colon_fails_loud() -> None:
    with pytest.raises(SystemExit, match="got no ':'"):
        common.resolve_model_spec("claritymed.ingest.vision.rsna_pneumonia_yolo")


def test_resolve_model_spec_bad_module_fails_loud() -> None:
    with pytest.raises(SystemExit, match="cannot import"):
        common.resolve_model_spec("definitely_not_a_package:THING")


def test_resolve_model_spec_bad_attr_fails_loud() -> None:
    with pytest.raises(SystemExit, match="no attribute"):
        common.resolve_model_spec(
            "claritymed.ingest.vision.yolo_forge.spec:THIS_DOES_NOT_EXIST"
        )


def test_resolve_model_spec_wrong_type_fails_loud() -> None:
    """Pointing at a non-YoloModelSpec attribute → fail loud, no silent cast."""
    with pytest.raises(SystemExit, match="expected YoloModelSpec"):
        common.resolve_model_spec(
            "claritymed.ingest.vision.yolo_forge.common:ALL_PHASES"
        )


# --- audit log roundtrip ------------------------------------------------


def test_append_and_read_last_entry_filters_by_model_id(tmp_path: Path) -> None:
    log = tmp_path / "LATEST.jsonl"
    common.append_entry(log, {"model_id": "a", "v": 1})
    common.append_entry(log, {"model_id": "b", "v": 2})
    common.append_entry(log, {"model_id": "a", "v": 3})
    assert common.read_last_entry(log, model_id="a") == {"model_id": "a", "v": 3}
    assert common.read_last_entry(log, model_id="b") == {"model_id": "b", "v": 2}
    assert common.read_last_entry(log, model_id="missing") is None


def test_read_last_entry_missing_file_returns_none(tmp_path: Path) -> None:
    assert common.read_last_entry(tmp_path / "nope.jsonl", model_id="x") is None


# --- staging paths -----------------------------------------------------


def test_staging_dir_uses_timestamp_under_disease_root(
    monkeypatch, tmp_path: Path
) -> None:
    """staging_dir lays out ``<home>/models/vision/<dataset>/run/<id>_<ts>/``."""
    monkeypatch.setattr(common._cfg, "CLARITYMED_HOME", tmp_path)
    out = common.staging_dir(dataset_id="ds1", model_id="m1")
    assert out.parent.parent == tmp_path / "models" / "vision" / "ds1"
    assert out.name.startswith("m1_")


def test_latest_staging_dir_picks_freshest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common._cfg, "CLARITYMED_HOME", tmp_path)
    run_root = tmp_path / "models" / "vision" / "ds1" / "run"
    run_root.mkdir(parents=True)
    (run_root / "m1_20250101T000000Z").mkdir()
    (run_root / "m1_20260101T000000Z").mkdir()
    (run_root / "m2_20270101T000000Z").mkdir()  # different model
    out = common.latest_staging_dir(dataset_id="ds1", model_id="m1")
    assert out.name == "m1_20260101T000000Z"


def test_latest_staging_dir_no_run_dir_fails_loud(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(common._cfg, "CLARITYMED_HOME", tmp_path)
    with pytest.raises(SystemExit, match="no run/ dir"):
        common.latest_staging_dir(dataset_id="never", model_id="x")
