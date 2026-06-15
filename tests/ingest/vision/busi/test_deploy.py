"""Deploy phase: floor gate, regression gate, YAML patcher, LATEST.jsonl."""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.ingest.vision.busi.deploy import (
    FLOOR_ACCURACY,
    FLOOR_DICE,
    FLOOR_MALIGNANT_RECALL,
    _build_latest_entry,
    _check_floors,
    _read_latest_entry,
    _replace_model_fields,
)


# --- floor gate -----------------------------------------------------------


def test_floor_gate_passes_when_breakdown_clears_every_threshold() -> None:
    """Sanity: a breakdown above every floor must not raise."""
    _check_floors(
        {
            "malignant_recall": FLOOR_MALIGNANT_RECALL + 0.01,
            "accuracy": FLOOR_ACCURACY + 0.01,
            "dice": FLOOR_DICE + 0.01,
        }
    )


def test_floor_gate_fails_loud_on_malignant_recall_below_threshold() -> None:
    """A missed-malignant breakdown is the worst error mode — must abort."""
    with pytest.raises(SystemExit) as exc:
        _check_floors(
            {
                "malignant_recall": FLOOR_MALIGNANT_RECALL - 0.1,
                "accuracy": 0.99,
                "dice": 0.99,
            }
        )
    assert "malignant_recall" in str(exc.value)


def test_floor_gate_reports_every_failing_metric_in_one_pass() -> None:
    """Operator shouldn't need to retry deploy three times to find all fails."""
    with pytest.raises(SystemExit) as exc:
        _check_floors({"malignant_recall": 0.10, "accuracy": 0.10, "dice": 0.10})
    msg = str(exc.value)
    assert "malignant_recall" in msg
    assert "accuracy" in msg
    assert "dice" in msg


# --- LATEST.jsonl ----------------------------------------------------------


def test_read_latest_entry_returns_none_when_file_missing(tmp_path: Path) -> None:
    """First deploy: regression gate must skip cleanly when the log doesn't exist."""
    assert _read_latest_entry(tmp_path / "LATEST.jsonl") is None


def test_read_latest_entry_returns_last_row_skipping_blank_lines(
    tmp_path: Path,
) -> None:
    """Append-only file; the active deploy is whatever is on the last non-blank line."""
    log = tmp_path / "LATEST.jsonl"
    log.write_text(
        '{"version_tag": "v1", "metrics": {"tuned_test_composite": 0.55}}\n'
        "\n"
        '{"version_tag": "v2", "metrics": {"tuned_test_composite": 0.62}}\n',
        encoding="utf-8",
    )
    entry = _read_latest_entry(log)
    assert entry is not None
    assert entry["version_tag"] == "v2"


def test_build_latest_entry_captures_full_provenance() -> None:
    """The deploy log must carry every join key for MLflow + Optuna observability."""
    provenance = {
        "params": {"backbone": "custom_unet", "lr": 1e-3},
        "mlflow": {
            "tracking_uri": "sqlite:///tracking/mlflow.db",
            "experiment_name": "claritymed-vision-breast_cancer_ultrasound",
            "train_run_id": "train-abc",
        },
        "optuna": {
            "storage_uri": "sqlite:///tracking/optuna.db",
            "search_study_name": "claritymed-vision-breast_cancer_ultrasound-hparam",
        },
        "tune": {
            "params": {"temperature": 1.2},
            "mlflow": {"tune_run_id": "tune-def"},
            "optuna": {
                "tune_study_name": "claritymed-vision-breast_cancer_ultrasound-tune"
            },
        },
    }
    eval_metrics = {
        "best_val_score": 0.62,
        "test_score": 0.58,
        "tuned_val_breakdown": {"composite": 0.65, "malignant_recall": 0.88},
        "best_epoch": 24,
        "early_stopped": True,
        "epochs_trained": 38,
    }
    tuned_test = {
        "composite": 0.63,
        "malignant_recall": 0.86,
        "dice": 0.72,
        "accuracy": 0.85,
    }
    entry = _build_latest_entry(
        version_tag="20260615T120000Z",
        weights_subpath="vision/breast_cancer_ultrasound/breast_busi_unet_v1__20260615T120000Z",
        manifest_path=Path("/fake/manifest.json"),
        manifest_sha="a" * 64,
        provenance=provenance,
        eval_metrics=eval_metrics,
        tuned_test=tuned_test,
        previous_score=0.55,
    )
    # Every observability join key must be reachable.
    assert entry["mlflow"]["tracking_uri"] == "sqlite:///tracking/mlflow.db"
    assert entry["mlflow"]["train_run_id"] == "train-abc"
    assert entry["mlflow"]["tune_run_id"] == "tune-def"
    assert entry["optuna"]["search_study_name"].endswith("-hparam")
    assert entry["optuna"]["tune_study_name"].endswith("-tune")
    assert entry["best_hp"] == {"backbone": "custom_unet", "lr": 1e-3}
    assert entry["best_inference_params"] == {"temperature": 1.2}
    # Delta vs previous deploy must be computed.
    assert entry["delta"] == pytest.approx(0.63 - 0.55, abs=1e-9)
    # Floors recorded for audit even though the gate has already passed.
    assert entry["floors_passed"]["malignant_recall"] is True


def test_build_latest_entry_delta_is_none_for_first_deploy() -> None:
    """No previous deploy → no delta to report."""
    entry = _build_latest_entry(
        version_tag="v1",
        weights_subpath="x",
        manifest_path=Path("/fake/m.json"),
        manifest_sha="a" * 64,
        provenance={},
        eval_metrics={},
        tuned_test={
            "composite": 0.7,
            "malignant_recall": 0.9,
            "dice": 0.8,
            "accuracy": 0.9,
        },
        previous_score=None,
    )
    assert entry["delta"] is None
    assert entry["previous_tuned_test_composite"] is None


# --- YAML surgical patch -------------------------------------------------


_VISION_YAML_SAMPLE = """\
# Top comment must survive.

diseases:
  - id: breast_cancer_ultrasound
    enabled: false
    primary_model_id: breast_busi_unet_v1
    flow: [breast_busi_unet_v1]
    cancer_class: true
    intent_hints_i18n_key: vision.intent.breast_cancer_ultrasound

servers:
  - id: local_default
    base_url: http://127.0.0.1:8085
    expected_ms: 800
    health_check_interval_s: 300

models:
  - id: breast_busi_unet_v1
    disease_id: breast_cancer_ultrasound
    server_id: local_default
    framework: pytorch
    accepted_modality: ultrasound
    weights_subpath: vision/breast_cancer_ultrasound/breast_busi_unet_v1
    # placeholder sha (boot check overrides)
    manifest_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
    expected_ms: 800

tool:
  total_budget_ms: 20000
  fallback_safety_factor: 1.5
  catalog_refresh_seconds: 300
  confirm_before_run: true
  shadow_inference_on_report_override: false
  top_k: 3

ocr_report:
  min_chars: 200
  markers:
    en: [findings]
    zh: ["所见"]
"""


def test_replace_model_fields_updates_only_targeted_lines() -> None:
    """Surgical patch: two lines change, every other line + comment survives."""
    edited = _replace_model_fields(
        _VISION_YAML_SAMPLE,
        model_id="breast_busi_unet_v1",
        new_weights_subpath="vision/breast_cancer_ultrasound/breast_busi_unet_v1__20260615T120000Z",
        new_manifest_sha="b" * 64,
    )
    assert "# Top comment must survive." in edited
    assert "# placeholder sha (boot check overrides)" in edited
    assert "expected_ms: 800" in edited
    assert (
        "vision/breast_cancer_ultrasound/breast_busi_unet_v1__20260615T120000Z"
        in edited
    )
    assert "b" * 64 in edited
    # Old values gone.
    assert (
        "0000000000000000000000000000000000000000000000000000000000000000" not in edited
    )


def test_replace_model_fields_round_trips_through_yaml_parse() -> None:
    """The edited text must remain valid YAML (no broken indentation)."""
    import yaml as _yaml

    edited = _replace_model_fields(
        _VISION_YAML_SAMPLE,
        model_id="breast_busi_unet_v1",
        new_weights_subpath="vision/x/y__v2",
        new_manifest_sha="c" * 64,
    )
    parsed = _yaml.safe_load(edited)
    model = next(m for m in parsed["models"] if m["id"] == "breast_busi_unet_v1")
    assert model["weights_subpath"] == "vision/x/y__v2"
    assert model["manifest_sha256"] == "c" * 64


def test_replace_model_fields_aborts_on_missing_model_id() -> None:
    """A typoed model_id must fail loud, not silently no-op."""
    with pytest.raises(SystemExit):
        _replace_model_fields(
            _VISION_YAML_SAMPLE,
            model_id="not_a_real_model",
            new_weights_subpath="x",
            new_manifest_sha="d" * 64,
        )
