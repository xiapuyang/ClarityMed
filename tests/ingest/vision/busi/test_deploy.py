"""Deploy phase: floor gate, regression gate, YAML patcher, LATEST.jsonl.

The deploy logic is now in :mod:`claritymed.ingest.vision.forge` —
this file points its assertions at the forge helpers but stays under
``busi/`` because BUSI's ModelSpec (with three deploy floors including
``dice``) is the test fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.ingest.vision.busi.models.unet_resnet50 import UNET_RESNET50
from claritymed.ingest.vision.forge.common import (
    check_floors,
    read_latest_entry,
    replace_model_fields,
)
from claritymed.ingest.vision.forge.framework import (
    _build_latest_entry,
    _check_regression,
    _previous_tuned_test_score,
)
from claritymed.ingest.vision.forge.scoring import (
    FEASIBLE_OFFSET,
    feasibility_aware_score,
)

_FLOORS = UNET_RESNET50.task.deploy_floors_map()
FLOOR_MALIGNANT_RECALL = _FLOORS["malignant_recall"]
FLOOR_ACCURACY = _FLOORS["accuracy"]
FLOOR_DICE = _FLOORS["dice"]


# --- floor gate -----------------------------------------------------------


def test_floor_gate_passes_when_breakdown_clears_every_threshold() -> None:
    check_floors(
        {
            "malignant_recall": FLOOR_MALIGNANT_RECALL + 0.01,
            "accuracy": FLOOR_ACCURACY + 0.01,
            "dice": FLOOR_DICE + 0.01,
        },
        _FLOORS,
    )


def test_floor_gate_fails_loud_on_malignant_recall_below_threshold() -> None:
    with pytest.raises(SystemExit) as exc:
        check_floors(
            {
                "malignant_recall": FLOOR_MALIGNANT_RECALL - 0.1,
                "accuracy": 0.99,
                "dice": 0.99,
            },
            _FLOORS,
        )
    assert "malignant_recall" in str(exc.value)


def test_floor_gate_reports_every_failing_metric_in_one_pass() -> None:
    with pytest.raises(SystemExit) as exc:
        check_floors(
            {"malignant_recall": 0.10, "accuracy": 0.10, "dice": 0.10}, _FLOORS
        )
    msg = str(exc.value)
    assert "malignant_recall" in msg
    assert "accuracy" in msg
    assert "dice" in msg


# --- LATEST.jsonl ----------------------------------------------------------


def test_read_latest_entry_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert read_latest_entry(tmp_path / "LATEST.jsonl") is None


def test_read_latest_entry_returns_last_row_skipping_blank_lines(
    tmp_path: Path,
) -> None:
    log = tmp_path / "LATEST.jsonl"
    log.write_text(
        '{"version_tag": "v1", "metrics": {"tuned_test_score": 0.55}}\n'
        "\n"
        '{"version_tag": "v2", "metrics": {"tuned_test_score": 0.62}}\n',
        encoding="utf-8",
    )
    entry = read_latest_entry(log)
    assert entry is not None
    assert entry["version_tag"] == "v2"


def test_read_latest_entry_filters_by_model_id_when_provided(tmp_path: Path) -> None:
    """Per-(dataset, model_id) regression gate compares apples-to-apples."""
    log = tmp_path / "LATEST.jsonl"
    log.write_text(
        '{"version_tag": "v1", "model_id": "modelA", "metrics": {"tuned_test_score": 0.55}}\n'
        '{"version_tag": "v2", "model_id": "modelB", "metrics": {"tuned_test_score": 0.62}}\n'
        '{"version_tag": "v3", "model_id": "modelA", "metrics": {"tuned_test_score": 0.58}}\n',
        encoding="utf-8",
    )
    assert read_latest_entry(log, model_id="modelA")["version_tag"] == "v3"
    assert read_latest_entry(log, model_id="modelB")["version_tag"] == "v2"


def test_build_latest_entry_captures_full_provenance() -> None:
    provenance = {
        "params": {"backbone": "custom_unet", "lr": 1e-3},
        "task_id": "20260615T120000Z-abcd",
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
        UNET_RESNET50,
        version_tag="20260615T120000Z",
        weights_subpath="vision/breast_cancer_ultrasound/breast_busi_unet_v1__20260615T120000Z",
        manifest_path=Path("/fake/manifest.json"),
        manifest_sha="a" * 64,
        provenance=provenance,
        eval_metrics=eval_metrics,
        tuned_test=tuned_test,
        candidate_score=0.63,
        candidate_composite=tuned_test["composite"],
        previous_score=0.55,
    )
    assert entry["mlflow"]["tracking_uri"] == "sqlite:///tracking/mlflow.db"
    assert entry["mlflow"]["train_run_id"] == "train-abc"
    assert entry["mlflow"]["tune_run_id"] == "tune-def"
    assert entry["optuna"]["search_study_name"].endswith("-hparam")
    assert entry["optuna"]["tune_study_name"].endswith("-tune")
    assert entry["best_hp"] == {"backbone": "custom_unet", "lr": 1e-3}
    assert entry["best_inference_params"] == {"temperature": 1.2}
    assert entry["delta"] == pytest.approx(0.63 - 0.55, abs=1e-9)
    assert entry["floors_passed"]["malignant_recall"] is True
    assert entry["model_id"] == "breast_busi_unet_v1"
    assert entry["disease_id"] == "breast_cancer_ultrasound"
    assert entry["task_id"] == "20260615T120000Z-abcd"


def test_build_latest_entry_delta_is_none_for_first_deploy() -> None:
    entry = _build_latest_entry(
        UNET_RESNET50,
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
        candidate_score=0.7,
        candidate_composite=0.7,
        previous_score=None,
    )
    assert entry["delta"] is None
    assert entry["previous_tuned_test_score"] is None


# --- regression gate legacy-row fallback --------------------------------


def test_previous_tuned_test_score_uses_persisted_score_when_present() -> None:
    """New entries persist ``tuned_test_score`` directly — read it verbatim."""
    previous = {"metrics": {"tuned_test_score": 1.42, "test_score": 0.99}}
    assert _previous_tuned_test_score(previous, UNET_RESNET50) == 1.42


def test_previous_tuned_test_score_reconstructs_legacy_entry_from_breakdown() -> None:
    """Legacy rows have ``tuned_test_breakdown`` but no ``tuned_test_score``.

    Old scoring persisted the raw composite under ``test_score`` (range
    ``[0, 1]``); current scoring is feasibility-aware (``(1, 2]`` if
    feasible, ``≤ 0`` if not). Comparing the two scales directly would
    silently mis-rank, so the fallback recomputes from the preserved
    breakdown using today's tune-phase floors + composite weights — the
    exact formula the candidate is scored with.
    """
    # A feasible legacy breakdown (clears every deploy floor) should
    # reconstruct into the feasible region: FEASIBLE_OFFSET + composite.
    feasible_breakdown = {
        "malignant_recall": 0.90,
        "accuracy": 0.88,
        "dice": 0.75,
    }
    legacy = {
        "version_tag": "old-feasible-v1",
        "metrics": {
            "test_score": 0.6 * 0.90 + 0.4 * 0.75,  # historical raw composite
            "tuned_test_breakdown": feasible_breakdown,
        },
    }
    reconstructed = _previous_tuned_test_score(legacy, UNET_RESNET50)
    expected = feasibility_aware_score(
        feasible_breakdown,
        UNET_RESNET50.task.phase_floors("tune"),
        UNET_RESNET50.task.composite_weights,
    )
    assert reconstructed == pytest.approx(expected)
    assert reconstructed > FEASIBLE_OFFSET  # feasible region


def test_previous_tuned_test_score_fails_loud_when_breakdown_missing() -> None:
    """No score and no breakdown → fail loud, not silent skip.

    Falling back to "skip regression gate" would let any candidate
    deploy unchecked against a known-old-but-quality baseline. The
    operator should know they need to re-deploy or clean ``LATEST.jsonl``.
    """
    pre_breakdown_entry = {
        "version_tag": "ancient",
        "metrics": {"some_unrelated_key": 0.5},
    }
    with pytest.raises(SystemExit, match="pre-dates"):
        _previous_tuned_test_score(pre_breakdown_entry, UNET_RESNET50)


def test_check_regression_blocks_infeasible_candidate_vs_feasible_baseline() -> None:
    """The whole reason this gate exists — drove the recent fix.

    A candidate that missed a clinical floor (score in ``(-inf, 0]``)
    must not overwrite a previously-feasible baseline (score in
    ``(1, 2]``). Floor-cliff comparison falls out of the score scale.
    """
    with pytest.raises(SystemExit, match="regression gate failed"):
        _check_regression(
            previous_version="old-feasible-v1",
            prev_score=1.75,
            candidate_score=-0.28,
        )


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
    edited = replace_model_fields(
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
    assert "0" * 64 not in edited


def test_replace_model_fields_round_trips_through_yaml_parse() -> None:
    import yaml as _yaml

    edited = replace_model_fields(
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
    with pytest.raises(SystemExit):
        replace_model_fields(
            _VISION_YAML_SAMPLE,
            model_id="not_a_real_model",
            new_weights_subpath="x",
            new_manifest_sha="d" * 64,
        )
