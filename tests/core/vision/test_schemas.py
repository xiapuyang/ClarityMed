"""Validate ``configs/vision.yaml`` parses + every fail-loud path fires."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from claritymed.config import CONFIGS_DIR
from claritymed.core.vision.schemas import (
    ClassificationResult,
    ClinicalAction,
    DiseaseSpec,
    LabelMeta,
    LLMDetectionPayload,
    Manifest,
    ModalityMismatchResult,
    ModelSpec,
    OcrOverrideResult,
    RawDetection,
    ServerSpec,
    VisionConfig,
)

_HEX64 = "a" * 64


# --- helpers ---------------------------------------------------------------


def _minimal_payload(**overrides) -> dict:
    """Return a minimal VisionConfig-valid YAML-shaped dict."""
    payload: dict = {
        "diseases": [
            {
                "id": "breast_cancer_ultrasound",
                "enabled": True,
                "primary_model_id": "breast_busi_unet_v1",
                "flow": [],
                "cancer_class": True,
                "intent_hints_i18n_key": "vision.intent.breast_cancer_ultrasound",
            }
        ],
        "servers": [
            {
                "id": "local_default",
                "base_url": "http://127.0.0.1:8085",
                "expected_ms": 800,
            }
        ],
        "models": [
            {
                "id": "breast_busi_unet_v1",
                "disease_id": "breast_cancer_ultrasound",
                "server_id": "local_default",
                "framework": "pytorch",
                "accepted_modality": "ultrasound",
                "weights_subpath": "vision/breast_cancer_ultrasound/breast_busi_unet_v1",
                "manifest_sha256": _HEX64,
                "expected_ms": 800,
            }
        ],
        "ocr_report": {
            "min_chars": 200,
            "markers": {
                "en": ["findings", "impression"],
                "zh": ["所见", "印象"],
            },
        },
    }
    payload.update(overrides)
    return payload


def _minimal_manifest(**overrides) -> dict:
    """Return a minimal Manifest-valid dict for the BUSI shape."""
    payload: dict = {
        "model_id": "breast_busi_unet_v1",
        "model_version": "v1.0.0",
        "framework": "pytorch",
        "accepted_modality": "ultrasound",
        "sha256_weights": _HEX64,
        "task": "classification+segmentation",
        "labels": ["benign", "malignant", "normal"],
        "labels_meta": {
            "benign": {
                "description": "non-cancerous",
                "cancer_status": "benign",
                "clinical_action": "routine_followup",
            },
            "malignant": {
                "description": "suspicious for cancer",
                "cancer_status": "malignant",
                "clinical_action": "urgent_specialist",
            },
            "normal": {
                "description": "no lesion identified",
                "cancer_status": "normal",
                "clinical_action": "no_action",
            },
        },
        "cancer_class": True,
        "cancer_status_mapping": {
            "benign": "benign",
            "malignant": "malignant",
            "normal": "normal",
        },
        "clinical_action_mapping": {
            "benign": "routine_followup",
            "malignant": "urgent_specialist",
            "normal": "no_action",
        },
        "supports_saliency": False,
        "supports_tta": True,
    }
    payload.update(overrides)
    return payload


# --- happy paths -----------------------------------------------------------


def test_shipped_configs_vision_yaml_loads() -> None:
    """The repo's configs/vision.yaml is parseable + valid.

    Catches drift between the YAML and the schema during code review.
    """
    raw = yaml.safe_load(Path(CONFIGS_DIR / "vision.yaml").read_text())
    cfg = VisionConfig.model_validate(raw)
    assert cfg.diseases[0].id == "breast_cancer_ultrasound"
    assert cfg.servers[0].base_url == "http://127.0.0.1:8085"
    assert cfg.models[0].accepted_modality == "ultrasound"
    assert cfg.tool.shadow_inference_on_report_override is False


def test_minimal_vision_config_loads() -> None:
    cfg = VisionConfig.model_validate(_minimal_payload())
    assert cfg.diseases[0].cancer_class is True
    assert cfg.tool.total_budget_ms == 20_000  # default applied


def test_clinical_action_literal_accepts_all_five_values() -> None:
    """KTD-V1: the sibling axis carries exactly five values."""
    values: list[ClinicalAction] = [
        "urgent_specialist",
        "soon_specialist",
        "routine_followup",
        "no_action",
        "inconclusive_review",
    ]
    for v in values:
        # round-trip through a model field
        LabelMeta(description="x", cancer_status="benign", clinical_action=v)


# --- VisionConfig cross-reference validators -------------------------------


def test_primary_model_id_not_in_models_fails() -> None:
    payload = _minimal_payload()
    payload["diseases"][0]["primary_model_id"] = "missing_model"
    payload["diseases"][0]["flow"] = []
    with pytest.raises(ValidationError) as exc:
        VisionConfig.model_validate(payload)
    err = str(exc.value)
    assert "missing_model" in err and "unknown" in err


def test_model_server_id_unknown_fails() -> None:
    payload = _minimal_payload()
    payload["models"][0]["server_id"] = "nonexistent_server"
    with pytest.raises(ValidationError) as exc:
        VisionConfig.model_validate(payload)
    assert "server_id" in str(exc.value)


def test_model_disease_id_unknown_fails() -> None:
    payload = _minimal_payload()
    payload["models"][0]["disease_id"] = "nonexistent_disease"
    with pytest.raises(ValidationError) as exc:
        VisionConfig.model_validate(payload)
    assert "disease_id" in str(exc.value)


def test_duplicate_disease_ids_fail() -> None:
    payload = _minimal_payload()
    payload["diseases"].append(payload["diseases"][0])
    with pytest.raises(ValidationError) as exc:
        VisionConfig.model_validate(payload)
    assert "unique" in str(exc.value)


def test_duplicate_model_ids_fail() -> None:
    payload = _minimal_payload()
    payload["models"].append(payload["models"][0])
    with pytest.raises(ValidationError) as exc:
        VisionConfig.model_validate(payload)
    assert "unique" in str(exc.value)


def test_duplicate_server_ids_fail() -> None:
    payload = _minimal_payload()
    payload["servers"].append(payload["servers"][0])
    with pytest.raises(ValidationError) as exc:
        VisionConfig.model_validate(payload)
    assert "unique" in str(exc.value)


def test_disease_flow_must_not_contain_primary_model_id() -> None:
    """``primary_model_id`` is auto-prepended via :attr:`effective_flow`.

    Listing it again in ``flow`` is a config bug — flagged loudly so
    operators don't accidentally double-run the primary or split its
    audit trail across two attempts.
    """
    payload = _minimal_payload()
    payload["diseases"][0]["primary_model_id"] = "breast_busi_unet_v1"
    payload["diseases"][0]["flow"] = ["breast_busi_unet_v1"]
    with pytest.raises(ValidationError) as exc:
        VisionConfig.model_validate(payload)
    assert "must NOT appear in" in str(exc.value)


def test_effective_flow_prepends_primary() -> None:
    """``effective_flow`` is the runtime order: primary first, then fallbacks."""
    payload = _minimal_payload()
    payload["models"].append(
        {
            **payload["models"][0],
            "id": "fallback_model",
        }
    )
    payload["diseases"][0]["flow"] = ["fallback_model"]
    cfg = VisionConfig.model_validate(payload)
    disease = cfg.diseases[0]
    assert disease.effective_flow == ["breast_busi_unet_v1", "fallback_model"]


def test_effective_flow_when_no_fallbacks() -> None:
    """Empty ``flow`` is the common case; ``effective_flow`` is just primary."""
    cfg = VisionConfig.model_validate(_minimal_payload())
    assert cfg.diseases[0].effective_flow == ["breast_busi_unet_v1"]


def test_disease_flow_must_be_unique() -> None:
    payload = _minimal_payload()
    payload["models"].append(
        {
            **payload["models"][0],
            "id": "fallback_a",
        }
    )
    payload["diseases"][0]["flow"] = ["fallback_a", "fallback_a"]
    with pytest.raises(ValidationError) as exc:
        VisionConfig.model_validate(payload)
    assert "unique" in str(exc.value)


def test_cancer_class_disease_mixed_modality_flow_fails() -> None:
    """KTD-V3: cancer-class fallback chain must stay on one modality.

    A breast-ultrasound primary + a CT secondary would mean the hard-gate
    check upstream of inference is meaningless for the second hop.
    """
    payload = _minimal_payload()
    payload["models"].append(
        {
            "id": "ct_fallback",
            "disease_id": "breast_cancer_ultrasound",
            "server_id": "local_default",
            "framework": "pytorch",
            "accepted_modality": "ct",
            "weights_subpath": "vision/foo/bar",
            "manifest_sha256": _HEX64,
            "expected_ms": 600,
        }
    )
    # Primary stays BUSI (ultrasound); the CT fallback is what trips the gate.
    payload["diseases"][0]["flow"] = ["ct_fallback"]
    with pytest.raises(ValidationError) as exc:
        VisionConfig.model_validate(payload)
    assert "accepted_modality" in str(exc.value)


# --- ModelSpec validators --------------------------------------------------


def test_model_spec_accepted_modality_cannot_be_unknown() -> None:
    """`unknown` is a runtime fallback; the hard gate has no semantics against it."""
    with pytest.raises(ValidationError) as exc:
        ModelSpec.model_validate(
            {
                "id": "x",
                "disease_id": "y",
                "server_id": "z",
                "framework": "pytorch",
                "accepted_modality": "unknown",
                "weights_subpath": "foo/bar",
                "manifest_sha256": _HEX64,
                "expected_ms": 500,
            }
        )
    assert "unknown" in str(exc.value)


def test_model_spec_rejects_absolute_weights_path() -> None:
    with pytest.raises(ValidationError) as exc:
        ModelSpec.model_validate(
            {
                "id": "x",
                "disease_id": "y",
                "server_id": "z",
                "framework": "pytorch",
                "accepted_modality": "ultrasound",
                "weights_subpath": "/abs/path",
                "manifest_sha256": _HEX64,
                "expected_ms": 500,
            }
        )
    assert "absolute" in str(exc.value)


def test_model_spec_rejects_dotdot_segments() -> None:
    with pytest.raises(ValidationError) as exc:
        ModelSpec.model_validate(
            {
                "id": "x",
                "disease_id": "y",
                "server_id": "z",
                "framework": "pytorch",
                "accepted_modality": "ultrasound",
                "weights_subpath": "foo/../etc",
                "manifest_sha256": _HEX64,
                "expected_ms": 500,
            }
        )
    assert ".." in str(exc.value)


def test_model_spec_rejects_bad_sha256_length() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            {
                "id": "x",
                "disease_id": "y",
                "server_id": "z",
                "framework": "pytorch",
                "accepted_modality": "ultrasound",
                "weights_subpath": "foo",
                "manifest_sha256": "abc",  # too short
                "expected_ms": 500,
            }
        )


# --- ServerSpec ------------------------------------------------------------


def test_server_spec_round_trip() -> None:
    s = ServerSpec.model_validate(
        {"id": "local", "base_url": "http://127.0.0.1:8085", "expected_ms": 800}
    )
    assert s.health_check_interval_s == 300  # default


# --- DiseaseSpec -----------------------------------------------------------


def test_disease_spec_round_trip() -> None:
    d = DiseaseSpec.model_validate(
        {
            "id": "breast_cancer_ultrasound",
            "primary_model_id": "m1",
            # ``flow`` is fallbacks-only and defaults to []; the primary
            # is auto-prepended via effective_flow.
            "intent_hints_i18n_key": "vision.intent.breast_cancer_ultrasound",
        }
    )
    assert d.enabled is True
    assert d.cancer_class is False  # default
    assert d.flow == []
    assert d.effective_flow == ["m1"]


# --- OcrReportConfig -------------------------------------------------------


def test_ocr_report_markers_must_be_non_empty() -> None:
    payload = _minimal_payload()
    payload["ocr_report"]["markers"]["en"] = []
    with pytest.raises(ValidationError) as exc:
        VisionConfig.model_validate(payload)
    assert "markers" in str(exc.value)


# --- Manifest --------------------------------------------------------------


def test_manifest_happy_path() -> None:
    m = Manifest.model_validate(_minimal_manifest())
    assert m.cancer_class is True
    assert m.labels_meta["malignant"].clinical_action == "urgent_specialist"


def test_manifest_cancer_class_requires_cancer_status_mapping() -> None:
    """Edge case from the plan: cancer_class=True without mapping fails."""
    payload = _minimal_manifest(cancer_status_mapping=None)
    with pytest.raises(ValidationError) as exc:
        Manifest.model_validate(payload)
    assert "cancer_status_mapping" in str(exc.value)


def test_manifest_cancer_class_requires_clinical_action_mapping() -> None:
    payload = _minimal_manifest(clinical_action_mapping=None)
    with pytest.raises(ValidationError) as exc:
        Manifest.model_validate(payload)
    assert "clinical_action_mapping" in str(exc.value)


def test_manifest_rejects_unknown_clinical_action() -> None:
    """Edge case from the plan: invalid action literal fails fast."""
    payload = _minimal_manifest()
    payload["clinical_action_mapping"]["malignant"] = "unknown_action"
    with pytest.raises(ValidationError):
        Manifest.model_validate(payload)


def test_manifest_labels_meta_must_cover_all_labels() -> None:
    payload = _minimal_manifest()
    del payload["labels_meta"]["normal"]
    with pytest.raises(ValidationError) as exc:
        Manifest.model_validate(payload)
    assert "normal" in str(exc.value)


def test_manifest_cancer_status_mapping_must_cover_all_labels() -> None:
    payload = _minimal_manifest()
    del payload["cancer_status_mapping"]["normal"]
    with pytest.raises(ValidationError) as exc:
        Manifest.model_validate(payload)
    assert "cancer_status_mapping" in str(exc.value)


def test_manifest_non_cancer_class_skips_mapping_requirement() -> None:
    """A future non-oncology disease shouldn't be forced to declare the mappings."""
    payload = _minimal_manifest(
        cancer_class=False,
        cancer_status_mapping=None,
        clinical_action_mapping=None,
    )
    Manifest.model_validate(payload)  # does not raise


# --- tuned inference params (new tuning pipeline output) --------------------


def test_confidence_thresholds_requires_low_below_medium() -> None:
    """A degenerate confidence band (low_max ≥ medium_max) is rejected."""
    from claritymed.core.vision.schemas import ConfidenceThresholds

    with pytest.raises(ValidationError) as exc:
        ConfidenceThresholds.model_validate({"low_max": 0.8, "medium_max": 0.5})
    assert "less than medium_max" in str(exc.value)


def test_tuned_inference_classification_threshold_out_of_unit_rejected() -> None:
    """A threshold like 1.5 has no meaning — reject before it lands in a manifest."""
    from claritymed.core.vision.schemas import TunedInferenceParams

    with pytest.raises(ValidationError) as exc:
        TunedInferenceParams.model_validate(
            {"classification_thresholds": {"malignant": 1.5}}
        )
    assert "must be in [0, 1]" in str(exc.value)


def test_manifest_tuned_inference_unknown_label_rejected() -> None:
    """A tune output pointing at a missing label means train/tune drifted."""
    payload = _minimal_manifest(
        tuned_inference={
            "temperature": 1.0,
            "classification_thresholds": {"NOT_A_LABEL": 0.5},
        }
    )
    with pytest.raises(ValidationError) as exc:
        Manifest.model_validate(payload)
    assert "NOT_A_LABEL" in str(exc.value)


def test_manifest_tuned_inference_roundtrip() -> None:
    """Happy path: a fully-populated tuned block round-trips through the schema."""
    payload = _minimal_manifest(
        tuned_inference={
            "temperature": 1.2,
            "classification_thresholds": {"malignant": 0.45},
            "seg_threshold": 0.5,
            "confidence_thresholds": {"low_max": 0.55, "medium_max": 0.8},
            "tta_default": True,
        }
    )
    manifest = Manifest.model_validate(payload)
    assert manifest.tuned_inference is not None
    assert manifest.tuned_inference.temperature == 1.2
    assert manifest.tuned_inference.classification_thresholds == {"malignant": 0.45}
    assert manifest.tuned_inference.confidence_thresholds.low_max == 0.55
    assert manifest.tuned_inference.tta_default is True


def test_manifest_tuned_inference_optional_for_backwards_compat() -> None:
    """Manifests written before the tune phase shipped must still load."""
    payload = _minimal_manifest()
    manifest = Manifest.model_validate(payload)
    assert manifest.tuned_inference is None


# --- result models ---------------------------------------------------------


def test_classification_result_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValidationError):
        ClassificationResult.model_validate(
            {
                "labels": ["a", "b"],
                "probabilities": [0.5, 0.3, 0.2],  # 3 probs, 2 labels
                "top1": "a",
                "top1_prob": 0.5,
                "confidence_tier": "high",
            }
        )


def test_classification_result_top1_must_be_in_labels() -> None:
    with pytest.raises(ValidationError):
        ClassificationResult.model_validate(
            {
                "labels": ["a", "b"],
                "probabilities": [0.5, 0.5],
                "top1": "c",
                "top1_prob": 0.5,
                "confidence_tier": "high",
            }
        )


def test_raw_detection_round_trip() -> None:
    raw = RawDetection.model_validate(
        {
            "request_id": "req_1",
            "disease_id": "breast_cancer_ultrasound",
            "model_id": "breast_busi_unet_v1",
            "model_version": "v1.0.0",
            "elapsed_ms": 412,
            "input_quality": {"passed": True, "checks": []},
            "classification": {
                "labels": ["benign", "malignant", "normal"],
                "probabilities": [0.1, 0.85, 0.05],
                "top1": "malignant",
                "top1_prob": 0.85,
                "confidence_tier": "high",
            },
            "cancer_status": "malignant",
            "clinical_action": "urgent_specialist",
            "labels_meta": {
                "benign": {
                    "description": "x",
                    "cancer_status": "benign",
                    "clinical_action": "routine_followup",
                },
                "malignant": {
                    "description": "y",
                    "cancer_status": "malignant",
                    "clinical_action": "urgent_specialist",
                },
                "normal": {
                    "description": "z",
                    "cancer_status": "normal",
                    "clinical_action": "no_action",
                },
            },
            "warnings": [],
        }
    )
    assert raw.classification.top1 == "malignant"
    assert raw.clinical_action == "urgent_specialist"


def test_llm_detection_payload_carries_kind_discriminator() -> None:
    payload = LLMDetectionPayload.model_validate(
        {
            "request_id": "req_1",
            "disease_id": "breast_cancer_ultrasound",
            "model_id": "breast_busi_unet_v1",
            "model_version": "v1.0.0",
            "elapsed_ms": 412,
            "top_labels": ["malignant", "benign", "normal"],
            "top_probabilities": [0.85, 0.1, 0.05],
            "top1": "malignant",
            "top1_prob": 0.85,
            "confidence_tier": "high",
            "cancer_status": "malignant",
            "clinical_action": "urgent_specialist",
            "labels_meta": {
                "malignant": {
                    "description": "y",
                    "cancer_status": "malignant",
                    "clinical_action": "urgent_specialist",
                },
            },
        }
    )
    assert payload.kind == "detection"


def test_short_circuit_results_carry_discriminator() -> None:
    """Reply-prompt branches on `kind` — each short-circuit dict must set it."""
    mm = ModalityMismatchResult(
        model_accepts="ultrasound", image_modality="ct", message="mismatch"
    )
    assert mm.kind == "modality_mismatch"

    ocr = OcrOverrideResult(message="OCR carries a clinician report")
    assert ocr.kind == "ocr_override"


def test_extra_field_is_forbidden() -> None:
    """frozen + extra=forbid is the safety net against hallucinated fields."""
    with pytest.raises(ValidationError):
        DiseaseSpec.model_validate(
            {
                "id": "x",
                "primary_model_id": "m1",
                "flow": [],
                "intent_hints_i18n_key": "vision.intent.x",
                "hallucinated_field": True,
            }
        )
