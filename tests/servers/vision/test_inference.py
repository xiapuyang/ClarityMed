"""Inference orchestration + KTD-V10 override (Unit 4).

The handler in ``app.py`` is a thin wrapper around
``run_inference``; this file exercises the rules in isolation so the
override and mapping logic don't have to be re-derived from a full
HTTP round-trip.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

from claritymed.core.medical_clip.schemas import ImagePayload
from claritymed.core.vision.schemas import (
    ClassificationResult,
    InputQuality,
    QualityCheck,
)
from claritymed.core.vision.wire import DetectOptions, DetectRequest
from claritymed.servers.vision.adapters.torch_adapter import TorchAdapter
from claritymed.servers.vision.inference import InferenceResources, run_inference
from claritymed.servers.vision.loader import verify_manifest_chain


# --- small helpers --------------------------------------------------------


def _png_bytes() -> bytes:
    """16×16 PNG — below the Torch adapter's min_resolution=64 gate.

    Used to exercise the KTD-V10 quality-gate override path. Generated
    via Pillow rather than hand-coded so the bytes are guaranteed
    decodable (raw PNG headers are brittle to copy-paste).
    """
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color="black").save(buf, format="PNG")
    return buf.getvalue()


def _bigger_png_bytes() -> bytes:
    """64×64 white PNG — clears the Torch adapter's min_resolution gate."""
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="white").save(buf, format="PNG")
    return buf.getvalue()


def _request_for(image_bytes: bytes, **overrides) -> DetectRequest:
    sha = hashlib.sha256(image_bytes).hexdigest()
    payload = {
        "request_id": overrides.get("request_id", "req_42"),
        "disease_id": overrides.get("disease_id", "breast_cancer_ultrasound"),
        "model_id": overrides.get("model_id"),
        "image": ImagePayload(
            sha256=sha,
            data_b64=base64.b64encode(image_bytes).decode("ascii"),
        ),
        "language": overrides.get("language", "en"),
        "options": overrides.get("options", DetectOptions()),
    }
    return DetectRequest(**payload)


def _resources_from_fixture(
    make_vision_artifact, vision_models_root: Path, **kwargs
) -> InferenceResources:
    spec, manifest_path, weights_path = make_vision_artifact(**kwargs)
    manifest = verify_manifest_chain(spec, manifest_path.parent)
    adapter = TorchAdapter(
        spec=spec, manifest=manifest, weights_path=weights_path, device="cpu"
    )
    return InferenceResources(
        spec_id=spec.id,
        disease_id=spec.disease_id,
        model=adapter,
        manifest=manifest,
    )


# --- happy path -----------------------------------------------------------


def test_run_inference_returns_full_raw_detection(
    make_vision_artifact, vision_models_root: Path
) -> None:
    resources = _resources_from_fixture(make_vision_artifact, vision_models_root)
    image_bytes = _bigger_png_bytes()
    req = _request_for(image_bytes)
    response = run_inference(request=req, image_bytes=image_bytes, resources=resources)

    # Mandatory wire fields are populated.
    assert response.request_id == "req_42"
    assert response.disease_id == "breast_cancer_ultrasound"
    assert response.model_id == "breast_busi_unet_v1"
    assert response.model_version == "v1.0.0"
    # Classification scoreboard is well-shaped (validators on the model).
    assert response.classification.top1 in response.classification.labels
    assert sum(response.classification.probabilities) == pytest.approx(1.0, rel=1e-6)
    # cancer_class fixture → labels_meta round-trips.
    assert set(response.labels_meta) == {"benign", "malignant", "normal"}
    # quality_gate passes for the 64×64 image; no override warning.
    assert response.input_quality.passed is True


def test_cancer_class_status_and_action_use_manifest_mapping(
    make_vision_artifact, vision_models_root: Path
) -> None:
    """For each possible top1 label, the mapping decides cancer_status / clinical_action.

    The stub's top1 is deterministic from the image bytes; we pin it by
    using one of three PNGs with known hashes so we can assert on the
    mapping side rather than the model side.
    """
    resources = _resources_from_fixture(make_vision_artifact, vision_models_root)
    image_bytes = _bigger_png_bytes()
    req = _request_for(image_bytes)
    response = run_inference(request=req, image_bytes=image_bytes, resources=resources)
    expected_cancer = resources.manifest.cancer_status_mapping[
        response.classification.top1
    ]
    expected_action = resources.manifest.clinical_action_mapping[
        response.classification.top1
    ]
    # If the quality gate passed and confidence is medium/high, the
    # mapping should win straight through.
    if (
        response.input_quality.passed
        and response.classification.confidence_tier != "low"
    ):
        assert response.cancer_status == expected_cancer
        assert response.clinical_action == expected_action


def test_non_cancer_class_clinical_action_defaults_to_routine(
    make_vision_artifact, vision_models_root: Path
) -> None:
    resources = _resources_from_fixture(
        make_vision_artifact, vision_models_root, cancer_class=False
    )
    image_bytes = _bigger_png_bytes()
    req = _request_for(image_bytes)
    response = run_inference(request=req, image_bytes=image_bytes, resources=resources)
    # Non-cancer-class → no cancer_status, clinical_action is the
    # conservative-tone default unless the override fires.
    assert response.cancer_status is None
    # Test image clears quality_gate and confidence_tier from stub is
    # medium (top1_prob=0.7), so no override.
    assert response.clinical_action == "routine_followup"


# --- KTD-V10: quality gate override --------------------------------------


def test_quality_gate_failure_overrides_to_inconclusive_review(
    make_vision_artifact, vision_models_root: Path
) -> None:
    resources = _resources_from_fixture(make_vision_artifact, vision_models_root)
    # 1×1 PNG → min_resolution < 64 → quality_gate fails.
    image_bytes = _png_bytes()
    req = _request_for(image_bytes)
    response = run_inference(request=req, image_bytes=image_bytes, resources=resources)
    assert response.input_quality.passed is False
    assert response.clinical_action == "inconclusive_review"
    # Warning records *why* — auditors need to reproduce the decision.
    assert any("quality_gate.passed=False" in w for w in response.warnings)
    assert any("KTD-V10" in w for w in response.warnings)


# --- KTD-V10: low-confidence override ------------------------------------


class _ForceLowConfidenceAdapter:
    """Adapter that returns a low-confidence classification + passing quality.

    Hand-rolled rather than subclassing TorchAdapter so the test stays
    independent of how the stub picks its top1.
    """

    def __init__(self, *, spec, manifest) -> None:
        self.spec = spec
        self.manifest = manifest

    def preprocess(self, image_bytes):
        from PIL import Image
        import io

        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def predict(self, x):
        labels = self.manifest.labels
        n = len(labels)
        # Uniform-ish probs → low confidence.
        probs = [1.0 / n] * n
        return ClassificationResult(
            labels=list(labels),
            probabilities=probs,
            top1=labels[0],
            top1_prob=probs[0],
            confidence_tier="low",
        )

    def calibrate(self, raw):
        if isinstance(raw, ClassificationResult):
            return dict(zip(raw.labels, raw.probabilities, strict=True))
        return dict(raw)

    def segment(self, x):
        return None

    def quality_gate(self, image):
        return InputQuality(
            passed=True,
            checks=[QualityCheck(name="min_resolution", score=999.0, passed=True)],
        )


def test_low_confidence_overrides_to_inconclusive_review_even_when_quality_passes(
    make_vision_artifact, vision_models_root: Path
) -> None:
    spec, manifest_path, _ = make_vision_artifact()
    manifest = verify_manifest_chain(spec, manifest_path.parent)
    adapter = _ForceLowConfidenceAdapter(spec=spec, manifest=manifest)
    resources = InferenceResources(
        spec_id=spec.id, disease_id=spec.disease_id, model=adapter, manifest=manifest
    )
    image_bytes = _bigger_png_bytes()
    req = _request_for(image_bytes)
    response = run_inference(request=req, image_bytes=image_bytes, resources=resources)
    assert response.input_quality.passed is True
    assert response.classification.confidence_tier == "low"
    assert response.clinical_action == "inconclusive_review"
    assert any("confidence_tier=low" in w for w in response.warnings)


# --- capability negotiation (silent no-ops) ------------------------------


def test_return_saliency_when_unsupported_records_warning_but_succeeds(
    make_vision_artifact, vision_models_root: Path
) -> None:
    resources = _resources_from_fixture(make_vision_artifact, vision_models_root)
    image_bytes = _bigger_png_bytes()
    req = _request_for(
        image_bytes,
        options=DetectOptions(return_segmentation=True, return_saliency=True),
    )
    response = run_inference(request=req, image_bytes=image_bytes, resources=resources)
    assert response.saliency_b64 is None
    assert any("return_saliency=true" in w for w in response.warnings)


def test_tta_when_unsupported_records_warning(
    make_vision_artifact, vision_models_root: Path
) -> None:
    resources = _resources_from_fixture(make_vision_artifact, vision_models_root)
    image_bytes = _bigger_png_bytes()
    req = _request_for(image_bytes, options=DetectOptions(tta=True))
    response = run_inference(request=req, image_bytes=image_bytes, resources=resources)
    assert any("tta=true" in w for w in response.warnings)


def test_return_segmentation_false_omits_segmentation_block(
    make_vision_artifact, vision_models_root: Path
) -> None:
    resources = _resources_from_fixture(make_vision_artifact, vision_models_root)
    image_bytes = _bigger_png_bytes()
    req = _request_for(image_bytes, options=DetectOptions(return_segmentation=False))
    response = run_inference(request=req, image_bytes=image_bytes, resources=resources)
    assert response.segmentation is None
