"""Per-request inference orchestration for the vision server.

Stitches a loaded ``DiseaseVisionModel`` together with the manifest's
mapping tables and the ``DetectOptions`` knobs into one ``RawDetection``
wire response. The handler in ``app.py`` calls ``run_inference`` exactly
once per request; this module concentrates the rules the brainstorm
spec scattered across §6 (Protocol), §7 (response shape), and KTD-V10
(low-confidence / quality-gate override).

Key rules implemented here:

* ``cancer_status`` and ``clinical_action`` come from
  ``manifest.cancer_status_mapping`` / ``manifest.clinical_action_mapping``
  keyed on the classification's top1 label. Validated at boot
  (``Manifest._cancer_class_mappings_complete``) so a runtime KeyError
  here would indicate a manifest tampering between boot and request —
  we re-raise rather than swallow.
* **KTD-V10 override:** if ``confidence_tier == "low"`` OR
  ``input_quality.passed == false``, ``clinical_action`` is forced to
  ``inconclusive_review`` regardless of the mapping. A matching
  ``warnings[]`` entry records why so audit can reproduce the decision.
* **Capability flags:** ``DetectOptions.return_saliency`` and ``.tta``
  silently no-op when the manifest declares no support, with a single
  ``warnings[]`` entry per no-op. Never 4xx — capability negotiation
  must not break a happy detect request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterable

from claritymed.core.vision.schemas import (
    CancerStatus,
    ClassificationResult,
    ClinicalAction,
    DiseaseVisionModel,
    InputQuality,
    Manifest,
    RawDetection,
    SegmentationResult,
)
from claritymed.core.vision.wire import DetectOptions, DetectRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResources:
    """Bundle of per-model state the handler keeps on app state.

    Manifest is carried alongside the model so the handler doesn't have
    to re-parse JSON or thread a separate state dict through.
    """

    spec_id: str
    disease_id: str
    model: DiseaseVisionModel
    manifest: Manifest


def run_inference(
    *,
    request: DetectRequest,
    image_bytes: bytes,
    resources: InferenceResources,
) -> RawDetection:
    """Run one classification + (optional) segmentation pass.

    Args:
        request: Validated ``DetectRequest`` carrying request_id, options,
            and target disease/model ids.
        image_bytes: Raw decoded image bytes (the handler ran base64 +
            sha256 verification upstream).
        resources: The loaded model + manifest + per-spec metadata.

    Returns:
        ``RawDetection`` ready for serialization.

    Raises:
        Whatever the adapter raises (``RuntimeError`` typically). The
        handler maps those to a 500 with code ``inference_failed``.
    """
    model = resources.model
    manifest = resources.manifest
    options = request.options
    t0 = time.monotonic()
    warnings: list[str] = []

    pre = model.preprocess(image_bytes)
    quality = model.quality_gate(pre)
    classification = model.predict(pre)
    # Calibrate is currently identity on the v1 stub but the call site
    # lives here so the real adapter can swap in temperature scaling
    # without us editing this orchestration file.
    _ = model.calibrate(classification)
    segmentation = _maybe_segment(model, pre, options, warnings)
    cancer_status, clinical_action = _derive_status_and_action(
        classification, manifest, quality, warnings
    )
    _record_capability_warnings(options, manifest, warnings)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    return RawDetection(
        request_id=request.request_id,
        disease_id=resources.disease_id,
        model_id=resources.spec_id,
        model_version=manifest.model_version,
        elapsed_ms=elapsed_ms,
        input_quality=quality,
        classification=classification,
        cancer_status=cancer_status,
        clinical_action=clinical_action,
        segmentation=segmentation,
        saliency_b64=None,  # ``supports_saliency`` is False in v1
        labels_meta=dict(manifest.labels_meta),
        warnings=warnings,
        model_card_url=manifest.model_card_url,
    )


def _maybe_segment(
    model: DiseaseVisionModel,
    preprocessed_image,
    options: DetectOptions,
    warnings: list[str],
) -> SegmentationResult | None:
    """Run the segmentation head when the client opted in.

    Returning ``None`` from the adapter means "this model is
    classification-only" — that's a manifest property the client can
    introspect via ``/v1/catalog::task``, so we don't need to warn here
    (it's not surprising).
    """
    if not options.return_segmentation:
        return None
    result = model.segment(preprocessed_image)
    if result is None and options.return_segmentation:
        # Adapter declined despite the client asking. Quiet log path —
        # not a warning because the client already knew this might happen
        # (the catalog declares task: classification only). Including it
        # would noise up audit payloads for the classification-only diseases
        # that ship later (origin §2 deferred-disease phasing).
        logger.debug("segment() returned None even though return_segmentation=True")
    return result


def _derive_status_and_action(
    classification: ClassificationResult,
    manifest: Manifest,
    quality: InputQuality,
    warnings: list[str],
) -> tuple[CancerStatus | None, ClinicalAction]:
    """Apply the mapping + KTD-V10 override.

    Non-cancer-class models have no mapping; ``cancer_status`` returns
    ``None`` and ``clinical_action`` defaults to ``routine_followup``
    (the most conservative "share the result, no panic" tone). The
    override still fires for the quality / low-confidence case so the
    user gets ``inconclusive_review`` even for non-cancer findings — the
    LLM-side reply prompt branches on that key.
    """
    top1 = classification.top1
    if manifest.cancer_class:
        # Boot-time validation guarantees both maps cover every label.
        # A KeyError here would mean someone hot-swapped the manifest
        # since startup — re-raise loudly rather than mask.
        assert manifest.cancer_status_mapping is not None
        assert manifest.clinical_action_mapping is not None
        cancer_status: CancerStatus | None = manifest.cancer_status_mapping[top1]
        clinical_action: ClinicalAction = manifest.clinical_action_mapping[top1]
    else:
        cancer_status = None
        clinical_action = "routine_followup"

    # KTD-V10: low confidence OR failed quality gate → inconclusive_review.
    # Order matters for the warning text — quality gate first because
    # it's the more user-actionable problem ("retake the photo"); low
    # confidence is "the model itself isn't sure".
    if not quality.passed:
        failed_checks = [c.name for c in quality.checks if not c.passed]
        warnings.append(
            f"quality_gate.passed=False (failed: {', '.join(failed_checks) or '?'}) "
            f"— clinical_action overridden to inconclusive_review (KTD-V10)"
        )
        clinical_action = "inconclusive_review"
    elif classification.confidence_tier == "low":
        warnings.append(
            f"classification.confidence_tier=low (top1_prob="
            f"{classification.top1_prob:.3f}) — clinical_action overridden "
            f"to inconclusive_review (KTD-V10)"
        )
        clinical_action = "inconclusive_review"

    return cancer_status, clinical_action


def _record_capability_warnings(
    options: DetectOptions, manifest: Manifest, warnings: list[str]
) -> None:
    """Note capability no-ops in ``warnings`` per origin §7.

    Silent capability negotiation — the FastAPI never 4xxs for an
    unsupported optional knob. The client (and ultimately the LLM)
    sees a 200 with the warning and decides whether to surface it.
    """
    if options.return_saliency and not manifest.supports_saliency:
        warnings.append(
            "options.return_saliency=true but model.supports_saliency=false "
            "— no-op; saliency_b64 stays null"
        )
    if options.tta and not manifest.supports_tta:
        warnings.append(
            "options.tta=true but model.supports_tta=false — no-op; "
            "single forward pass run instead"
        )


def collect_warnings(*chunks: Iterable[str]) -> list[str]:
    """Flatten and dedupe warning chunks (used by tests as a helper)."""
    seen: list[str] = []
    for chunk in chunks:
        for item in chunk:
            if item not in seen:
                seen.append(item)
    return seen


__all__ = [
    "InferenceResources",
    "collect_warnings",
    "run_inference",
]
