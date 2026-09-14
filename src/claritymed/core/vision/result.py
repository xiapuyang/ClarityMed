"""Wire-result → LLM-payload projection.

``to_llm_payload`` is the only public surface. Given a ``RawDetection``
returned by the vision-server, it produces a ``LLMDetectionPayload``
suitable for handing back to the LLM tool call.

What changes between the two shapes:

* Top-K labels only (``ToolConfig.top_k``). Near-zero probabilities are
  dropped so the LLM doesn't waste tokens reasoning about noise.
* Binary segmentation mask base64 is stripped and replaced with a one-
  line textual summary the LLM can quote ("lesion covers 3.4% of image,
  bounded by box (12,34,56,78)"). The TUI can fetch the full mask out
  of band when it grows that capability.
* Per-label localized names are looked up via ``configs/i18n/<lang>/
  vision.yaml::vision.label.<disease>.<label>.name`` and merged onto
  the ``labels_meta`` block; the ``description`` is also localized so
  the LLM can quote it when the user asks "what does X mean here?".
* ``clinical_action`` + ``cancer_status`` are promoted to top-level keys
  for prompt-template branching.
* ``model_version`` rides through verbatim — the LLM may surface it if
  the user asks "which model did you run?", and Unit 8's audit hook
  reads it off the payload to keep the audit row aligned with the
  reply.
"""

from __future__ import annotations

import logging

from claritymed.core.i18n.loader import t
from claritymed.core.vision.schemas import (
    LabelMeta,
    LLMDetectionPayload,
    RawDetection,
    SegmentationResult,
)

logger = logging.getLogger(__name__)


def to_llm_payload(
    raw: RawDetection,
    *,
    top_k: int,
    language: str,
) -> LLMDetectionPayload:
    """Project a ``RawDetection`` into the LLM-facing payload.

    Args:
        raw: Wire result from ``POST /v1/detect``.
        top_k: Number of label/probability pairs to keep, sorted by
            descending probability. Pulled from ``ToolConfig.top_k`` by
            the caller so the value isn't baked into the result layer.
        language: ``"en"`` / ``"zh"``. Selects the i18n bundle used to
            localize label names + descriptions.
    """
    # Sort indices by descending probability; keep the top_k slice plus
    # the top1 label even when it lands outside that window (e.g. when a
    # caller passes top_k=1 against a 7-class head with a strange top1).
    indexed = sorted(
        enumerate(raw.classification.probabilities),
        key=lambda pair: pair[1],
        reverse=True,
    )
    keep_indices: list[int] = []
    for idx, _ in indexed:
        if len(keep_indices) >= top_k:
            break
        keep_indices.append(idx)
    top1_idx = raw.classification.labels.index(raw.classification.top1)
    if top1_idx not in keep_indices:
        keep_indices.append(top1_idx)

    top_labels: list[str] = []
    top_probabilities: list[float] = []
    localized_meta: dict[str, LabelMeta] = {}
    for idx in keep_indices:
        label = raw.classification.labels[idx]
        top_labels.append(_localize_label_name(raw.disease_id, label, language))
        top_probabilities.append(round(float(raw.classification.probabilities[idx]), 4))
        original = raw.labels_meta.get(label)
        if original is None:
            continue
        localized_meta[label] = LabelMeta(
            description=_localize_label_description(
                raw.disease_id, label, original.description, language
            ),
            cancer_status=original.cancer_status,
            clinical_action=original.clinical_action,
        )

    top1_localized = _localize_label_name(
        raw.disease_id, raw.classification.top1, language
    )

    segmentation_summary = _summarize_segmentation(raw.segmentation)

    return LLMDetectionPayload(
        request_id=raw.request_id,
        disease_id=raw.disease_id,
        model_id=raw.model_id,
        model_version=raw.model_version,
        elapsed_ms=raw.elapsed_ms,
        top_labels=top_labels,
        top_probabilities=top_probabilities,
        top1=top1_localized,
        top1_prob=round(float(raw.classification.top1_prob), 4),
        confidence_tier=raw.classification.confidence_tier,
        cancer_status=raw.cancer_status,
        clinical_action=raw.clinical_action,
        segmentation_summary=segmentation_summary,
        labels_meta=localized_meta,
        warnings=list(raw.warnings),
        model_card_url=raw.model_card_url,
    )


def _localize_label_name(disease_id: str, label: str, language: str) -> str:
    """Return the localized label name; fall back to the raw label on miss."""
    key = f"vision.label.{disease_id}.{label}.name"
    localized = t(key, lang=language)
    # ``t`` returns the bare key as a sentinel when the lookup misses.
    # Surface that as the original label so the LLM-facing payload reads
    # cleanly rather than including a dotted i18n key.
    return label if localized == key else localized


def _localize_label_description(
    disease_id: str, label: str, fallback: str, language: str
) -> str:
    key = f"vision.label.{disease_id}.{label}.description"
    localized = t(key, lang=language)
    return fallback if localized == key else localized


def _summarize_segmentation(segmentation: SegmentationResult | None) -> str | None:
    """Build a one-line textual summary of the segmentation mask.

    The wire ``mask_png_b64`` itself is dropped — base64 PNGs blow up
    the LLM context for no diagnostic value the LLM can use directly.
    The summary keeps the area fraction (which is the actionable
    signal) and the bounding box (which lets the LLM ground its prose
    to a spatial region — "upper-right of the scan").
    """
    if segmentation is None:
        return None
    x0, y0, x1, y1 = segmentation.bbox
    pct = round(segmentation.area_ratio * 100, 1)
    return (
        f"lesion covers {pct}% of the image, bounded by box "
        f"(x0={x0}, y0={y0}, x1={x1}, y1={y1})"
    )


__all__ = ["to_llm_payload"]
