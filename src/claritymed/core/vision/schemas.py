"""Pydantic contracts for ``configs/vision.yaml`` and the LLM-facing tool.

Two axes live in this file:

1. **Config models** — ``VisionConfig`` / ``DiseaseSpec`` / ``ModelSpec``
   / ``ServerSpec`` / ``ToolConfig`` / ``OcrReportConfig``. Loaded from
   ``configs/vision.yaml``; cross-references are validated up-front so
   a typo'd ``primary_model_id`` or ``server_id`` fails at boot rather
   than mid-request.

2. **Tool-result models** — ``RawDetection`` / ``LLMDetectionPayload``
   / ``ModalityMismatchResult`` / ``LabelMeta``. The server returns
   ``RawDetection`` over the wire; ``orchestrator/features/vision_plugin``
   strips it into ``LLMDetectionPayload`` (binary masks removed, top-K
   labels, localized strings) before handing the dict back to the LLM.

The ``Manifest`` model sits between the two: it describes the on-disk
``manifest.json`` that ships next to a trained checkpoint. The
server's loader reads it at startup and pins the two-level sha256
chain (KTD-V7) against ``ModelSpec.manifest_sha256``.

Decoupling rationale (KTD-V1, KTD-V5): ``ClinicalAction`` is a sibling
enum to symptoms' ``SeverityTier`` — not a child. Mapping a malignant
finding to symptoms' ``Urgent`` tier would inherit acute-triage copy
("go to the ER") that is medically wrong for cancer findings; vision
owns its own enum, prompt file, and i18n bundle.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from claritymed.core.medical_clip.schemas import Modality

# Default sig-fig count adapters use when finalizing wire-bound
# probabilities. Two is the readability sweet spot: float-noise tails
# (``0.10000000000000002``) collapse to ``0.10`` so logs and audit
# payloads stay grep-friendly, while values below 0.01 (rare outlier
# classes) still keep two meaningful digits (``0.0042`` survives, where
# decimal-place rounding would collapse to ``0.00``).
PROB_SIG_FIGS = 2


def round_sig(value: float, sig_figs: int = PROB_SIG_FIGS) -> float:
    """Round ``value`` to ``sig_figs`` significant figures.

    ``round(0.001, 2)`` collapses to ``0.0`` (2 decimal places). For
    classification probabilities we want ``0.0010`` to survive — hence
    significant figures, not decimal places. NaN / inf / zero pass
    through unchanged because ``log10`` is undefined there.

    Adapters import this from the schemas module and apply it inside
    ``calibrate()`` (or wherever they finalize the probabilities) before
    constructing a :class:`ClassificationResult`. Keeping it out of the
    schema validator means precise inputs are honored — debugging,
    entropy calculations, and calibration audits all see the raw values
    the producer computed.
    """
    if not math.isfinite(value) or value == 0:
        return value
    digits = sig_figs - int(math.floor(math.log10(abs(value)))) - 1
    return round(value, digits)


# --- enums shared across config + result models ----------------------------

# What the user should DO next about a vision finding. Sibling axis to
# symptoms' SeverityTier — different vocabulary, different reply prompt,
# different i18n bundle. ``inconclusive_review`` is the override target
# when ``confidence_tier == "low"`` or the quality gate failed
# (KTD-V10).
ClinicalAction = Literal[
    "urgent_specialist",
    "soon_specialist",
    "routine_followup",
    "no_action",
    "inconclusive_review",
]

# Cancer-class membership for one top1 label. ``unknown`` is the
# explicit "I cannot tell" value; absence (``None``) is reserved for
# non-cancer-class diseases that never carry the field.
CancerStatus = Literal["benign", "malignant", "normal", "unknown"]

# Calibrated probability bucket. The mapping from raw ``top1_prob`` to
# tier lives server-side per checkpoint (``tune.py`` output) so the
# tier is comparable across models with different calibration curves.
ConfidenceTier = Literal["low", "medium", "high"]

# Inference framework backing one model. Adapter dispatch in the
# vision server keys off this — adding a new framework = one adapter
# file + one enum entry.
ModelFramework = Literal["pytorch", "onnx", "ultralytics"]


# --- config models (``configs/vision.yaml``) -------------------------------


class ServerSpec(BaseModel):
    """One vision server entry under ``configs/vision.yaml::servers``.

    ``base_url`` is loopback-only in v1 (KTD-V8 — PHI never leaves the
    box). The server's own startup assertion enforces this; the schema
    permits any URL so future deployments that bind a private host can
    do so without a schema bump.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    base_url: str = Field(min_length=1, max_length=256)
    expected_ms: int = Field(ge=1, le=60_000)
    health_check_interval_s: int = Field(default=300, ge=1, le=86_400)


class ModelSpec(BaseModel):
    """One model checkpoint registered against a disease.

    ``server_id`` (KTD-V2) points at ``ServerSpec.id`` so the client
    routes without runtime catalog discovery; the boot-time
    ``/v1/catalog`` cross-check (Unit 5) verifies the server actually
    loaded what this entry claims.

    ``accepted_modality`` (KTD-V3) gates inference upstream of the
    server — the tool body refuses to invoke a model whose declared
    modality does not match the image's tagged modality, so a
    breast-ultrasound model is never asked to read a chest X-ray.

    ``manifest_sha256`` is the committed root of the two-level
    integrity chain (KTD-V7). The server hashes the on-disk
    ``manifest.json`` at startup and refuses to load on mismatch.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    disease_id: str = Field(min_length=1, max_length=64)
    server_id: str = Field(min_length=1, max_length=64)
    framework: ModelFramework
    accepted_modality: Modality
    weights_subpath: str = Field(min_length=1, max_length=256)
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_ms: int = Field(ge=1, le=60_000)

    @model_validator(mode="after")
    def _accepted_modality_not_unknown(self) -> ModelSpec:
        if self.accepted_modality == "unknown":
            raise ValueError(
                f"models[id={self.id!r}].accepted_modality must be a concrete "
                f"modality, not 'unknown' — the hard gate has no useful "
                f"semantics against the fallback bucket"
            )
        return self

    @model_validator(mode="after")
    def _weights_subpath_relative(self) -> ModelSpec:
        path = self.weights_subpath
        if path.startswith("/") or path.startswith("~"):
            raise ValueError(
                f"models[id={self.id!r}].weights_subpath must be relative to "
                f"CLARITYMED_HOME/models/, got absolute path: {path!r}"
            )
        if ".." in path.replace("\\", "/").split("/"):
            raise ValueError(
                f"models[id={self.id!r}].weights_subpath must not contain "
                f"'..' segments: {path!r}"
            )
        return self


class DiseaseSpec(BaseModel):
    """One disease registered with the vision tool.

    ``primary_model_id`` is the canonical model for the disease —
    always tried first and used by every catalog / audit / intent
    surface. ``flow`` is the **fallback-only** list: the orchestrator
    runs ``primary_model_id`` first, then walks ``flow`` in order if
    the primary returned low confidence or was unreachable within
    ``tool.total_budget_ms``. ``flow`` must not list the primary again
    — it is implicitly prepended via :attr:`effective_flow`.

    ``cancer_class=True`` requires that every model serving this
    disease ships a manifest declaring ``cancer_status_mapping`` and
    ``clinical_action_mapping`` (enforced by ``Manifest._mappings_required``
    at server boot); without these the reply prompt's
    ``clinical_action`` branching has no data to drive it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    enabled: bool = True
    primary_model_id: str = Field(min_length=1, max_length=64)
    # Default empty list; combined with the implicit primary this still
    # gives a max effective chain length of 8 (1 primary + 7 fallbacks).
    flow: list[str] = Field(default_factory=list, max_length=7)
    cancer_class: bool = False
    intent_hints_i18n_key: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _primary_not_in_flow(self) -> DiseaseSpec:
        if self.primary_model_id in self.flow:
            raise ValueError(
                f"diseases[id={self.id!r}].primary_model_id="
                f"{self.primary_model_id!r} must NOT appear in "
                f"flow={self.flow!r}; flow lists fallbacks only, the "
                f"primary is prepended implicitly via effective_flow"
            )
        return self

    @model_validator(mode="after")
    def _flow_unique(self) -> DiseaseSpec:
        if len(set(self.flow)) != len(self.flow):
            raise ValueError(
                f"diseases[id={self.id!r}].flow must be unique, got {self.flow!r}"
            )
        return self

    @property
    def effective_flow(self) -> list[str]:
        """Primary + declared fallbacks. The order callers should iterate.

        ``primary_model_id`` is always ``effective_flow[0]``; the
        ``_primary_not_in_flow`` validator guarantees no duplicate work
        is needed here. Every runtime consumer (registry cross-check,
        orchestrator fallback loop, vision-server lifespan loader,
        ``/v1/detect`` model_id allow list) walks this property so the
        YAML stays uncluttered while the execution chain stays explicit.
        """
        return [self.primary_model_id, *self.flow]


class OcrReportConfig(BaseModel):
    """KTD-V6: when the image already carries a clinician report.

    ``min_chars`` and ``markers`` together form the cheap text-level
    heuristic that decides whether ``ocr_has_report=true`` propagates
    onto the ``<image>`` tag. False positives on textbook excerpts are
    documented as acceptable for v1 (the failure mode is "LLM uses OCR
    text instead of running the model" — conservative wrong direction).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_chars: int = Field(default=200, ge=1, le=100_000)
    markers: dict[str, list[str]] = Field(
        description=(
            "Per-language marker keywords; presence of any keyword (case-"
            "insensitive for EN, exact match for ZH) flips ocr_has_report "
            "to true when min_chars also satisfied."
        ),
    )

    @model_validator(mode="after")
    def _markers_non_empty(self) -> OcrReportConfig:
        for lang, words in self.markers.items():
            if not words:
                raise ValueError(
                    f"ocr_report.markers[{lang!r}] must contain at least one keyword"
                )
        return self


class ToolConfig(BaseModel):
    """Tool-body budgets and feature flags.

    ``shadow_inference_on_report_override`` defaults ``False`` (KTD-V9)
    — offline-eval mode only. Flipping it to ``True`` makes the tool
    fire a background inference even when ``ocr_has_report=True`` so
    the audit payload accumulates "what would the model have said?"
    data without exposing the user to model-vs-clinician disagreement.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_budget_ms: int = Field(default=20_000, ge=100, le=120_000)
    fallback_safety_factor: float = Field(default=1.5, ge=1.0, le=10.0)
    catalog_refresh_seconds: int = Field(default=300, ge=1, le=86_400)
    confirm_before_run: bool = True
    shadow_inference_on_report_override: bool = False
    top_k: int = Field(default=3, ge=1, le=10)


class VisionConfig(BaseModel):
    """Root of ``configs/vision.yaml``.

    Cross-references are validated up-front: every entry in each
    ``DiseaseSpec.effective_flow`` (i.e. ``primary_model_id`` and every
    fallback in ``flow``) resolves to a ``ModelSpec.id``; every
    ``ModelSpec.server_id`` resolves to a ``ServerSpec.id``; every
    ``ModelSpec.disease_id`` resolves to a ``DiseaseSpec.id``. Typos
    fail at load time, not mid-request.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    diseases: list[DiseaseSpec] = Field(min_length=1)
    servers: list[ServerSpec] = Field(min_length=1)
    models: list[ModelSpec] = Field(min_length=1)
    tool: ToolConfig = Field(default_factory=lambda: ToolConfig())
    ocr_report: OcrReportConfig

    @model_validator(mode="after")
    def _unique_disease_ids(self) -> VisionConfig:
        ids = [d.id for d in self.diseases]
        if len(set(ids)) != len(ids):
            raise ValueError(f"diseases[].id must be unique, got {ids!r}")
        return self

    @model_validator(mode="after")
    def _unique_model_ids(self) -> VisionConfig:
        ids = [m.id for m in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError(f"models[].id must be unique, got {ids!r}")
        return self

    @model_validator(mode="after")
    def _unique_server_ids(self) -> VisionConfig:
        ids = [s.id for s in self.servers]
        if len(set(ids)) != len(ids):
            raise ValueError(f"servers[].id must be unique, got {ids!r}")
        return self

    @model_validator(mode="after")
    def _model_disease_refs_resolve(self) -> VisionConfig:
        known = {d.id for d in self.diseases}
        for m in self.models:
            if m.disease_id not in known:
                raise ValueError(
                    f"models[id={m.id!r}].disease_id={m.disease_id!r} "
                    f"not in diseases[]: known={sorted(known)!r}"
                )
        return self

    @model_validator(mode="after")
    def _model_server_refs_resolve(self) -> VisionConfig:
        known = {s.id for s in self.servers}
        for m in self.models:
            if m.server_id not in known:
                raise ValueError(
                    f"models[id={m.id!r}].server_id={m.server_id!r} "
                    f"not in servers[]: known={sorted(known)!r}"
                )
        return self

    @model_validator(mode="after")
    def _disease_flow_refs_resolve(self) -> VisionConfig:
        known = {m.id for m in self.models}
        for d in self.diseases:
            unknown = [mid for mid in d.effective_flow if mid not in known]
            if unknown:
                raise ValueError(
                    f"diseases[id={d.id!r}] references unknown models "
                    f"{unknown!r} (primary_model_id + flow); "
                    f"known: {sorted(known)!r}"
                )
        return self

    @model_validator(mode="after")
    def _cancer_class_modality_consistent(self) -> VisionConfig:
        models_by_id = {m.id: m for m in self.models}
        for d in self.diseases:
            if not d.cancer_class:
                continue
            modalities = {
                models_by_id[mid].accepted_modality for mid in d.effective_flow
            }
            if len(modalities) > 1:
                raise ValueError(
                    f"diseases[id={d.id!r}] models declare mixed "
                    f"accepted_modality values {sorted(modalities)!r}; a "
                    f"cancer-class disease's fallback chain must stay on "
                    f"a single modality so the upstream hard gate is "
                    f"meaningful"
                )
        return self


# --- on-disk manifest model (``manifest.json`` next to weights) ------------


class LabelMeta(BaseModel):
    """Per-label metadata block. Loaded from manifest, returned verbatim.

    The ``description`` is translator-facing copy (one short sentence);
    the LLM can surface it when the user asks "what does benign mean
    here?". ``cancer_status`` + ``clinical_action`` mirror the manifest's
    cancer_status_mapping / clinical_action_mapping so the response
    payload doesn't have to round-trip through three different shapes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str = Field(min_length=1, max_length=512)
    cancer_status: CancerStatus | None = None
    clinical_action: ClinicalAction | None = None


class ConfidenceThresholds(BaseModel):
    """Top1-probability boundaries between low/medium/high confidence tiers.

    Written by the tune phase per-checkpoint so each model's calibration
    curve sets its own tier boundaries. The adapter applies
    ``low_max < p ≤ medium_max → medium``, ``p > medium_max → high``,
    everything below ``low_max`` is ``"low"`` (which triggers KTD-V10).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    low_max: float = Field(ge=0.0, le=1.0)
    medium_max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> ConfidenceThresholds:
        if not self.low_max < self.medium_max:
            raise ValueError(
                f"confidence_thresholds.low_max ({self.low_max}) must be "
                f"strictly less than medium_max ({self.medium_max})"
            )
        return self


class TunedInferenceParams(BaseModel):
    """Inference-time parameters tuned by the post-training tune phase.

    Every field is optional with a sane fallback baked into the adapter,
    so a manifest written before the tune phase landed (or by a server
    that doesn't tune) still loads. The fields collectively cover the
    non-HP knobs that materially shift the production composite score
    (recall × dice). Carrying them on the manifest — not in
    ``configs/vision.yaml`` — keeps tuning a per-checkpoint concern: a
    re-trained model gets re-tuned without any config churn.

    Field semantics:

    * ``temperature`` — logit scaling applied **before** softmax in
      ``calibrate()``. ``T > 1`` softens probabilities (spreads mass),
      ``T < 1`` sharpens. Default behavior when absent: ``T = 1.0``
      (bare softmax).
    * ``classification_thresholds`` — per-label probability cutoffs
      for "treat this class as the prediction". When present, the
      top1 selection still uses argmax for tie-breaking but a class
      is only allowed to be top1 if its probability ≥ its threshold.
      v1 BUSI primarily uses the malignant cutoff to trade recall vs
      precision on the medically-critical class.
    * ``seg_threshold`` — sigmoid cutoff for binarizing the soft
      segmentation mask. Affects dice + bbox + area_ratio.
    * ``confidence_thresholds`` — see :class:`ConfidenceThresholds`.
    * ``tta_default`` — default value of ``DetectOptions.tta`` when
      the client doesn't override. ``None`` defers to the client's
      explicit choice (`False` in the schema default).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float | None = Field(default=None, gt=0.0, le=10.0)
    classification_thresholds: dict[str, float] | None = None
    seg_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_thresholds: ConfidenceThresholds | None = None
    tta_default: bool | None = None

    @model_validator(mode="after")
    def _thresholds_in_unit(self) -> TunedInferenceParams:
        if self.classification_thresholds is None:
            return self
        for label, value in self.classification_thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"classification_thresholds[{label!r}]={value} must be in [0, 1]"
                )
        return self


class Manifest(BaseModel):
    """On-disk ``manifest.json`` describing one trained checkpoint.

    The server reads this at startup and pins the two-level sha256
    chain (KTD-V7): ``ModelSpec.manifest_sha256`` must equal the
    SHA-256 of the manifest file's bytes, and ``sha256_weights`` must
    equal the SHA-256 of the weights file on disk. Mismatch at either
    layer aborts startup.

    When ``cancer_class=True`` the manifest must declare both
    ``cancer_status_mapping`` and ``clinical_action_mapping`` covering
    every entry in ``labels`` — without those the reply prompt's
    ``clinical_action`` branching has no data to drive it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(min_length=1, max_length=64)
    model_version: str = Field(min_length=1, max_length=32)
    framework: ModelFramework
    accepted_modality: Modality
    sha256_weights: str = Field(pattern=r"^[a-f0-9]{64}$")
    task: Literal["classification", "classification+segmentation", "detection"]
    labels: list[str] = Field(min_length=1, max_length=64)
    labels_meta: dict[str, LabelMeta]
    cancer_class: bool = False
    cancer_status_mapping: dict[str, CancerStatus] | None = None
    clinical_action_mapping: dict[str, ClinicalAction] | None = None
    supports_saliency: bool = False
    supports_tta: bool = False
    model_card_url: str | None = None
    # Encoder architecture identifier — the adapter passes this to
    # build_busi_model at load time so resnet50 / efficientnet_b0
    # checkpoints get the right matching architecture (the seg head's
    # decoder channels depend on the encoder). Default keeps old
    # manifests valid; train.py always writes the concrete choice.
    backbone: str = "custom_unet"
    tuned_inference: TunedInferenceParams | None = None

    @model_validator(mode="after")
    def _tuned_threshold_keys_valid(self) -> Manifest:
        """Reject classification_thresholds with unknown label keys.

        Pre-launch invariant: a tune output that points at a missing
        label means train/tune ran against a different label set than
        the manifest declares — that's drift we want to catch at boot,
        not silently ignore.
        """
        if self.tuned_inference is None:
            return self
        keys = self.tuned_inference.classification_thresholds
        if keys is None:
            return self
        unknown = sorted(set(keys) - set(self.labels))
        if unknown:
            raise ValueError(
                f"manifest(model_id={self.model_id!r}).tuned_inference."
                f"classification_thresholds has unknown labels {unknown!r}; "
                f"known labels: {sorted(self.labels)!r}"
            )
        return self

    @model_validator(mode="after")
    def _labels_meta_covers_labels(self) -> Manifest:
        missing = [label for label in self.labels if label not in self.labels_meta]
        if missing:
            raise ValueError(
                f"manifest(model_id={self.model_id!r}).labels_meta missing "
                f"entries for labels {missing!r}"
            )
        return self

    @model_validator(mode="after")
    def _cancer_class_mappings_complete(self) -> Manifest:
        if not self.cancer_class:
            return self
        if self.cancer_status_mapping is None:
            raise ValueError(
                f"manifest(model_id={self.model_id!r}, cancer_class=True) "
                f"must declare cancer_status_mapping"
            )
        if self.clinical_action_mapping is None:
            raise ValueError(
                f"manifest(model_id={self.model_id!r}, cancer_class=True) "
                f"must declare clinical_action_mapping"
            )
        missing_cs = [
            label for label in self.labels if label not in self.cancer_status_mapping
        ]
        if missing_cs:
            raise ValueError(
                f"manifest(model_id={self.model_id!r}).cancer_status_mapping "
                f"missing entries for labels {missing_cs!r}"
            )
        missing_ca = [
            label for label in self.labels if label not in self.clinical_action_mapping
        ]
        if missing_ca:
            raise ValueError(
                f"manifest(model_id={self.model_id!r}).clinical_action_mapping "
                f"missing entries for labels {missing_ca!r}"
            )
        return self


# --- result models (server response + LLM-facing transform) ----------------


class ClassificationResult(BaseModel):
    """Per-image classification head output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    labels: list[str] = Field(min_length=1)
    probabilities: list[float] = Field(min_length=1)
    top1: str = Field(min_length=1)
    top1_prob: float = Field(ge=0.0, le=1.0)
    confidence_tier: ConfidenceTier

    @model_validator(mode="after")
    def _shapes_match(self) -> ClassificationResult:
        if len(self.labels) != len(self.probabilities):
            raise ValueError(
                f"labels ({len(self.labels)}) and probabilities "
                f"({len(self.probabilities)}) must have equal length"
            )
        if self.top1 not in self.labels:
            raise ValueError(f"top1={self.top1!r} not in labels={self.labels!r}")
        return self


class SegmentationResult(BaseModel):
    """Per-image segmentation head output.

    ``mask_png_b64`` is stripped before the payload reaches the LLM
    (``result.to_llm_payload`` replaces it with a textual summary).
    Kept on the wire so the TUI can render an overlay when the
    user-facing UI grows that capability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mask_png_b64: str = Field(min_length=1)
    bbox: tuple[int, int, int, int]
    area_ratio: float = Field(ge=0.0, le=1.0)


class DetectionBox(BaseModel):
    """One bounding-box prediction from an object-detection model (YOLO).

    Coordinates are normalized to ``[0, 1]`` against the input image's
    pixel dimensions and laid out as ``xyxy`` (top-left, bottom-right).
    Normalizing at the adapter boundary keeps the wire payload
    resolution-independent — the TUI can render against the displayed
    image size without re-knowing the network's letterbox target.

    ``label`` is the manifest label (so the LLM-side reply prompt can
    reuse ``LabelMeta`` lookups) and ``confidence`` is the model's
    post-NMS class probability for this box.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1, max_length=64)
    confidence: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _xyxy_ordered(self) -> DetectionBox:
        if not (self.x2 >= self.x1 and self.y2 >= self.y1):
            raise ValueError(
                f"DetectionBox xyxy must satisfy x2>=x1 and y2>=y1; got "
                f"({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )
        return self


class ObjectDetectionResult(BaseModel):
    """Per-image object-detection head output (YOLO and similar).

    Returned alongside (not instead of) the per-disease
    ``ClassificationResult``: detection-style models still expose a
    derived "is this disease present" classification on
    :class:`RawDetection` so the existing reply prompt branches
    (``cancer_status`` / ``clinical_action``) keep working without a
    schema fork. The boxes themselves are an additive payload for
    detection-aware UIs and per-finding reasoning.

    Empty ``boxes`` means "model ran, found nothing above the confidence
    threshold" — distinct from "model didn't run" (sibling axis is
    ``None`` on :attr:`RawDetection.object_detection`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    boxes: list[DetectionBox] = Field(default_factory=list, max_length=1024)


class QualityCheck(BaseModel):
    """One row of the input-quality gate's per-check breakdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    score: float
    passed: bool


class InputQuality(BaseModel):
    """Server-side gate output. ``passed=False`` overrides clinical_action
    to ``inconclusive_review`` (KTD-V10) in the server's inference path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    checks: list[QualityCheck] = Field(default_factory=list)


class RawDetection(BaseModel):
    """Server ``POST /v1/detect`` response payload.

    This is the wire shape the vision-server returns and the
    orchestrator's tool body receives. ``to_llm_payload`` in
    ``core/vision/result.py`` (Unit 7) projects this into the leaner
    ``LLMDetectionPayload`` that goes back to the LLM.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, max_length=64)
    disease_id: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=64)
    model_version: str = Field(min_length=1, max_length=32)
    elapsed_ms: int = Field(ge=0)
    input_quality: InputQuality
    classification: ClassificationResult
    cancer_status: CancerStatus | None = None
    clinical_action: ClinicalAction
    segmentation: SegmentationResult | None = None
    object_detection: ObjectDetectionResult | None = None
    saliency_b64: str | None = None
    labels_meta: dict[str, LabelMeta]
    warnings: list[str] = Field(default_factory=list)
    model_card_url: str | None = None


class LLMDetectionPayload(BaseModel):
    """Tool-body return value handed back to the LLM.

    Differences from ``RawDetection``: top-K labels only, no binary
    mask base64 (replaced with a string summary the LLM can quote),
    localized label names. ``kind`` lets the reply-prompt template
    branch without re-parsing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["detection"] = "detection"
    request_id: str = Field(min_length=1, max_length=64)
    disease_id: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=64)
    model_version: str = Field(min_length=1, max_length=32)
    elapsed_ms: int = Field(ge=0)
    top_labels: list[str] = Field(min_length=1)
    top_probabilities: list[float] = Field(min_length=1)
    top1: str = Field(min_length=1)
    top1_prob: float = Field(ge=0.0, le=1.0)
    confidence_tier: ConfidenceTier
    cancer_status: CancerStatus | None = None
    clinical_action: ClinicalAction
    segmentation_summary: str | None = None
    labels_meta: dict[str, LabelMeta]
    warnings: list[str] = Field(default_factory=list)
    model_card_url: str | None = None


# --- tool-body short-circuit result types ----------------------------------
#
# These dicts are returned in lieu of running inference; the LLM-side
# reply prompt branches on ``kind`` to compose the right user-facing
# response. Keeping them as Pydantic models gives the tool body a
# single typed surface; ``.model_dump()`` is called before handoff so
# pydantic-ai sees a plain dict.


class ModalityMismatchResult(BaseModel):
    """KTD-V3 hard refuse — image modality does not match model.accepted_modality."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["modality_mismatch"] = "modality_mismatch"
    model_accepts: Modality
    image_modality: Modality
    message: str = Field(min_length=1)


class OcrOverrideResult(BaseModel):
    """KTD-V6 / Unit 7 step 2 — image carries clinician report; skip the tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["ocr_override"] = "ocr_override"
    message: str = Field(min_length=1)


class NotMedicalResult(BaseModel):
    """Image is not a medical scan; vision tools refuse upstream of inference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["not_medical"] = "not_medical"
    message: str = Field(min_length=1)


class UserDeclinedResult(BaseModel):
    """User answered no on the confirm modal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["user_declined"] = "user_declined"
    message: str = Field(min_length=1)


# --- server-side Protocol --------------------------------------------------


@runtime_checkable
class DiseaseVisionModel(Protocol):
    """Server-side contract for one per-disease vision model.

    Each adapter (``torch_adapter`` / ``onnx_adapter`` / ...) lifts a
    concrete checkpoint into this Protocol so the server's inference
    loop stays framework-agnostic. ``runtime_checkable`` enables the
    loader's ``isinstance`` guard at module-load time.

    v1 drops the v0-stub's ``conformal_set`` and ``ood_score`` methods —
    those land in v2 per ARCHITECTURE.md §7's "deferred" note.
    """

    spec: ModelSpec

    def preprocess(self, image: Any) -> Any: ...

    def predict(self, x: Any) -> Any: ...

    def calibrate(self, raw: Any) -> dict[str, float]: ...

    def segment(self, x: Any) -> SegmentationResult | None: ...

    def detect_boxes(self, x: Any) -> ObjectDetectionResult | None: ...

    def quality_gate(self, image: Any) -> InputQuality: ...
