"""Pydantic wire models for the medical-clip server (BiomedCLIP).

The medical-clip server owns the canonical ``Modality`` Literal. The
vision module imports it from here for :class:`ModelSpec.accepted_modality`
so the modality vocabulary is defined exactly once. Adding a new
modality therefore requires editing only this file plus the BiomedCLIP
candidate prompts in ``configs/medical_clip.yaml``.

Server-side processing (origin brainstorm §5.5):

* ``modality`` collapses to ``"unknown"`` when ``top1_score`` falls
  below ``configs/medical_clip.yaml::tasks.modality.gating.min_confidence``.
* ``is_medical`` is ``true`` only when ``top1`` is one of the medical
  labels (everything except ``photo`` / ``document`` / ``unknown``) AND
  ``top1_score >= min_medical_confidence``. The conjunction keeps a
  borderline-confident "photo" from leaking through as a medical scan.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Canonical modality vocabulary. Order matches the BiomedCLIP candidate
# list in ``configs/medical_clip.yaml::tasks.modality.candidates``;
# ``unknown`` is the fallback emitted when the top1 score is below the
# gating threshold. Adding a value here requires (a) a matching prompt
# block in the medical-clip YAML and (b) at least one disease in
# ``configs/vision.yaml`` whose ``accepted_modality`` references it,
# otherwise the new modality has no downstream consumer.
Modality = Literal[
    "ultrasound",
    "ct",
    "xray",
    "dermoscopy",
    "histopathology",
    "photo",
    "document",
    "unknown",
]


class ImagePayload(BaseModel):
    """One image carried over the wire (request bodies).

    ``sha256`` is the content digest computed by the attachment pipeline
    when the blob was first stored; the server re-hashes the decoded
    bytes and rejects the request when the digests disagree so a
    tampered or stale ``data_b64`` cannot bypass the attachment-ingest
    modality tag.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    data_b64: str = Field(min_length=1)


class ModalityScore(BaseModel):
    """One row of the BiomedCLIP scoreboard."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: Modality
    score: float = Field(ge=0.0, le=1.0)


class ModalityRequest(BaseModel):
    """``POST /v1/classify_modality`` request body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, max_length=64)
    image: ImagePayload


class ModalityResponse(BaseModel):
    """``POST /v1/classify_modality`` response body.

    ``confidence`` is the post-softmax ``top1`` score — the raw value
    *before* the ``min_confidence`` gate clamps :attr:`modality` to
    ``unknown``. Keeping the raw score lets downstream consumers
    (audit, evals) measure how often the gate fired without re-running
    the model. ``is_medical`` already encodes both the modality AND the
    confidence floor, so callers should branch on it instead of
    re-checking the score.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, max_length=64)
    modality: Modality
    confidence: float = Field(ge=0.0, le=1.0)
    is_medical: bool
    scores: list[ModalityScore] = Field(min_length=1)
    elapsed_ms: int = Field(ge=0)


class HealthResponse(BaseModel):
    """``GET /health`` response body for the medical-clip server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok", "loading"]
    model_id: str = Field(min_length=1)
    model_revision: str | None = Field(
        default=None,
        description=(
            "Pinned HuggingFace revision hash. None during early dev "
            "before the pin lands; once set the lifespan check refuses "
            "to start when HF Hub serves a different revision."
        ),
    )
    tasks_loaded: list[str] = Field(default_factory=list)
    device: str = Field(min_length=1)
    uptime_s: int = Field(ge=0)
