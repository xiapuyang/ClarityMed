"""HTTP wire schemas for the vision server.

Split from ``schemas.py`` along the same axis symptoms uses: config + result
models stay in ``core/vision/schemas.py`` (client + server share them), HTTP
request envelopes live here. The matching ``servers/vision/wire.py`` is a
re-export shim so the server module imports look natural.

``RawDetection`` is the ``POST /v1/detect`` response body — already defined in
``schemas.py`` and re-exported here as ``DetectResponse`` so callers see a
consistent "request + response" pair without the schemas module also being
the wire surface.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from claritymed.core.medical_clip.schemas import ImagePayload, Modality
from claritymed.core.vision.schemas import (
    ModelFramework,
    RawDetection as DetectResponse,
)


class DetectOptions(BaseModel):
    """Request-side knobs for ``POST /v1/detect``.

    Each flag is silently no-op when the loaded model's manifest does not
    declare the matching capability (``supports_saliency`` / ``supports_tta``).
    The server records the no-op in ``DetectResponse.warnings`` so the caller
    knows the flag had no effect — there's no 4xx for "unsupported option",
    that would be a brittle contract for an optional capability negotiation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    return_segmentation: bool = True
    return_saliency: bool = False
    tta: bool = False


class DetectRequest(BaseModel):
    """``POST /v1/detect`` request body.

    ``model_id`` is optional — when absent the server falls back to the
    ``DiseaseSpec.primary_model_id`` of the resolved disease. ``language``
    is forwarded so the server can localize ``warnings`` and ``labels_meta``
    descriptions before returning them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(min_length=1, max_length=64)
    disease_id: str = Field(min_length=1, max_length=64)
    model_id: str | None = Field(default=None, min_length=1, max_length=64)
    image: ImagePayload
    language: Literal["en", "zh"] = "en"
    options: DetectOptions = Field(default_factory=lambda: DetectOptions())


class CatalogModel(BaseModel):
    """One model entry in ``GET /v1/catalog``.

    Boot-time cross-check target for ``VisionRegistry`` (KTD-V2). The
    client compares each field against ``configs/vision.yaml::models``
    and fails-loud at orchestrator boot when the server's truth disagrees
    with the committed config (e.g. server loaded a different
    ``manifest_sha`` than what the YAML pins).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    disease_id: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=64)
    model_version: str = Field(min_length=1, max_length=32)
    framework: ModelFramework
    task: Literal["classification", "classification+segmentation", "detection"]
    labels: list[str] = Field(min_length=1)
    cancer_class: bool
    accepted_modality: Modality
    manifest_sha: str = Field(pattern=r"^[a-f0-9]{64}$")
    expected_ms: int = Field(ge=1, le=60_000)
    supports_saliency: bool = False
    supports_tta: bool = False


class CatalogResponse(BaseModel):
    """``GET /v1/catalog`` response body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    models: list[CatalogModel] = Field(default_factory=list)


class HealthLoadedModel(BaseModel):
    """One entry of the health summary's loaded-model list."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    disease_id: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=64)


class HealthResponse(BaseModel):
    """``GET /health`` response body for the vision server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok", "loading"]
    models_loaded: list[HealthLoadedModel] = Field(default_factory=list)
    uptime_s: int = Field(ge=0)


class ErrorDetail(BaseModel):
    """Inner payload of the uniform error envelope.

    Stable, machine-readable ``code`` (e.g. ``modality_mismatch``,
    ``unknown_disease``) is the field clients dispatch on; ``details``
    carries context the client / LLM can surface to the user without
    re-deriving it (e.g. ``model_accepts`` + ``image_modality`` on a
    modality mismatch).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1)
    request_id: str | None = None
    details: dict[str, object] | None = None


class ErrorResponse(BaseModel):
    """Uniform error envelope for 4xx / 5xx responses.

    Every endpoint returns this shape on failure so the client has a
    single parse path. Pydantic validation errors are normalized to
    ``code='bad_request'`` upstream of the handler so a client doesn't
    have to handle FastAPI's stock 422 shape separately.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    error: ErrorDetail


__all__ = [
    "CatalogModel",
    "CatalogResponse",
    "DetectOptions",
    "DetectRequest",
    "DetectResponse",
    "ErrorDetail",
    "ErrorResponse",
    "HealthLoadedModel",
    "HealthResponse",
]
