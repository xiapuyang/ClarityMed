"""OCR service configuration schema.

``OcrConfig`` is parsed from ``configs/ocr.yaml``.

Routing rules:
- Document files (.pdf .doc .docx .ppt .pptx .xls .xlsx) always go to
  ``document_provider`` (currently only ``"mineru"``).
- Image files go to ``image.default``; on failure, to ``image.fallback``.

Provider sections:
- ``mineru`` — MineRU standard API (requires MINERU_API_TOKEN).
- ``llm``    — vision-capable LLM via pydantic-ai BinaryContent.

``LLMOcrConfig`` supports two ways to specify the model:
* ``provider_id`` — reference an existing entry in ``models.yaml`` by ID.
* ``model`` (+optional ``base_url``, ``api_key_env``) — inline connection.
Exactly one of ``provider_id`` or ``model`` must be set.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OcrExtraction(BaseModel):
    """Structured output returned by the LLM OCR agent."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(
        description="True if document content was visible and extracted."
    )
    text: str = Field(
        default="", description="Extracted plain text. Empty when success=False."
    )
    failure_reason: str | None = Field(
        default=None,
        description="Short reason string when success=False, else null.",
    )


class LLMOcrConfig(BaseModel):
    """Connection config for the LLM-based OCR backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_id: str | None = Field(default=None, min_length=1, max_length=64)
    model: str | None = Field(default=None, min_length=1)
    base_url: str | None = Field(default=None, min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def _check_source(self) -> "LLMOcrConfig":
        has_id = self.provider_id is not None
        has_model = self.model is not None
        if has_id and has_model:
            raise ValueError(
                "Set either provider_id or model in ocr.yaml llm section, not both."
            )
        if not has_id and not has_model:
            raise ValueError(
                "ocr.yaml llm section requires either provider_id (to reference "
                "a models.yaml entry) or model (inline connection details)."
            )
        return self


class MineRUOcrConfig(BaseModel):
    """Connection config for the MineRU API backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key_env: str = Field(default="MINERU_API_TOKEN", min_length=1, max_length=64)
    model_version: Literal["pipeline", "vlm"] = "vlm"
    poll_interval: float = Field(default=3.0, gt=0)
    poll_timeout: float = Field(default=300.0, gt=0)


class ImageOcrConfig(BaseModel):
    """Provider routing config for image files."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default: Literal["llm", "mineru"] = "llm"
    fallback: Literal["mineru"] | None = "mineru"


class OcrConfig(BaseModel):
    """Parsed ``configs/ocr.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_provider: Literal["mineru"] = "mineru"
    image: ImageOcrConfig = Field(default_factory=ImageOcrConfig)
    mineru: MineRUOcrConfig | None = None
    llm: LLMOcrConfig | None = None


def load_ocr_config() -> OcrConfig:
    """Parse ``configs/ocr.yaml`` through the mtime-cached loader."""
    from claritymed.config import load_yaml

    raw = load_yaml("ocr.yaml")
    return OcrConfig.model_validate(raw) if raw else OcrConfig()
