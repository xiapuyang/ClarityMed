"""OCR service configuration schema.

``OcrConfig`` is parsed from ``configs/ocr.yaml``.

Two configuration styles (use one, not both):

* Legacy single-provider: ``document_provider`` (one of mineru/llm) +
  ``image.default``/``image.fallback``. Kept so the existing image-flow
  configs continue to work.
* Chain: ``document_chain`` / ``image_chain`` — ordered lists of
  ``{name, ...}`` entries. The first provider that returns text wins;
  ``OcrError`` from one falls through to the next. ``phi_policy``
  filters cloud providers out at composition time.

Provider names:

* ``pymupdf`` — local digital-PDF text extraction (no model load).
* ``marker``  — local scanned-PDF OCR (PyTorch; optional extra).
* ``pandoc``  — local office-format → markdown (optional extra).
* ``llm``     — pydantic-ai vision LLM (locality depends on the model).
* ``mineru``  — MineRU cloud SaaS (env-gated; structurally
  excluded from PHI chains via ``is_local=False``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ProviderName = Literal["pymupdf", "marker", "pandoc", "llm", "mineru"]
PhiPolicy = Literal["local-only", "any"]


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


class ChainEntry(BaseModel):
    """One entry in a provider chain (``document_chain`` / ``image_chain``).

    The ``optional_extra`` field documents which uv extra installs the
    provider; the factory uses it to surface a helpful error when a
    chain entry references a missing optional dep (e.g. ``marker`` listed
    in ``document_chain`` but ``marker-pdf`` not installed).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: ProviderName
    is_local: bool | None = None  # ``None`` → use provider's class default.
    optional_extra: str | None = None
    # ``llm`` chain entries may carry the same provider_id / model fields as
    # ``LLMOcrConfig``; modeled inline so a chain entry can self-describe a
    # vision model without forcing every YAML to declare a top-level
    # ``llm:`` block.
    provider_id: str | None = Field(default=None, min_length=1, max_length=64)
    model: str | None = Field(default=None, min_length=1)
    base_url: str | None = Field(default=None, min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1, max_length=64)


class OcrConfig(BaseModel):
    """Parsed ``configs/ocr.yaml``.

    Either the legacy fields or the chain fields are populated; the
    factory picks the right path. ``phi_policy`` filters chains and is
    ignored in legacy mode (caller manages cloud/local themselves).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Legacy fields — kept for back-compat with the existing TUI/CLI OCR
    # path.  Default values match the prior shipping config.
    document_provider: Literal["mineru", "pymupdf"] = "mineru"
    image: ImageOcrConfig = Field(default_factory=ImageOcrConfig)
    # New chain fields.  Empty by default; presence flips the factory to
    # chain mode.
    document_chain: list[ChainEntry] = Field(default_factory=list)
    image_chain: list[ChainEntry] = Field(default_factory=list)
    phi_policy: PhiPolicy = "any"
    mineru: MineRUOcrConfig | None = None
    llm: LLMOcrConfig | None = None


def load_ocr_config() -> OcrConfig:
    """Parse ``configs/ocr.yaml`` through the mtime-cached loader."""
    from claritymed.config import load_yaml

    raw = load_yaml("ocr.yaml")
    return OcrConfig.model_validate(raw) if raw else OcrConfig()
