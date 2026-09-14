"""OCR service configuration schema.

``OcrConfig`` is parsed from ``configs/ocr.yaml``.

Configuration is chain-only: ``document_chain`` / ``image_chain`` are
ordered lists of ``{name, ...}`` entries. The first provider that
returns text wins; ``OcrError`` from one falls through to the next.
``phi_policy`` filters cloud providers out at composition time.

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

ProviderName = Literal["pymupdf", "marker", "pandoc", "llm", "mineru", "rapidocr"]
PhiPolicy = Literal["local-only", "any"]


class OcrExtraction(BaseModel):
    """Structured output returned by the LLM OCR agent."""

    model_config = ConfigDict(frozen=True)

    status: Literal["done", "empty", "failed"] | None = Field(
        default=None,
        description=(
            "'done': text extracted successfully; "
            "'empty': document visible but no readable text; "
            "'failed': could not process (no attachment / unsupported format). "
            "Preferred over success for v2+ prompts."
        ),
    )
    success: bool | None = Field(
        default=None,
        description=(
            "True when content was extracted. Kept for v1 prompt compatibility; "
            "v2 prompts set status instead."
        ),
    )
    text: str = Field(
        default="",
        description=(
            "Extracted plain text exactly as written. "
            "Exclude watermarks, decorative stamps, and page numbers. "
            "Empty when status is 'empty' or 'failed'."
        ),
    )
    failure_reason: str | None = Field(
        default=None,
        description="Short reason (≤50 words) when status='failed', else null.",
    )
    modality: str | None = Field(
        default=None,
        description=(
            "Medical imaging modality inferred from image and text content. "
            "One of: ultrasound, ct, xray, dermoscopy, histopathology, photo, "
            "document, unknown. Null when the input is clearly not a "
            "medical/clinical document."
        ),
    )
    is_medical: bool | None = Field(
        default=None,
        description=(
            "True if the content is clinical or medical "
            "(imaging study, lab report, prescription, clinical note, etc.). "
            "False for non-medical documents. Null when not determinable."
        ),
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


_DEFAULT_TEXT_EXTENSIONS: tuple[str, ...] = (
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".log",
    ".xml",
    ".yaml",
    ".yml",
)


class ModalityFallbackConfig(BaseModel):
    """LLM provider used when BiomedCLIP is unavailable and no LLM-OCR ran.

    Shares the same shape as ``LLMOcrConfig`` — either reference a
    ``models.yaml`` entry via ``provider_id``, or supply inline connection
    details via ``model`` (+ optional ``base_url`` / ``api_key_env``).
    The referenced provider must have ``supports_vision: true``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_id: str | None = Field(default=None, min_length=1, max_length=64)
    model: str | None = Field(default=None, min_length=1)
    base_url: str | None = Field(default=None, min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def _check_source(self) -> "ModalityFallbackConfig":
        has_id = self.provider_id is not None
        has_model = self.model is not None
        if has_id and has_model:
            raise ValueError(
                "Set either provider_id or model in "
                "ocr.yaml modality_fallback section, not both."
            )
        if not has_id and not has_model:
            raise ValueError(
                "ocr.yaml modality_fallback section requires either "
                "provider_id or model."
            )
        return self


class OcrConfig(BaseModel):
    """Parsed ``configs/ocr.yaml`` — chain-only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_chain: list[ChainEntry] = Field(default_factory=list)
    image_chain: list[ChainEntry] = Field(default_factory=list)
    phi_policy: PhiPolicy = "any"
    # Extensions that bypass the OCR worker entirely — the paste handler /
    # rag CLI reads them as utf-8 text and writes the sentinel inline.
    # Keeps the worker queue dedicated to actually slow extractions.
    text_extensions: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_TEXT_EXTENSIONS)
    )
    mineru: MineRUOcrConfig | None = None
    llm: LLMOcrConfig | None = None
    # Optional LLM fallback for modality classification. Called when
    # BiomedCLIP is unavailable AND the OCR chain had no LLM provider
    # (so result.modality is None). The provider must support vision.
    modality_fallback: ModalityFallbackConfig | None = None

    @model_validator(mode="after")
    def _require_at_least_one_chain(self) -> "OcrConfig":
        if not self.document_chain and not self.image_chain:
            raise ValueError(
                "ocr.yaml: at least one of document_chain / image_chain must be set."
            )
        return self

    @model_validator(mode="after")
    def _normalize_text_extensions(self) -> "OcrConfig":
        """Lowercase + ensure leading dot so caller doesn't need to.

        Pydantic's ``frozen=True`` blocks ``self.text_extensions = ...``;
        re-route through ``object.__setattr__`` since this is the only
        in-model normalization and we want callers to read a clean list.
        """
        normalized = [
            (e if e.startswith(".") else f".{e}").lower() for e in self.text_extensions
        ]
        object.__setattr__(self, "text_extensions", normalized)
        return self


def load_ocr_config() -> OcrConfig:
    """Parse ``configs/ocr.yaml`` through the mtime-cached loader."""
    from claritymed.config import load_yaml

    raw = load_yaml("ocr.yaml")
    if not raw:
        raise ValueError("ocr.yaml is missing or empty.")
    return OcrConfig.model_validate(raw)
