"""OCR service configuration schema.

``OcrConfig`` is parsed from ``configs/ocr.yaml``.  The only supported
provider today is ``llm``, which sends the file to a vision-capable LLM
using pydantic-ai ``BinaryContent``.

``LLMOcrConfig`` supports two ways to specify the model:

* ``provider_id`` — reference an existing entry in ``models.yaml`` by ID.
  The factory resolves the full connection details (model, base_url,
  api_key_env) from the catalog.  Use this to avoid duplicating config.
* ``model`` (+optional ``base_url``, ``api_key_env``) — inline connection
  details, following the same two-shape convention as ``ProviderConfig``.

Exactly one of ``provider_id`` or ``model`` must be set.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LLMOcrConfig(BaseModel):
    """Connection config for the LLM-based OCR backend.

    Set ``provider_id`` to reuse an existing ``models.yaml`` provider entry,
    or set ``model`` (+ optional ``base_url`` / ``api_key_env``) for inline
    connection details.
    """

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


class OcrConfig(BaseModel):
    """Parsed ``configs/ocr.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["llm"] = "llm"
    llm: LLMOcrConfig | None = None


def load_ocr_config() -> OcrConfig:
    """Parse ``configs/ocr.yaml`` through the mtime-cached loader."""
    from claritymed.config import load_yaml

    raw = load_yaml("ocr.yaml")
    return OcrConfig.model_validate(raw) if raw else OcrConfig()
