"""Factory for OCR providers.

Reads ``configs/ocr.yaml`` via ``load_ocr_config()`` and returns the
appropriate ``OcrProvider``.  Callers never import a concrete provider
directly — they call ``make_ocr_provider()`` and depend only on the
``OcrProvider`` interface.

Adding a new backend:
1. Implement ``OcrProvider`` in a new module under this package.
2. Add its id to the ``Literal`` in ``OcrConfig.provider``.
3. Add a case in ``make_ocr_provider`` below.
4. Update ``configs/ocr.yaml`` docs.
"""

from __future__ import annotations

import os

from claritymed.core.ocr.base import OcrProvider
from claritymed.errors import MissingApiKeyError


def make_ocr_provider() -> OcrProvider:
    """Build the configured OCR provider from ``configs/ocr.yaml``."""
    from claritymed.core.schemas.ocr import load_ocr_config

    cfg = load_ocr_config()

    if cfg.provider == "llm":
        return _make_llm_provider(cfg)

    raise ValueError(
        f"Unknown OCR provider={cfg.provider!r} in ocr.yaml. Supported values: 'llm'."
    )


def _make_llm_provider(cfg) -> OcrProvider:
    """Construct an LLMOcrProvider from the ``llm`` section of ocr.yaml."""
    from claritymed.core.ocr.llm_provider import LLMOcrProvider

    llm_cfg = cfg.llm
    if llm_cfg is None:
        raise ValueError(
            "ocr.yaml: provider='llm' requires an 'llm:' section with either "
            "provider_id or model."
        )

    if llm_cfg.provider_id is not None:
        model = _model_from_catalog(llm_cfg.provider_id)
    else:
        model = _model_from_inline(llm_cfg)

    return LLMOcrProvider(model)


def _model_from_catalog(provider_id: str):
    """Resolve a pydantic-ai Model from an existing models.yaml entry."""
    from claritymed.core.llm.model import build_model
    from claritymed.stores.models import resolve_provider

    provider = resolve_provider(override=provider_id)
    return build_model(provider)


def _model_from_inline(llm_cfg):
    """Build a pydantic-ai Model from inline ocr.yaml llm fields."""
    from pydantic_ai.models import infer_model

    if llm_cfg.base_url is None:
        return infer_model(llm_cfg.model)

    api_key = _resolve_api_key(llm_cfg)
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider

    return OpenAIChatModel(
        llm_cfg.model,
        provider=OllamaProvider(base_url=llm_cfg.base_url, api_key=api_key),
    )


def _resolve_api_key(llm_cfg) -> str | None:
    if llm_cfg.api_key_env is None:
        return None
    key = os.environ.get(llm_cfg.api_key_env)
    if not key:
        raise MissingApiKeyError(
            f"ocr.yaml llm.api_key_env={llm_cfg.api_key_env!r} is set "
            "but the environment variable is missing or empty."
        )
    return key
