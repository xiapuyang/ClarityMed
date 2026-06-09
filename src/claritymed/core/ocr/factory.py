"""Factory for OCR providers.

Reads ``configs/ocr.yaml`` via ``load_ocr_config()`` and returns a
``RoutingOcrProvider`` that dispatches to the right backend by file type:
  - document files → MineRU
  - image files    → LLM (default) with optional MineRU fallback

Adding a new backend:
1. Implement ``OcrProvider`` in a new module under this package.
2. Add its id to the appropriate ``Literal`` in ``ocr.py`` schemas.
3. Add a builder in ``_build_provider`` below.
4. Update ``configs/ocr.yaml`` docs.
"""

from __future__ import annotations

import os

from claritymed.core.ocr.base import OcrProvider
from claritymed.errors import MissingApiKeyError


def make_ocr_provider() -> OcrProvider:
    """Build a ``RoutingOcrProvider`` from ``configs/ocr.yaml``."""
    from claritymed.core.ocr.routing_provider import RoutingOcrProvider
    from claritymed.core.schemas.ocr import load_ocr_config

    cfg = load_ocr_config()

    document_provider = _build_provider(cfg.document_provider, cfg)
    image_default = _build_provider(cfg.image.default, cfg)
    image_fallback = (
        _build_provider(cfg.image.fallback, cfg) if cfg.image.fallback else None
    )

    return RoutingOcrProvider(
        document_provider=document_provider,
        image_default=image_default,
        image_fallback=image_fallback,
    )


def _build_provider(name: str | None, cfg) -> OcrProvider:
    if name == "mineru":
        return _make_mineru_provider(cfg)
    if name == "llm":
        return _make_llm_provider(cfg)
    raise ValueError(f"Unknown OCR provider={name!r} in ocr.yaml.")


def _make_mineru_provider(cfg) -> OcrProvider:
    from claritymed.core.ocr.mineru_provider import MineRUOcrProvider

    mineru_cfg = cfg.mineru
    if mineru_cfg is None:
        from claritymed.core.schemas.ocr import MineRUOcrConfig

        mineru_cfg = MineRUOcrConfig()

    api_key = os.environ.get(mineru_cfg.api_key_env)
    if not api_key:
        raise MissingApiKeyError(
            f"ocr.yaml mineru.api_key_env={mineru_cfg.api_key_env!r} is set "
            "but the environment variable is missing or empty."
        )

    return MineRUOcrProvider(
        api_key=api_key,
        model_version=mineru_cfg.model_version,
        poll_interval=mineru_cfg.poll_interval,
        poll_timeout=mineru_cfg.poll_timeout,
    )


def _make_llm_provider(cfg) -> OcrProvider:
    from claritymed.core.ocr.llm_provider import LLMOcrProvider

    llm_cfg = cfg.llm
    if llm_cfg is None:
        raise ValueError(
            "ocr.yaml: image.default='llm' requires an 'llm:' section with either "
            "provider_id or model."
        )

    if llm_cfg.provider_id is not None:
        model = _model_from_catalog(llm_cfg.provider_id)
    else:
        model = _model_from_inline(llm_cfg)

    return LLMOcrProvider(model)


def _model_from_catalog(provider_id: str):
    from claritymed.core.llm.model import build_model
    from claritymed.stores.models import resolve_provider

    provider = resolve_provider(override=provider_id)
    return build_model(provider)


def _model_from_inline(llm_cfg):
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
