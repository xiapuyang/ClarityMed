"""Factory for OCR providers.

Reads ``configs/ocr.yaml`` via ``load_ocr_config()`` and returns a
``RoutingOcrProvider`` built from the configured chains.

Builds each chain entry in order, filters by ``phi_policy``, hands the
list to ``RoutingOcrProvider``. Failed-to-construct providers (missing
optional extra, MineRU without env opt-in) are skipped with a logged
warning so a partial install still gets a working chain.

Adding a new backend:

1. Implement ``OcrProvider`` in a new module under this package; set
   ``label`` to the short name used in configs.
2. Add its id to ``ProviderName`` in ``schemas/ocr.py``.
3. Add a builder in ``_build_provider`` below.
4. Update ``configs/ocr.yaml`` docs.
"""

from __future__ import annotations

import logging
import os

from claritymed.core.ocr.base import OcrProvider
from claritymed.errors import MinerUNotAllowed, MissingApiKeyError

logger = logging.getLogger(__name__)


def chain_supported_extensions(cfg) -> frozenset[str] | None:
    """Compute the union of ``supported_extensions`` across active chains.

    Resolves provider names to their *class-level* ``supported_extensions``
    without instantiating any provider — the paste-time gate runs on
    every clipboard paste and must not pay model-construction cost.

    Applies the same ``phi_policy`` filter that ``_filter_chain_for_policy``
    applies at composition time, so a cloud-only provider listed in
    ``ocr.yaml`` under ``phi_policy: local-only`` does NOT widen the
    accepted set.

    ``cfg.text_extensions`` is unioned in because those bypass the OCR
    worker via the paste fast-path (``ClarityMedApp._is_text_extension``)
    and are equally valid paste targets.

    Returns:
        ``frozenset`` of accepted extensions (with leading dots,
        lowercase). ``None`` if any active provider declares
        ``supported_extensions = None`` ("catch-all") — the gate
        should treat that as "no gating" because any restriction we'd
        add would be more restrictive than the actual chain behavior.
    """
    from claritymed.core.ocr.llm_provider import LLMOcrProvider
    from claritymed.core.ocr.marker_provider import MarkerOcrProvider
    from claritymed.core.ocr.mineru_provider import MineRUOcrProvider
    from claritymed.core.ocr.pandoc_provider import PandocOcrProvider
    from claritymed.core.ocr.pymupdf_provider import PyMuPDFOcrProvider
    from claritymed.core.ocr.rapidocr_provider import RapidOcrProvider

    classes_by_name: dict[str, type[OcrProvider]] = {
        "pymupdf": PyMuPDFOcrProvider,
        "marker": MarkerOcrProvider,
        "pandoc": PandocOcrProvider,
        "rapidocr": RapidOcrProvider,
        "llm": LLMOcrProvider,
        "mineru": MineRUOcrProvider,
    }
    accepted: set[str] = set()
    for entry in list(cfg.document_chain) + list(cfg.image_chain):
        cls = classes_by_name.get(entry.name)
        if cls is None:
            continue
        # Mirror ``_filter_chain_for_policy``: an entry whose effective
        # ``is_local`` is False is dropped under local-only policy.
        # The entry override (``entry.is_local``) wins over the class
        # default just like at runtime.
        effective_is_local = (
            entry.is_local if entry.is_local is not None else cls.is_local
        )
        if cfg.phi_policy == "local-only" and not effective_is_local:
            continue
        supported = cls.supported_extensions
        if supported is None:
            return None
        accepted.update(supported)
    accepted.update(cfg.text_extensions)
    return frozenset(accepted)


def make_ocr_provider() -> OcrProvider:
    """Build a ``RoutingOcrProvider`` from ``configs/ocr.yaml``.

    The routing-disambiguation set is derived inside
    ``RoutingOcrProvider`` from each chain's non-vision providers —
    no separate ``document_extensions`` parameter is needed. Plain-text
    extensions (``cfg.text_extensions``) are handled by the paste-time
    fast-path before the worker is invoked at all; they never reach
    the router, so no widening is required here.
    """
    from claritymed.core.ocr.routing_provider import RoutingOcrProvider
    from claritymed.core.schemas.ocr import load_ocr_config

    cfg = load_ocr_config()
    document_chain = _build_chain(cfg.document_chain, cfg)
    image_chain = _build_chain(cfg.image_chain, cfg)
    return RoutingOcrProvider(
        document_chain=document_chain,
        image_chain=image_chain,
        phi_policy=cfg.phi_policy,
    )


def _build_chain(entries, cfg) -> list[OcrProvider]:
    """Construct each chain entry, skipping ones that fail to build.

    ``MinerUNotAllowed`` (env-gate) and ``ImportError`` (missing extra)
    are the expected skip reasons; logged at INFO so operators see why
    a chain shrank without being scary. Other errors propagate.
    """
    result: list[OcrProvider] = []
    for entry in entries:
        try:
            provider = _build_provider(entry.name, cfg, chain_entry=entry)
        except MinerUNotAllowed:
            logger.info(
                "ocr chain: skipping mineru entry — CLARITYMED_ALLOW_MINERU not set"
            )
            continue
        except ImportError as exc:
            extra = entry.optional_extra or "<unknown>"
            logger.info(
                "ocr chain: skipping %s entry — optional extra %r not installed (%s)",
                entry.name,
                extra,
                exc,
            )
            continue
        # Per-entry override: a chain entry may force is_local=False (e.g.
        # an LLM pointing at a cloud model).
        if entry.is_local is not None:
            provider.is_local = entry.is_local
        result.append(provider)
    return result


def _build_provider(name: str | None, cfg, *, chain_entry=None) -> OcrProvider:
    if name == "pymupdf":
        from claritymed.core.ocr.pymupdf_provider import PyMuPDFOcrProvider

        return PyMuPDFOcrProvider()
    if name == "marker":
        from claritymed.core.ocr.marker_provider import MarkerOcrProvider

        return MarkerOcrProvider()
    if name == "pandoc":
        from claritymed.core.ocr.pandoc_provider import PandocOcrProvider

        return PandocOcrProvider()
    if name == "rapidocr":
        from claritymed.core.ocr.rapidocr_provider import RapidOcrProvider

        return RapidOcrProvider()
    if name == "mineru":
        return _make_mineru_provider(cfg)
    if name == "llm":
        return _make_llm_provider(cfg, chain_entry=chain_entry)
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


def _make_llm_provider(cfg, *, chain_entry=None) -> OcrProvider:
    from claritymed.core.ocr.llm_provider import LLMOcrProvider

    # Chain entries can carry their own provider config inline so a chain
    # can declare multiple distinct LLM entries (e.g. local vision +
    # cloud vision). A chain entry that names ``llm`` but supplies neither
    # ``provider_id`` nor ``model`` falls back to the top-level ``llm:``
    # block.
    use_chain = chain_entry is not None and (
        chain_entry.provider_id is not None or chain_entry.model is not None
    )
    llm_cfg = chain_entry if use_chain else cfg.llm
    if llm_cfg is None:
        raise ValueError(
            "ocr.yaml: 'llm' chain entry requires either an inline "
            "provider_id/model on the entry or a top-level 'llm:' block."
        )

    if llm_cfg.provider_id is not None:
        model = _model_from_catalog(llm_cfg.provider_id)
    else:
        model = _model_from_inline(llm_cfg)

    # Default to local; chain composer overrides via ``ChainEntry.is_local``
    # when an entry points at a cloud vision model.
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
