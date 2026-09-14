"""Factory for translation providers.

Reads ``translation.provider`` from ``configs/retrieval.yaml`` and returns the
appropriate ``TranslationProvider`` implementation.  Callers never import a
concrete provider directly — they call ``make_translation_provider`` and depend
only on the ``TranslationProvider`` interface.

Adding a new backend:
1. Implement ``TranslationProvider`` in a new module under this package.
2. Add its id to the ``Literal`` in ``schemas.TranslationConfig``.
3. Add a case in ``make_translation_provider`` below.
4. Add the new id to ``retrieval.yaml: translation.provider`` docs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from claritymed.core.translation.base import TranslationProvider


def make_translation_provider(
    model: "Model",
    phi_kind: str = "local",
) -> "TranslationProvider":
    """Build the configured translation provider.

    Args:
        model: The pydantic-ai Model used for translation calls.
        phi_kind: ``"cloud"`` activates PHI scrubbing of input text before
            each translation call; ``"local"`` (default) skips scrubbing.
            Pass the resolved ``ProviderConfig.kind`` so translation inherits
            the same PHI policy as the main LLM.

    Falls back to ``LLMTranslationProvider`` if the config cannot be read,
    so a missing or malformed ``translation`` section never breaks startup.
    """
    from claritymed.core.phi.outbound_gate import make_outbound_gate
    from claritymed.core.translation.llm_provider import LLMTranslationProvider

    provider_id = "llm"
    try:
        from claritymed.core.rag.schemas import load_retrieval_config

        provider_id = load_retrieval_config().translation.provider
    except Exception:  # noqa: BLE001
        pass

    if provider_id == "llm":
        return LLMTranslationProvider(model, scrub_gate=make_outbound_gate(phi_kind))
    raise ValueError(
        f"Unknown translation.provider={provider_id!r} in retrieval.yaml. "
        "Supported values: 'llm'."
    )
