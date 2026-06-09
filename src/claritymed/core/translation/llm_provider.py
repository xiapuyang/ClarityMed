"""LLM-backed translation provider using pydantic-ai Agent.

Covers four use cases:

* ``translate_query``  — short embedding queries; output is the bare translation.
* ``translate_answer`` — full LLM responses; preserves markdown, citations, structure.
* ``translate_term``   — single medical terms; output is the bare translation.
* ``translate``        — general-purpose fallback.

All async methods fall back to the original text on any error so callers are
never blocked by a translation failure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from claritymed.core.translation.base import (
    Language,
    TranslationProvider,
    detect_language,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)

# Registry keys for translation prompts. The language argument to
# ``PromptRegistry.get`` is the TARGET language, not the source.
#
# ``_PROMPT_NAME`` is the faithful translator used for answers, terms,
# and the general fallback — preserves wording, markdown, citations.
#
# ``_QUERY_PROMPT_NAME`` is a clinical-query rewriter used only by
# ``translate_query``. It produces a focused search query suitable for
# dense retrieval against textbook corpora, so deterministic and (future)
# agentic modes get the same retrieval quality that tool mode gets "for
# free" from LLM-side query crafting. Tool mode also routes through this
# step but is near-identity when the input is already a focused clinical
# phrase, so it is not a regression risk for the tool path.
_PROMPT_NAME = "translate"
_QUERY_PROMPT_NAME = "translate_query"


class LLMTranslationProvider(TranslationProvider):
    """LLM-backed translation via a pydantic-ai Agent.

    Each call is stateless — no message history is passed, so the context
    window stays minimal.  The ask pipeline carries its own chat history
    separately; mixing the two would inflate token cost with no benefit.
    """

    def __init__(self, model: "Model") -> None:
        self._model = model

    async def translate_query(self, query: str, *, target_lang: Language) -> str:
        """Rewrite an input into a focused clinical search query in ``target_lang``.

        Combines translation (if the input is in another language) and
        clinical reformulation in a single LLM call. The output is what
        the retriever embeds and what the reranker scores against — so
        conversational user phrasing is intentionally stripped here, not
        deeper in the pipeline.
        """
        try:
            return await self._call(
                query,
                target_lang=target_lang,
                context="query",
                prompt_name=_QUERY_PROMPT_NAME,
            )
        except Exception:  # noqa: BLE001
            logger.warning("translate_query failed; using original")
            return query

    async def translate_answer(self, text: str, *, target_lang: Language) -> str:
        """Translate a full LLM answer, preserving markdown and citation markers."""
        try:
            return await self._call(text, target_lang=target_lang, context="answer")
        except Exception:  # noqa: BLE001
            logger.warning("translate_answer failed; using original")
            return text

    async def translate_term(self, term: str, *, target_lang: Language) -> str:
        """Translate a single medical term."""
        try:
            return await self._call(term, target_lang=target_lang, context="term")
        except Exception:  # noqa: BLE001
            logger.warning("translate_term failed; using original")
            return term

    async def translate(self, text: str, *, target_lang: Language) -> str:
        """General-purpose translation to target_lang."""
        try:
            return await self._call(text, target_lang=target_lang, context="general")
        except Exception:  # noqa: BLE001
            logger.warning("translate failed; using original")
            return text

    async def _call(
        self,
        text: str,
        *,
        target_lang: Language,
        context: str = "general",
        prompt_name: str = _PROMPT_NAME,
    ) -> str:
        """Shared LLM translation call. Raises on failure — callers handle fallback."""
        from pydantic_ai import Agent

        from claritymed.core.observability.steps import step
        from claritymed.core.prompts.registry import PromptRegistry

        system_prompt = PromptRegistry().get(prompt_name, language=target_lang)
        agent: Agent[None, str] = Agent(
            self._model,
            system_prompt=system_prompt,
            output_type=str,
        )
        with step(f"translate.{context}", details=f"translate/{target_lang}") as s:
            result = await agent.run(text)
            translated = result.output.strip()
            if translated:
                logger.debug(
                    "translation (%s → %s, %s): %r",
                    detect_language(text),
                    target_lang,
                    context,
                    translated[:120],
                )
                s.summary = "done"
                return translated
        return text
