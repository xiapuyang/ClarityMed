"""Medical translation service backed by the configured LLM.

Covers four use cases:

* ``translate_query``  — short embedding queries; output is the bare translation.
* ``translate_answer`` — full LLM responses; preserves markdown, citations, structure.
* ``translate_term``   — single medical terms; output is the bare translation.
* ``translate``        — general-purpose fallback.

``detect_language`` is a pure heuristic (no LLM call) and can be called as a
static method anywhere in the codebase without constructing an instance.

All async methods fall back to the original text on any error so callers are
never blocked by a translation failure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)

Language = Literal["en", "zh"]

# Registry key for the translation prompt.  The language argument to
# ``PromptRegistry.get`` is the TARGET language, not the source.
_PROMPT_NAME = "translate"


class TranslationService:
    """LLM-backed translation with a static language detector."""

    def __init__(self, model: "Model") -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Language detection (no LLM — pure heuristic)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_language(text: str) -> Language:
        """Return 'zh' if >20 % of characters are CJK, else 'en'.

        Handles empty strings, ASCII-only text, and mixed-script inputs.
        """
        if not text:
            return "en"
        cjk = sum(1 for c in text if "一" <= c <= "鿿")
        return "zh" if cjk / len(text) > 0.2 else "en"

    # ------------------------------------------------------------------
    # Public translation methods
    # ------------------------------------------------------------------

    async def translate_query(self, query: str, *, target_lang: Language) -> str:
        """Translate a retrieval query for embedding alignment.

        Short input — the model outputs only the bare translation with no
        explanation, per the system prompt.
        """
        try:
            return await self._call(query, target_lang=target_lang, context="query")
        except Exception:  # noqa: BLE001
            logger.warning("translate_query failed; using original")
            return query

    async def translate_answer(self, text: str, *, target_lang: Language) -> str:
        """Translate a full LLM answer.

        The system prompt instructs the model to preserve markdown structure,
        citation markers ([N]), and medical term accuracy.
        """
        try:
            return await self._call(text, target_lang=target_lang, context="answer")
        except Exception:  # noqa: BLE001
            logger.warning("translate_answer failed; using original")
            return text

    async def translate_term(self, term: str, *, target_lang: Language) -> str:
        """Translate a single medical term.

        Short input — the model outputs only the bare translation.
        """
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _call(
        self,
        text: str,
        *,
        target_lang: Language,
        context: str = "general",
    ) -> str:
        """Shared LLM translation call. Raises on failure — callers handle fallback."""
        from pydantic_ai import Agent

        from claritymed.core.prompts.registry import PromptRegistry

        system_prompt = PromptRegistry().get(_PROMPT_NAME, language=target_lang)
        agent: Agent[None, str] = Agent(
            self._model,
            system_prompt=system_prompt,
            output_type=str,
        )
        result = await agent.run(text)
        translated = result.output.strip()
        if translated:
            logger.debug(
                "translation (%s → %s, %s): %r",
                self.detect_language(text),
                target_lang,
                context,
                translated[:120],
            )
            return translated
        return text
