"""Translation provider interface and shared language utilities.

``detect_language`` is a pure CJK-ratio heuristic with no dependencies — it
can be called anywhere without constructing a provider.  ``TranslationProvider``
is the ABC that every backend implements.  Current implementations:

* ``LLMTranslationProvider`` — pydantic-ai Agent call; works out of the box
  with any model in the provider catalog.

Planned (not yet implemented):
* ``BgeM3TranslationProvider`` — multilingual instruction embedding; no LLM
  call, uses BGE-M3's cross-lingual capability directly.
* ``DeepLTranslationProvider`` / ``GoogleTranslationProvider`` — cloud
  translation APIs; requires ``api_key_env`` in config.

Callers depend only on ``TranslationProvider``; the concrete class is selected
by ``make_translation_provider`` via ``retrieval.yaml: translation.provider``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

Language = Literal["en", "zh"]


def detect_language(text: str) -> Language:
    """Return ``'zh'`` if >20 % of characters are CJK, else ``'en'``.

    Handles empty strings, ASCII-only text, and mixed-script inputs.
    Called as a standalone utility — no provider instance needed.
    """
    if not text:
        return "en"
    cjk = sum(1 for c in text if "一" <= c <= "鿿")
    return "zh" if cjk / len(text) > 0.2 else "en"


class TranslationProvider(ABC):
    """Interface for all translation backends.

    Every method falls back to the original text on any error so callers are
    never blocked by a translation failure.  ``detect_language`` is a module-level
    function, not a method, because it is provider-independent.
    """

    @abstractmethod
    async def translate_query(self, query: str, *, target_lang: Language) -> str:
        """Translate a retrieval query for embedding alignment.

        Short input — the implementation should output only the bare translation
        with no explanation, per the system prompt.
        """

    @abstractmethod
    async def translate_answer(self, text: str, *, target_lang: Language) -> str:
        """Translate a full LLM answer.

        Implementations must preserve markdown structure, citation markers ([N]),
        section headings, and medical term accuracy.
        """

    @abstractmethod
    async def translate_term(self, term: str, *, target_lang: Language) -> str:
        """Translate a single medical term.

        Short input — output only the bare translation.
        """

    @abstractmethod
    async def translate(self, text: str, *, target_lang: Language) -> str:
        """General-purpose translation to target_lang."""
