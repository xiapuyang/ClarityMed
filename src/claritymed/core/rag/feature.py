"""RAG as a ``FeaturePlugin``.

Wraps a configured ``RagStrategy`` and surfaces the right hook for the
selected mode:

* ``deterministic`` — ``pre_invoke`` runs the full retrieval pipeline,
  accumulates chunks onto ``deps.retrieved_chunks`` (for Sources), and
  returns the formatted evidence block to splice into the prompt.
* ``tool`` — ``as_tool`` returns the standalone
  ``retrieve_medical_literature`` pydantic-ai tool callable; the LLM
  decides per-turn whether to invoke it.
* ``agentic`` — accepted at construction so config typing is stable,
  but ``build_features`` raises ``NotImplementedError`` for any active
  agentic feature in v1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from claritymed.core.features.base import FeatureMode, TurnContext
from claritymed.core.rag.retrieval_pipeline import format_evidence, perform_retrieval

if TYPE_CHECKING:
    from claritymed.core.rag.strategies.base import RagStrategy


class RagFeature:
    """RAG plugin. One instance per ask turn (cheap; holds a strategy ref)."""

    name = "rag"

    def __init__(
        self,
        mode: "FeatureMode",
        strategy: "RagStrategy | None",
    ) -> None:
        self.mode: FeatureMode = mode
        self._strategy = strategy

    async def pre_invoke(self, ctx: TurnContext) -> str:
        """Deterministic-mode hook: retrieve once, return evidence block.

        No-op outside deterministic mode and when no strategy is
        configured (``rag.enabled=false``). Tool mode handles retrieval
        via the LLM-driven tool call, not here.
        """
        if self.mode != "deterministic" or self._strategy is None:
            return ""
        # ``perform_retrieval`` reads ``deps.strategy``; AskService is
        # responsible for putting our strategy there before the turn.
        chunks = await perform_retrieval(ctx.deps, ctx.scrubbed)
        ctx.deps.retrieved_chunks.extend(chunks)
        return format_evidence(chunks)

    def as_tool(self) -> Callable | None:
        """Tool-mode hook: return the ``retrieve_medical_literature`` callable."""
        if self.mode != "tool" or self._strategy is None:
            return None
        from claritymed.core.rag.tools.retrieve_medical_literature import (
            retrieve_medical_literature,
        )

        return retrieve_medical_literature
