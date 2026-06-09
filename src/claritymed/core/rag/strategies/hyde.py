"""HyDE — Hypothetical Document Embeddings strategy.

Implements the pattern from
`Precise Zero-Shot Dense Retrieval without Relevance Labels`
(Gao et al., 2022): an LLM drafts a hypothetical passage that would
answer the user's question; that passage is embedded and used as the
retrieval query, instead of (or alongside) the bare question.

Why a custom implementation rather than ``llama_index.core.indices.query
.query_transform.HyDEQueryTransform``: the LlamaIndex class wants a
``llama_index.core.llms.LLM`` instance. Bridging pydantic-ai → llama-
index would mean implementing the LlamaIndex LLM contract just to host
one ``predict`` call. The pattern itself is ~30 lines of code and reuses
the project's existing ``PromptRegistry`` + pydantic-ai ``Agent`` path,
so we stay on one LLM surface and one prompt loader.

Fail-soft: when the LLM call fails (network, timeout, content filter)
the strategy embeds the original query and sets
``RetrievalTrace.hyde_fallback=True`` so the failure is auditable
without breaking the retrieval pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from claritymed.core.rag.retriever import HybridRetriever
from claritymed.core.rag.schemas import (
    EvidenceBundle,
    HydeStrategyConfig,
    RetrievalTrace,
)
from claritymed.core.rag.strategies.base import RagStrategy, RetrievalContext

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)

_PROMPT_NAME = "hyde"

Hypothesizer = Callable[[str, str], Awaitable[str]]


class HydeStrategy(RagStrategy):
    """Generate a hypothetical passage, embed it, and retrieve.

    ``hypothesizer`` is the seam tests inject through — pass any
    ``async (query, language) -> str`` callable. When unset, the default
    path builds a pydantic-ai Agent against ``model`` using the ``hyde``
    prompt from the registry.
    """

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        config: HydeStrategyConfig,
        model: "Model | None" = None,
        max_evidence: int = 5,
        hypothesizer: Hypothesizer | None = None,
    ) -> None:
        if hypothesizer is None and model is None:
            raise ValueError(
                "HydeStrategy needs either a hypothesizer callable or a Model "
                "to run the default pydantic-ai Agent path"
            )
        self._retriever = retriever
        self._config = config
        self._model = model
        self._max_evidence = max_evidence
        self._hypothesizer = hypothesizer

    async def retrieve(self, ctx: RetrievalContext) -> EvidenceBundle:
        embedding_query, hyde_fallback = await self._build_embedding_query(
            ctx.query, ctx.language
        )
        # Rerank still uses the original query (Decision 7 in the origin
        # RAG plan): expansion is for recall, rerank scores against the
        # user's actual question to keep relevance honest.
        bundle = await self._retriever.retrieve(
            ctx.query,
            language=ctx.language,
            user_id=ctx.user_id,
            user_whitelist=ctx.user_whitelist,
            only_cloud_safe=ctx.only_cloud_safe,
            embedding_query_override=embedding_query,
        )
        return self._with_strategy(bundle, hyde_fallback=hyde_fallback)

    # --- internals ------------------------------------------------------

    async def _build_embedding_query(
        self, query: str, language: str
    ) -> tuple[str | None, bool]:
        """Return ``(embedding_query, hyde_fallback)``.

        ``embedding_query`` is ``None`` on LLM failure — the retriever
        then falls back to its own term-expansion path so the request
        still produces evidence.
        """
        try:
            hypothetical = await self._generate_hypothetical(query, language)
        except Exception as exc:  # noqa: BLE001 — fail-soft is the contract
            logger.warning("hyde: hypothesizer failed (%s); using original query", exc)
            return None, True
        hypothetical = hypothetical.strip()
        if not hypothetical:
            logger.warning("hyde: hypothesizer returned empty; using original query")
            return None, True
        if self._config.include_original:
            # Original query goes after the hypothetical so the dense
            # encoder weights both signals.
            return f"{hypothetical}\n\n{query}", False
        return hypothetical, False

    async def _generate_hypothetical(self, query: str, language: str) -> str:
        if self._hypothesizer is not None:
            return await self._hypothesizer(query, language)
        from pydantic_ai import Agent

        from claritymed.core.prompts.registry import PromptRegistry

        system_prompt = PromptRegistry().get(_PROMPT_NAME, language=language)
        agent: Agent[None, str] = Agent(
            self._model,
            system_prompt=system_prompt,
            output_type=str,
        )
        result = await agent.run(query)
        return result.output

    def _with_strategy(
        self, bundle: EvidenceBundle, *, hyde_fallback: bool
    ) -> EvidenceBundle:
        """Override the retriever's ``strategy`` field and stamp the hyde flag."""
        t = bundle.trace
        new_trace = RetrievalTrace(
            strategy="hyde",
            active_collections=t.active_collections,
            expanded_query=t.expanded_query,
            embed_ms=t.embed_ms,
            search_ms=t.search_ms,
            rerank_ms=t.rerank_ms,
            parent_expand_ms=t.parent_expand_ms,
            grader=t.grader,
            fallback_triggered=t.fallback_triggered,
            rerank_fallback=t.rerank_fallback,
            hyde_fallback=hyde_fallback,
        )
        chunks = bundle.chunks
        if len(chunks) > self._max_evidence:
            chunks = chunks[: self._max_evidence]
        return EvidenceBundle(chunks=chunks, trace=new_trace)
