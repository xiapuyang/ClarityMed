"""NaiveHybridStrategy + CRAG-lite grader hook.

One-shot retrieval via ``HybridRetriever``. When ``grader.enabled=True``:

1. Compute ``mean(top_n.rerank_score)`` over the surviving chunks.
2. If below ``threshold`` → produce a rewritten query and re-retrieve.
3. Merge first-pass and second-pass chunks by score, dedup by child id,
   keep the best ``max_evidence``.
4. Update the bundle's ``trace.grader`` with the decision + score.

This is **not** Agentic RAG — there is no LLM reflection loop. The
grader is a deterministic threshold; the rewrite is deterministic too
(keyword extraction by default; ``rewrite_mode="llm"`` raises
NotImplementedError in v1).
"""

from __future__ import annotations

import logging
import re

from claritymed.core.rag.retriever import HybridRetriever
from claritymed.core.rag.schemas import (
    EvidenceBundle,
    GraderConfig,
    GraderReport,
    RetrievalTrace,
)
from claritymed.core.rag.strategies.base import RagStrategy, RetrievalContext
from claritymed.core.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# Conservative English stopwords for the deterministic rewrite path. CJK
# queries skip this — substring matching on the Qdrant sparse side does
# most of the work and stopwords like "怎么样" are rare relative to total
# token volume.
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "should",
    "so",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}

_EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


class NaiveHybridStrategy(RagStrategy):
    """Single-shot hybrid retrieval, optional CRAG-lite fallback."""

    def __init__(
        self,
        retriever: HybridRetriever,
        grader: GraderConfig | None = None,
        max_evidence: int = 5,
    ) -> None:
        self._retriever = retriever
        self._grader = grader or GraderConfig()
        self._max_evidence = max_evidence

    async def retrieve(self, ctx: RetrievalContext) -> EvidenceBundle:
        first = await self._retrieve_one(ctx, query=ctx.query)
        if not self._grader.enabled:
            return self._cap(first)

        mean_score = self._mean_rerank(first.chunks, top_n=self._grader.top_n)
        if mean_score >= self._grader.threshold:
            return self._cap(
                self._with_grader(
                    first,
                    GraderReport(
                        threshold=self._grader.threshold,
                        mean_rerank_score=mean_score,
                        decision="pass",
                    ),
                    fallback=False,
                )
            )

        rewritten = self._rewrite(ctx.query, ctx.language)
        if rewritten == ctx.query:
            # Rewrite collapsed to the same string (e.g. all words were
            # already keywords). Skip the second pass; grader still records
            # the "would have rewritten" decision so the trace is honest.
            return self._cap(
                self._with_grader(
                    first,
                    GraderReport(
                        threshold=self._grader.threshold,
                        mean_rerank_score=mean_score,
                        decision="rewrite",
                    ),
                    fallback=True,
                )
            )

        second = await self._retrieve_one(ctx, query=rewritten)
        merged_chunks = self._merge_chunks(first.chunks, second.chunks)
        merged_trace = self._merge_trace(
            first.trace,
            second.trace,
            grader=GraderReport(
                threshold=self._grader.threshold,
                mean_rerank_score=mean_score,
                decision="rewrite",
            ),
            fallback=True,
        )
        return self._cap(EvidenceBundle(chunks=merged_chunks, trace=merged_trace))

    # --- internals -----------------------------------------------------

    async def _retrieve_one(
        self, ctx: RetrievalContext, *, query: str
    ) -> EvidenceBundle:
        return await self._retriever.retrieve(
            query,
            language=ctx.language,
            user_id=ctx.user_id,
            user_whitelist=ctx.user_whitelist,
            only_cloud_safe=ctx.only_cloud_safe,
        )

    @staticmethod
    def _mean_rerank(chunks: list[RetrievedChunk], *, top_n: int) -> float:
        scored = [c.rerank_score for c in chunks[:top_n] if c.rerank_score is not None]
        if not scored:
            return 0.0
        return sum(scored) / len(scored)

    def _rewrite(self, query: str, language: str) -> str:
        """Deterministic query rewrite. LLM mode raises NotImplementedError."""
        if self._grader.rewrite_mode == "llm":
            raise NotImplementedError(
                "NaiveHybridStrategy: rewrite_mode='llm' is reserved for a "
                "future LLM-backed grader; v1 ships deterministic only"
            )
        if language == "zh":
            # CJK: no useful stopword removal; squeeze whitespace only.
            return re.sub(r"\s+", " ", query).strip()
        tokens = _EN_TOKEN_RE.findall(query)
        keywords = [t for t in tokens if t.lower() not in _STOPWORDS]
        if not keywords:
            return query
        return " ".join(keywords)

    @staticmethod
    def _merge_chunks(
        a: list[RetrievedChunk], b: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Dedup by (collection, doc_id, chunk_index) keeping the higher rerank score."""
        by_key: dict[tuple, RetrievedChunk] = {}
        for chunk in (*a, *b):
            key = (chunk.collection_name, chunk.doc_id, chunk.chunk_index)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = chunk
                continue
            if (chunk.rerank_score or 0.0) > (existing.rerank_score or 0.0):
                by_key[key] = chunk
        merged = list(by_key.values())
        merged.sort(
            key=lambda c: (c.rerank_score or c.score or 0.0),
            reverse=True,
        )
        return merged

    @staticmethod
    def _merge_trace(
        a: RetrievalTrace,
        b: RetrievalTrace,
        *,
        grader: GraderReport,
        fallback: bool,
    ) -> RetrievalTrace:
        active = list(dict.fromkeys([*a.active_collections, *b.active_collections]))
        return RetrievalTrace(
            strategy="naive_hybrid",
            active_collections=active,
            expanded_query=b.expanded_query or a.expanded_query,
            embed_ms=a.embed_ms + b.embed_ms,
            search_ms=a.search_ms + b.search_ms,
            rerank_ms=a.rerank_ms + b.rerank_ms,
            parent_expand_ms=a.parent_expand_ms + b.parent_expand_ms,
            grader=grader,
            fallback_triggered=fallback,
            rerank_fallback=a.rerank_fallback or b.rerank_fallback,
        )

    @staticmethod
    def _with_grader(
        bundle: EvidenceBundle,
        grader: GraderReport,
        *,
        fallback: bool,
    ) -> EvidenceBundle:
        t = bundle.trace
        new_trace = RetrievalTrace(
            strategy=t.strategy,
            active_collections=t.active_collections,
            expanded_query=t.expanded_query,
            embed_ms=t.embed_ms,
            search_ms=t.search_ms,
            rerank_ms=t.rerank_ms,
            parent_expand_ms=t.parent_expand_ms,
            grader=grader,
            fallback_triggered=fallback,
            rerank_fallback=t.rerank_fallback,
        )
        return EvidenceBundle(chunks=bundle.chunks, trace=new_trace)

    def _cap(self, bundle: EvidenceBundle) -> EvidenceBundle:
        if len(bundle.chunks) <= self._max_evidence:
            return bundle
        return EvidenceBundle(
            chunks=bundle.chunks[: self._max_evidence],
            trace=bundle.trace,
        )
