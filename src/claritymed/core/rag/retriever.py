"""HybridRetriever — multi-collection async retrieval orchestrator.

End-to-end async path for one query:

1. ``TermService.expand_query`` rewrites the surface query with synonyms
   and cross-lingual aliases.
2. ``CollectionRouter.select`` picks active system collections.
3. ``Embedder.embed_dense`` + ``embed_sparse`` produce one (dense,
   sparse) pair from the *expanded* query.
4. For every active system collection + the user's user_rag (when it
   exists), call ``RagCollectionStore.search_hybrid`` and stitch the
   per-collection hits into a single list.
5. ``Reranker.rerank`` cross-encodes against the *original* query (not
   the expanded one — the expansion is to broaden recall, not to bias
   final relevance). Fail-soft: if the reranker is unreachable, audit
   and fall back to RRF order.
6. ``ParentStore.get_text`` hydrates the parent paragraph that each
   surviving child belongs to. Missing parents degrade gracefully —
   the child's own text is still used.
7. Return ``EvidenceBundle`` with ``RetrievalTrace`` covering router
   decisions, per-stage timings, and reranker fallback flag.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from claritymed.core.rag.embedding.base import Embedder
from claritymed.core.rag.parent_store import ParentStore
from claritymed.core.rag.qdrant_store import QdrantHit, RagCollectionStore
from claritymed.core.rag.reranking.base import Reranker
from claritymed.core.rag.routing.collection_router import Router
from claritymed.core.rag.schemas import EvidenceBundle, RetrievalTrace
from claritymed.core.rag.terms.base import TermService
from claritymed.core.rag.terms.expansion import expand_query
from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.errors import RerankerUnreachableError

logger = logging.getLogger(__name__)

QueryLanguage = Literal["en", "zh"]

# Per-collection prefetch headroom: ask for more than the final rerank
# limit so the cross-encoder has enough candidates to lift a strong
# match from a single collection above the RRF fusion order.
PER_COLLECTION_TOP_K = 10


class HybridRetriever:
    """Compose router + embedder + reranker + per-collection stores."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        reranker: Reranker,
        term_service: TermService,
        router: Router,
        system_store_factory: Callable[[str], RagCollectionStore],
        system_parent_store: ParentStore,
        user_store_factory: Callable[[str], Awaitable[RagCollectionStore | None]],
        user_parent_store_factory: Callable[[str], ParentStore | None],
        rerank_top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._reranker = reranker
        self._term_service = term_service
        self._router = router
        self._system_store = system_store_factory
        self._system_parent_store = system_parent_store
        self._user_store = user_store_factory
        self._user_parent_store_factory = user_parent_store_factory
        self._rerank_top_k = rerank_top_k

    # --- public API ----------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        language: QueryLanguage,
        user_id: str,
        user_whitelist: list[str] | None = None,
        only_cloud_safe: bool = False,
    ) -> EvidenceBundle:
        # 1. term expansion
        expanded = expand_query(query, language, self._term_service)
        # 2. routing
        router_trace = self._router.select_with_trace(query, language, user_whitelist)
        active_collections = router_trace.selected
        # 3. embedding
        t_embed = time.monotonic()
        dense_vecs = await self._embedder.embed_dense([expanded])
        sparse_vecs = await self._embedder.embed_sparse([expanded])
        embed_ms = int((time.monotonic() - t_embed) * 1000)
        if not dense_vecs or not sparse_vecs:
            return self._empty_bundle(active_collections, expanded, embed_ms)
        dense_q, sparse_q = dense_vecs[0], sparse_vecs[0]
        # 4. fan-out: system collections + user_rag
        t_search = time.monotonic()
        all_hits: list[tuple[str, QdrantHit]] = []
        for col in active_collections:
            store = self._system_store(col)
            hits = await store.search_hybrid(
                dense_q,
                sparse_q,
                PER_COLLECTION_TOP_K,
                only_cloud_safe=only_cloud_safe,
            )
            all_hits.extend((col, h) for h in hits)
        user_store = await self._user_store(user_id)
        if user_store is not None:
            user_hits = await user_store.search_hybrid(
                dense_q,
                sparse_q,
                PER_COLLECTION_TOP_K,
                only_cloud_safe=only_cloud_safe,
            )
            user_col = f"user_rag_{user_id}"
            all_hits.extend((user_col, h) for h in user_hits)
        search_ms = int((time.monotonic() - t_search) * 1000)
        # 5. rerank (fail-soft)
        t_rerank = time.monotonic()
        reranked_hits, rerank_fallback = await self._maybe_rerank(query, all_hits)
        rerank_ms = int((time.monotonic() - t_rerank) * 1000)
        # 6. parent hydrate
        t_parent = time.monotonic()
        chunks = [self._to_retrieved_chunk(col, hit) for col, hit in reranked_hits]
        for chunk in chunks:
            chunk_parent_text = self._lookup_parent(
                chunk.collection_name, chunk.parent_id, user_id
            )
            if chunk_parent_text is not None:
                # RetrievedChunk is mutable (extra="forbid", not frozen), so
                # we set parent_text directly after creation.
                chunk.parent_text = chunk_parent_text
        parent_ms = int((time.monotonic() - t_parent) * 1000)
        # 7. trace
        trace = RetrievalTrace(
            strategy="naive_hybrid",
            active_collections=active_collections,
            expanded_query=expanded if expanded != query else None,
            embed_ms=embed_ms,
            search_ms=search_ms,
            rerank_ms=rerank_ms,
            parent_expand_ms=parent_ms,
            grader=None,
            fallback_triggered=False,
            rerank_fallback=rerank_fallback,
        )
        return EvidenceBundle(chunks=chunks, trace=trace)

    # --- internals -----------------------------------------------------

    async def _maybe_rerank(
        self,
        query: str,
        hits: list[tuple[str, QdrantHit]],
    ) -> tuple[list[tuple[str, QdrantHit]], bool]:
        """Run the cross-encoder reranker over ``hits``. Fail-soft.

        Returns (selected_hits_in_rerank_order, fallback_triggered).
        Fallback semantics: if the reranker raises, keep the input order
        (RRF) and surface a flag so the caller can audit. We do not
        downgrade the result silently.
        """
        if not hits:
            return [], False
        docs = [hit.text for _, hit in hits]
        try:
            rerank_hits = await self._reranker.rerank(query, docs, self._rerank_top_k)
        except RerankerUnreachableError as exc:
            logger.warning("reranker fail-soft: %s", exc)
            return hits[: self._rerank_top_k], True
        ordered: list[tuple[str, QdrantHit]] = []
        for rh in rerank_hits:
            col, hit = hits[rh.index]
            # Stash the cross-encoder score onto the payload so callers
            # can lift it into RetrievedChunk.rerank_score.
            hit.payload["__rerank_score"] = rh.score
            ordered.append((col, hit))
        return ordered, False

    def _to_retrieved_chunk(self, collection: str, hit: QdrantHit) -> RetrievedChunk:
        payload = hit.payload
        rerank_score = payload.pop("__rerank_score", None)
        is_phi = bool(payload.get("is_phi", collection.startswith("user_rag_")))
        can_cloud = bool(
            payload.get("can_cloud", not collection.startswith("user_rag_"))
        )
        source: Literal["system_rag", "user_rag"] = (
            "user_rag" if collection.startswith("user_rag_") else "system_rag"
        )
        return RetrievedChunk(
            text=hit.text,
            source=source,
            score=hit.score,
            doc_id=str(payload.get("doc_id", "")),
            chunk_index=int(payload.get("chunk_index", 0)),
            is_phi=is_phi,
            can_cloud=can_cloud,
            user_id=None,
            source_uri=payload.get("source_uri"),
            doc_title=payload.get("doc_title"),
            ingested_at=None,
            collection_name=collection,
            parent_id=payload.get("parent_id"),
            parent_text=None,
            dense_score=None,
            sparse_score=None,
            rerank_score=rerank_score,
        )

    def _lookup_parent(
        self,
        collection: str | None,
        parent_id: str | None,
        user_id: str,
    ) -> str | None:
        if not collection or not parent_id:
            return None
        if collection.startswith("user_rag_"):
            user_store = self._user_parent_store_factory(user_id)
            return user_store.get_text(parent_id) if user_store else None
        return self._system_parent_store.get_text(parent_id)

    def _empty_bundle(
        self,
        active: list[str],
        expanded_query: str,
        embed_ms: int,
    ) -> EvidenceBundle:
        return EvidenceBundle(
            chunks=[],
            trace=RetrievalTrace(
                strategy="naive_hybrid",
                active_collections=active,
                expanded_query=expanded_query,
                embed_ms=embed_ms,
            ),
        )
