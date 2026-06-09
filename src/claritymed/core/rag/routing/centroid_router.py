"""Embedding-based collection router.

Picks active collections by cosine similarity between the query's dense
vector (from the running BGE-M3 server) and each collection's centroid
(precomputed at ingest time and persisted in ``data/shared/centroids/``).

Why this exists alongside the rule-based ``CollectionRouter``:

* Topic-overlap keyword matching is brittle for cross-lingual queries
  and growing catalogs — a new collection means another set of topic
  phrases to maintain by hand.
* Centroids capture the *semantic* footprint of a collection from its
  own embeddings, so routing scales with the corpus, not the catalog
  metadata.

Fallback to the rule-based router is per-collection: if a collection has
no centroid file yet, the rule-based decision for that collection is
adopted verbatim. This keeps rollout safe — a partially populated
``centroids/`` directory degrades gracefully rather than misrouting.

The Router protocol is async because this router awaits ``embed_dense``
at selection time; ``CollectionRouter`` adopted the async surface for
parity.
"""

from __future__ import annotations

import logging
import math

from claritymed.core.rag.embedding.base import Embedder
from claritymed.core.rag.routing.centroid_store import CentroidStore
from claritymed.core.rag.routing.collection_router import (
    CollectionRouter,
    QueryLanguage,
    Router,
    RouterTrace,
    RoutingDecision,
)
from claritymed.core.rag.schemas import CollectionMetadata, RouterEntry

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity over two equal-length float vectors.

    Returns 0.0 for degenerate inputs (mismatched dimensions, zero
    vector) — those should never flow through the centroid path in
    practice, but a panic here would surface as an obscure 500 in the
    retriever, so we degrade quietly and let the caller see the score.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class CentroidRouter(Router):
    """Cosine-similarity router with per-collection rule-based fallback."""

    def __init__(
        self,
        *,
        catalog: list[CollectionMetadata],
        config: RouterEntry,
        centroid_store: CentroidStore,
        embedder: Embedder,
        fallback: CollectionRouter,
        default_whitelist: list[str] | None = None,
    ) -> None:
        self._catalog = {c.name: c for c in catalog}
        self._max_active = config.max_active
        self._min_similarity = config.min_similarity
        self._centroid_store = centroid_store
        self._embedder = embedder
        self._fallback = fallback
        self._default_whitelist = (
            list(default_whitelist) if default_whitelist is not None else None
        )
        # Loaded on demand — avoids hitting disk on every query for a
        # vector that doesn't change between ingests.
        self._centroid_cache: dict[str, list[float] | None] = {}

    # --- Router protocol -----------------------------------------------

    async def select(
        self,
        query: str,
        language: QueryLanguage,
        user_whitelist: list[str] | None,
    ) -> list[str]:
        trace = await self.select_with_trace(query, language, user_whitelist)
        return trace.selected

    async def select_with_trace(
        self,
        query: str,
        language: QueryLanguage,
        user_whitelist: list[str] | None,
    ) -> RouterTrace:
        if user_whitelist == []:
            # Same opt-out semantics as CollectionRouter — the router
            # never overrides an explicit "no system RAG, please".
            return RouterTrace(
                selected=[],
                considered=[
                    RoutingDecision(
                        name=name,
                        selected=False,
                        reason="user_whitelist=[] (opted out)",
                    )
                    for name in self._catalog
                ],
            )

        centroid_vecs = {name: self._lookup(name) for name in self._catalog}
        has_any_centroid = any(v is not None for v in centroid_vecs.values())
        if not has_any_centroid:
            return await self._fallback.select_with_trace(
                query, language, user_whitelist
            )

        qvec_list = await self._embedder.embed_dense([query])
        if not qvec_list:
            logger.warning(
                "centroid router: embedder returned no vectors; "
                "falling back to rule-based for the whole catalog"
            )
            return await self._fallback.select_with_trace(
                query, language, user_whitelist
            )
        qvec = qvec_list[0]

        # Pull fallback decisions only when we actually need them (some
        # collection has no centroid) — saves one extra trace build on
        # the common path.
        no_centroid_names = [n for n, v in centroid_vecs.items() if v is None]
        fallback_decisions: dict[str, RoutingDecision] = {}
        if no_centroid_names:
            fallback_trace = await self._fallback.select_with_trace(
                query, language, user_whitelist
            )
            fallback_decisions = {d.name: d for d in fallback_trace.considered}

        whitelist = self._resolve_whitelist(user_whitelist)
        scored: list[tuple[str, float]] = []
        considered: list[RoutingDecision] = []

        for name, meta in self._catalog.items():
            centroid = centroid_vecs[name]
            if centroid is None:
                fdec = fallback_decisions.get(name)
                considered.append(
                    RoutingDecision(
                        name=name,
                        selected=fdec.selected if fdec else False,
                        reason=(
                            f"centroid_absent (rule_based_fallback: {fdec.reason})"
                            if fdec
                            else "centroid_absent (no fallback decision)"
                        ),
                        topic_score=fdec.topic_score if fdec else 0.0,
                    )
                )
                continue

            if name not in whitelist:
                considered.append(
                    RoutingDecision(
                        name=name,
                        selected=False,
                        reason="not in whitelist",
                    )
                )
                continue
            if not (meta.language == language or meta.cross_lingual):
                considered.append(
                    RoutingDecision(
                        name=name,
                        selected=False,
                        reason=(
                            f"language {language} does not match "
                            f"{meta.language} (cross_lingual={meta.cross_lingual})"
                        ),
                    )
                )
                continue

            score = _cosine_similarity(qvec, centroid)
            if score < self._min_similarity:
                considered.append(
                    RoutingDecision(
                        name=name,
                        selected=False,
                        reason=(
                            f"centroid_score {score:.3f} < "
                            f"min_similarity {self._min_similarity:.3f}"
                        ),
                        topic_score=score,
                    )
                )
                continue
            scored.append((name, score))

        scored.sort(key=lambda t: -t[1])
        selected_via_centroid = [n for n, _ in scored[: self._max_active]]
        selected_set = set(selected_via_centroid)

        for name, score in scored:
            kept = name in selected_set
            considered.append(
                RoutingDecision(
                    name=name,
                    selected=kept,
                    reason=(
                        f"centroid_score={score:.3f}"
                        if kept
                        else f"capped by max_active={self._max_active}"
                    ),
                    topic_score=score,
                )
            )

        # Final selected list: centroid winners + fallback's winners for
        # any collections that didn't have a centroid available.
        selected: list[str] = list(selected_via_centroid)
        for name in no_centroid_names:
            fdec = fallback_decisions.get(name)
            if fdec and fdec.selected:
                selected.append(name)

        return RouterTrace(selected=selected, considered=considered)

    # --- internals ------------------------------------------------------

    def _lookup(self, collection: str) -> list[float] | None:
        if collection not in self._centroid_cache:
            self._centroid_cache[collection] = self._centroid_store.vector(collection)
        return self._centroid_cache[collection]

    def _resolve_whitelist(self, user_whitelist: list[str] | None) -> set[str]:
        if user_whitelist is not None:
            return set(user_whitelist)
        if self._default_whitelist:
            return set(self._default_whitelist)
        return set(self._catalog)
