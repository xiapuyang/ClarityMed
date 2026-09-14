"""Reranker protocol.

A reranker takes ``(query, docs)`` and returns the top-``k`` doc indices
with cross-encoder scores. Indices reference positions in the input ``docs``
list, not chunk ids — the caller maps back to the original chunks.

Output is sorted descending by score. When ``len(docs) < top_k`` the
reranker returns at most ``len(docs)`` hits.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable


class RerankHit(NamedTuple):
    """One rerank result: input index + cross-encoder score."""

    index: int
    score: float


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder rerank contract."""

    async def rerank(
        self,
        query: str,
        docs: list[str],
        top_k: int,
    ) -> list[RerankHit]:
        """Score and re-rank ``docs`` against ``query``."""
        ...
