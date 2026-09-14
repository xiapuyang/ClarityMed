"""Embedder protocol.

Two output streams (``embed_dense``, ``embed_sparse``) because BGE-M3 emits
both natively and ``HybridRetriever`` weighs them together at search time.
A future ``Qwen3EmbeddingHttpEmbedder`` that only emits dense should return
an empty ``SparseVector`` for ``embed_sparse``; the retriever degrades to
dense-only ranking when sparse vectors are all empty.

Async-only by design: every production call site is async (``AskService``,
ingest workers using ``asyncio.run``). The legacy sync ``Embedder`` shim
in ``stores/user_rag.py`` is retired in Unit 8.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Qdrant's sparse-vector wire form is `{indices: [int], values: [float]}`,
# but at the protocol layer we use a `dict[int, float]` for ergonomic
# construction in Python. Convert at the Qdrant boundary, not here.
SparseVector = dict[int, float]


@runtime_checkable
class Embedder(Protocol):
    """Dense + sparse embedding contract for the RAG hybrid retriever."""

    @property
    def dimension(self) -> int:
        """Dense vector dimension (must match Qdrant collection config)."""
        ...

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` into dense vectors. Empty input → empty output."""
        ...

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Embed ``texts`` into sparse lexical vectors. Empty input → empty output."""
        ...
