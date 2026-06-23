"""Per-chunk cosine-similarity dedup against an existing RAG collection.

Used by both the system-RAG ingest path (``ingest_corpus`` in
``ingest/corpus/base.py``) and the per-user upload path
(``stores/user_rag.py``) so chunk-level dedup logic lives in exactly one
place.

The query is a single dense-only KNN against the target collection. We
don't reuse the hybrid (dense+sparse+RRF) path because we only need
"is anything in the collection this close?", not a ranked list — a plain
top-1 dense query is the cheapest correct answer.

Threshold semantics: a value of ``<= 0`` short-circuits to no dedup so
both call sites get the same kill-switch. ``0.92`` – ``0.95`` are the
typical operational range (see ``upload.dedupe_cosine_threshold`` in
``app.yaml``).
"""

from __future__ import annotations

from claritymed.core.rag.chunking.base import ChildChunk
from claritymed.core.rag.embedding.base import SparseVector
from claritymed.core.rag.qdrant_store import RagCollectionStore


async def filter_near_duplicates(
    *,
    children: list[ChildChunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVector],
    store: RagCollectionStore,
    threshold: float,
) -> tuple[list[ChildChunk], list[list[float]], list[SparseVector], int]:
    """Drop children whose nearest existing chunk meets or exceeds ``threshold``.

    Args:
        children: Newly produced child chunks (pre-upsert).
        dense_vectors: Dense embeddings aligned 1:1 with ``children``.
        sparse_vectors: Sparse embeddings aligned 1:1 with ``children``.
        store: The collection to query for existing chunks.
        threshold: Cosine score gate. ``<= 0`` disables dedup entirely
            and returns the inputs unchanged (no Qdrant calls).

    Returns:
        ``(kept_children, kept_dense, kept_sparse, skipped_count)`` where
        the kept lists are aligned 1:1 and ``skipped_count`` is the number
        of children dropped as near-duplicates.

    Raises:
        ValueError: If the three input lists are not the same length.
    """
    if not (len(children) == len(dense_vectors) == len(sparse_vectors)):
        raise ValueError(
            "children / dense_vectors / sparse_vectors must align: "
            f"{len(children)} / {len(dense_vectors)} / {len(sparse_vectors)}"
        )
    if threshold <= 0 or not children:
        return children, dense_vectors, sparse_vectors, 0

    kept_indices: list[int] = []
    skipped = 0
    for i, vec in enumerate(dense_vectors):
        score = await store.search_dense_max_score(vec)
        if score is not None and score >= threshold:
            skipped += 1
            continue
        kept_indices.append(i)

    return (
        [children[i] for i in kept_indices],
        [dense_vectors[i] for i in kept_indices],
        [sparse_vectors[i] for i in kept_indices],
        skipped,
    )
