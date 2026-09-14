"""CorpusSource protocol + shared ingest runner.

A corpus source is *just* an iterator of ``RawDocument`` plus a name.
``ingest_corpus`` is the shared runner that chunks → embeds → upserts
into Qdrant + parent docstore; every new source (PubMed, DailyMed,
MedlinePlus, DiaKG, …) plugs into the same pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from claritymed.core.rag.chunking.base import Chunker, RawDocument
from claritymed.core.rag.dedup import filter_near_duplicates
from claritymed.core.rag.embedding.base import Embedder
from claritymed.core.rag.parent_store import ParentStore
from claritymed.core.rag.qdrant_store import RagCollectionStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestStats:
    """Per-run summary returned by ``ingest_corpus``."""

    source: str
    docs_processed: int
    parents_written: int
    children_written: int
    docs_skipped: int
    docs_resumed: int = 0
    """Docs whose ``doc_id`` already had children in the store and were
    skipped to support cheap resume after an interrupted run."""
    children_deduped: int = 0
    """Children dropped by the cosine-similarity dedup pass (only
    populated when ``ingest_corpus(..., dedupe_threshold=...)`` is set)."""


@runtime_checkable
class CorpusSource(Protocol):
    """One system corpus. Adapters live under ``ingest/corpus/``."""

    name: str
    """The Qdrant collection name (must match ``retrieval.yaml`` entry)."""

    def iter_raw_docs(self) -> Iterator[RawDocument]:
        """Yield documents in any order; the runner handles persistence."""
        ...


async def ingest_corpus(
    source: CorpusSource,
    *,
    chunker: Chunker,
    embedder: Embedder,
    store: RagCollectionStore,
    parent_store: ParentStore,
    limit: int | None = None,
    dry_run: bool = False,
    dedupe_threshold: float | None = None,
    on_doc: Callable[[IngestStats], None] | None = None,
) -> IngestStats:
    """Chunk + embed + upsert every doc from ``source``.

    Caller wires the chunker + embedder + store + parent_store (typically
    via factories pointed at the active config). ``limit`` caps the
    number of documents (smoke tests, partial reingests). ``dry_run``
    skips the embed + qdrant write — useful for verifying parsing alone.

    ``dedupe_threshold`` enables per-chunk cosine-similarity dedup
    against the target collection (the same primitive ``stores/user_rag``
    uses): a child whose nearest existing chunk scores at or above the
    threshold is dropped before upsert. ``None`` (default) or ``<= 0``
    disables the layer — no extra Qdrant calls in the disabled case.
    When every child of a doc gets dropped, that doc's parents are also
    skipped so the parent docstore never holds orphan rows.

    ``on_doc`` is fired after every successful doc with the running
    ``IngestStats`` — the CLI uses it to drive a rich.progress bar
    without coupling this core module to any UI library.

    Returns counts for the CLI progress + post-run audit.
    """
    docs_processed = 0
    docs_skipped = 0
    docs_resumed = 0
    parents_written = 0
    children_written = 0
    children_deduped = 0
    threshold = dedupe_threshold or 0.0

    # Pre-load existing doc_ids into an in-memory set so the per-doc
    # resume check is O(1). One full scroll up front is faster than
    # ~10k per-doc round-trips even with the payload index on doc_id,
    # since the bottleneck is network latency not Qdrant work.
    if not dry_run:
        await store.ensure_collection()
        existing_doc_ids: set[str] = await store.list_doc_ids()
    else:
        existing_doc_ids = set()

    def _snapshot() -> IngestStats:
        return IngestStats(
            source=source.name,
            docs_processed=docs_processed,
            parents_written=parents_written,
            children_written=children_written,
            docs_skipped=docs_skipped,
            docs_resumed=docs_resumed,
            children_deduped=children_deduped,
        )

    for doc in _take(source.iter_raw_docs(), limit):
        # Resume probe: if this doc_id already has children in the store,
        # skip the expensive chunk + embed + upsert cycle. Gate on
        # ``not dry_run`` so dry-run stays a pure parse smoke test that
        # does not require a live Qdrant.
        if not dry_run and doc.doc_id in existing_doc_ids:
            docs_resumed += 1
            if on_doc is not None:
                on_doc(_snapshot())
            continue
        try:
            chunked = chunker.chunk(doc)
        except Exception:  # noqa: BLE001 — keep the run alive
            logger.exception("chunker failed on doc_id=%s", doc.doc_id)
            docs_skipped += 1
            continue
        if not chunked.children:
            docs_skipped += 1
            continue
        docs_processed += 1

        if dry_run:
            parents_written += len(chunked.parents)
            children_written += len(chunked.children)
            continue

        texts = [c.text for c in chunked.children]
        dense_vecs = await embedder.embed_dense(texts)
        sparse_vecs = await embedder.embed_sparse(texts)

        kept_children, kept_dense, kept_sparse, deduped = await filter_near_duplicates(
            children=chunked.children,
            dense_vectors=dense_vecs,
            sparse_vectors=sparse_vecs,
            store=store,
            threshold=threshold,
        )
        children_deduped += deduped

        if not kept_children:
            # Every child was a near-duplicate — skip parents + upsert
            # so the parent docstore never accumulates orphan rows.
            if on_doc is not None:
                on_doc(_snapshot())
            continue

        parent_store.bulk_put(chunked.parents)
        parents_written += len(chunked.parents)

        written = await store.upsert(
            children=kept_children,
            dense_vectors=kept_dense,
            sparse_vectors=kept_sparse,
            is_phi=False,
            can_cloud=True,
        )
        children_written += written
        existing_doc_ids.add(doc.doc_id)

        if on_doc is not None:
            on_doc(_snapshot())

    if not dry_run and parents_written:
        parent_store.persist()

    return _snapshot()


def _take(it: Iterable[RawDocument], limit: int | None) -> Iterator[RawDocument]:
    if limit is None:
        yield from it
        return
    for i, item in enumerate(it):
        if i >= limit:
            return
        yield item
