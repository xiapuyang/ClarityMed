"""Per-user PHI Qdrant store — records' OCR text embeds here.

Structurally distinct from ``UserRagStore`` (library): PHI records ingest
through ``save_record`` (Unit 6) and land in a separate Qdrant
collection ``user_phi_<user_id>``. This is layer 1 of the 3-layer PHI
defense: a buggy filter cannot leak PHI to the library collection
because the bytes never live there in the first place.

Important differences from ``UserRagStore``:

* **No PHI scrub on the write path.** OCR text from a checkup contains
  real lab values that the user wants to retrieve verbatim ("what was
  my HGB last month?"); scrubbing them would defeat the purpose.
  Defense at retrieval time — ``filter_chunks_for_provider(kind=cloud)``
  drops these chunks, and the layer-3 ``PhiAssertionModel`` blocks any
  that slip through.
* **Payload flags fixed.** ``is_phi=True`` and ``can_cloud=False`` are
  the only values this store ever writes. ``public`` is not a valid
  argument; a record is PHI by construction.
* **Keyed by ``record_path``, not arbitrary ``doc_id``.** The manifest
  directory's relative path inside ``data/users/<id>/records/`` is the
  stable identifier — same string as the row ``LibraryView`` shows.
  Delete cascade uses this key (``delete_by_doc_id(record_path)``).

``UserPhiRagStore`` shares the embedder, chunker, and the single
per-process ``AsyncQdrantClient`` with ``UserRagStore``; both live in
one process but write to separate collections.
"""

from __future__ import annotations

import logging

from qdrant_client import AsyncQdrantClient

from claritymed.core.rag.chunking.base import Chunker, RawDocument
from claritymed.core.rag.dedup import filter_near_duplicates
from claritymed.core.rag.embedding.base import Embedder
from claritymed.core.rag.parent_store import ParentStore
from claritymed.core.rag.qdrant_store import RagCollectionStore
from claritymed.stores.paths import user_parent_docstore_phi_path

logger = logging.getLogger(__name__)


def collection_name(user_id: str) -> str:
    """``user_phi_<user_id>`` — paired with ``user_rag_<user_id>`` (library)."""
    return f"user_phi_{user_id}"


class UserPhiRagStore:
    """Ingest + delete facade for the PHI side of per-user RAG.

    Constructed with the same primitives as ``UserRagStore`` but writes
    to a different collection. Retrieval still goes through
    ``HybridRetriever`` (Unit 4 wires it to query both collections).
    """

    def __init__(
        self,
        aclient: AsyncQdrantClient,
        embedder: Embedder,
        chunker: Chunker,
    ) -> None:
        self._aclient = aclient
        self._embedder = embedder
        self._chunker = chunker

    # --- factory caches (per user_id) ----------------------------------

    def _collection_store(self, user_id: str) -> RagCollectionStore:
        return RagCollectionStore(
            aclient=self._aclient,
            collection_name=collection_name(user_id),
            dense_dim=self._embedder.dimension,
        )

    def _parent_store(self, user_id: str) -> ParentStore:
        return ParentStore(user_parent_docstore_phi_path(user_id))

    # --- write ---------------------------------------------------------

    async def add_record(
        self,
        user_id: str,
        record_path: str,
        ocr_text: str,
        *,
        metadata: dict | None = None,
        dedupe_threshold: float | None = None,
    ) -> int:
        """Embed raw OCR text into the PHI collection. Returns chunk count.

        ``record_path`` is the manifest directory's path relative to
        ``data/users/<id>/records/`` — e.g. ``exam-reports/2026-06-11-ab12cd34``.
        Used as both the LlamaIndex ``doc_id`` and the Qdrant
        ``payload.doc_id`` so delete cascade has one stable key.

        Empty / whitespace-only OCR text writes nothing and returns 0
        (e.g. the worker's ``ocr_status="empty"`` outcome).

        ``dedupe_threshold`` is the third dedup layer (record-level via
        manifest slug + attachment-level via BlobStore CAS are the first
        two). ``None`` reads ``record_dedupe_cosine_threshold()`` from
        config; ``<= 0`` disables dedup entirely (same kill-switch as
        the library path).
        """
        if not ocr_text or not ocr_text.strip():
            return 0

        # Tag every chunk with the source record so the LibraryView can
        # surface "this record contains N chunks" without a separate
        # join table.
        meta = dict(metadata or {})
        meta.setdefault("record_path", record_path)

        doc = RawDocument(doc_id=record_path, text=ocr_text, metadata=meta)
        chunked = self._chunker.chunk(doc)
        if not chunked.children:
            return 0

        # Embed BEFORE the parent write so a dedupe pass that drops every
        # child can also skip the parent persist — no orphan ParentStore
        # entries (mirrors user_rag.py:164-198).
        child_texts = [c.text for c in chunked.children]
        dense_vecs = await self._embedder.embed_dense(child_texts)
        sparse_vecs = await self._embedder.embed_sparse(child_texts)

        col_store = self._collection_store(user_id)

        from claritymed import config as _cfg

        threshold = (
            dedupe_threshold
            if dedupe_threshold is not None
            else _cfg.record_dedupe_cosine_threshold()
        )

        (
            kept_children,
            kept_dense,
            kept_sparse,
            skipped_chunks,
        ) = await filter_near_duplicates(
            children=chunked.children,
            dense_vectors=dense_vecs,
            sparse_vectors=sparse_vecs,
            store=col_store,
            threshold=threshold,
        )

        if skipped_chunks:
            logger.debug(
                "user_phi_rag: dropped %d/%d chunks as near-duplicates "
                "(record_path=%s, threshold=%.3f)",
                skipped_chunks,
                len(chunked.children),
                record_path,
                threshold,
            )

        if not kept_children:
            return 0

        parent_store = self._parent_store(user_id)
        parent_store.bulk_put(chunked.parents)
        parent_store.persist()

        return await col_store.upsert(
            children=kept_children,
            dense_vectors=kept_dense,
            sparse_vectors=kept_sparse,
            is_phi=True,
            can_cloud=False,
        )

    # --- delete --------------------------------------------------------

    async def delete_by_doc_id(self, user_id: str, record_path: str) -> None:
        """Cascade-delete every chunk for the given ``record_path``.

        Caller (``delete_record`` tool) is responsible for the
        Qdrant-first / manifest-second ordering — see Unit 6.
        """
        col_store = self._collection_store(user_id)
        await col_store.delete_by_doc_id(record_path)


def make_phi_rag_store(user_id: str) -> "UserPhiRagStore":
    """Build a ``UserPhiRagStore`` for one user (local-mode file storage).

    Shares ``open_local_qdrant_client``'s process-level cache with
    ``make_user_rag_store``, so both library and PHI collections use the
    same ``AsyncQdrantClient`` instance for the same user directory — no
    second file-lock conflict when the retriever and a post-tool embed
    task run concurrently within one session.
    """
    from claritymed.core.rag.chunking.factory import build_chunker
    from claritymed.core.rag.embedding.factory import build_embedder
    from claritymed.core.rag.qdrant_store import open_local_qdrant_client
    from claritymed.stores.paths import user_rag_qdrant_dir

    user_dir = user_rag_qdrant_dir(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    aclient = open_local_qdrant_client(user_dir)
    return UserPhiRagStore(
        aclient=aclient,
        embedder=build_embedder(),
        chunker=build_chunker(),
    )
