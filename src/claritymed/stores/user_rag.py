"""Per-user RAG store: chunk + embed + persist user-uploaded reference text.

This module is the ingest-side facade for ``user_rag_<user_id>`` Qdrant
collections and their per-user ``ParentStore`` JSON. The retrieval side
(HybridRetriever) talks to the same underlying ``RagCollectionStore`` +
``ParentStore`` pair directly — UserRagStore just owns the writes.

Pipeline for ``add_document``:

1. Scrub the raw text through ``PhiGuard.scrub_free_text`` unless the
   caller marked the source ``public=True`` (e.g. a published paper the
   user explicitly chose to share with cloud models).
2. Chunk into (parents, children) via the configured ``Chunker`` (v1:
   parent-child via LlamaIndex HierarchicalNodeParser).
3. Embed every child with the async ``Embedder`` (dense + sparse).
4. Persist parents to the user's ``ParentStore`` JSON; persist children
   (with embeddings) to the user's ``RagCollectionStore`` Qdrant
   collection.

Per-user isolation is structural: each user has its own Qdrant collection
(``user_rag_<user_id>``) and its own ParentStore JSON
(``data/users/<id>/parent_docstore.json``). A forgotten ``user_id`` filter
cannot leak across users because there is no shared collection / file to
leak from.

Migration note: the previous fastembed BGE-small (384-dim) implementation
was retired in Unit 8 of the RAG plan. Existing collections created with
the old embedder are dimension-incompatible; use ``rag rm`` to remove stale docs
to drop + recreate (no automatic re-embed in v1 since user_rag is
expected to be empty during the alpha window).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient

from claritymed.core.rag.chunking.base import Chunker, RawDocument
from claritymed.core.rag.dedup import filter_near_duplicates
from claritymed.core.rag.embedding.base import Embedder
from claritymed.core.rag.parent_store import ParentStore
from claritymed.core.rag.qdrant_store import RagCollectionStore
from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.core.phi.guard import PhiGuard
from claritymed.stores.paths import user_parent_docstore_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of ``UserRagStore.add_document``.

    Splits "things we wrote" from "things we skipped because they
    cosine-matched existing content." Both counts are at the chunk
    level — a single source document can land partly-new (e.g. 3 new
    chunks, 2 dedup'd) when an earlier upload covered some of the same
    material.

    Callers that only care about the boolean "did anything new land"
    can test ``result.written > 0``.
    """

    written: int
    skipped_chunks: int = 0


def collection_name(user_id: str) -> str:
    """``user_rag_<user_id>`` — also used by HybridRetriever for routing."""
    return f"user_rag_{user_id}"


class UserRagStore:
    """High-level ingest facade for per-user RAG content."""

    def __init__(
        self,
        aclient: AsyncQdrantClient,
        embedder: Embedder,
        chunker: Chunker,
        guard: PhiGuard,
    ) -> None:
        self._aclient = aclient
        self._embedder = embedder
        self._chunker = chunker
        self._guard = guard

    # --- factory caches (per user_id) ----------------------------------

    def _collection_store(self, user_id: str) -> RagCollectionStore:
        return RagCollectionStore(
            aclient=self._aclient,
            collection_name=collection_name(user_id),
            dense_dim=self._embedder.dimension,
        )

    def _parent_store(self, user_id: str) -> ParentStore:
        return ParentStore(user_parent_docstore_path(user_id))

    # --- write ---------------------------------------------------------

    async def add_document(
        self,
        user_id: str,
        doc_id: str,
        text: str,
        *,
        metadata: dict | None = None,
        public: bool = False,
    ) -> IngestResult:
        """Scrub, chunk, embed, dedupe, persist.

        When ``public=False`` (default) the raw text is PHI-scrubbed and
        the resulting chunks land with ``is_phi=True / can_cloud=False``.
        When ``public=True`` the scrub is skipped and chunks are marked
        ``is_phi=False / can_cloud=True`` — the user has explicitly
        consented to share this document with cloud models.

        Two dedupe layers run before persist:

        * **Document-level** (``source_uri``) — if the caller supplied
          one and it matches an existing row, raises
          ``DuplicateDocumentError`` immediately.
        * **Chunk-level** (cosine similarity) — each child's dense
          embedding is KNN-searched against the user's existing
          collection; matches at or above
          ``upload.dedupe_cosine_threshold`` are dropped. Threshold
          ``<= 0`` disables this layer.

        Empty / whitespace-only input writes nothing and returns
        ``IngestResult(0, 0)``.
        """
        if not text or not text.strip():
            return IngestResult(written=0)

        # 0. dedup by source_uri (indexed — single lookup, not full scan)
        source_uri = (metadata or {}).get("source_uri")
        col_store = self._collection_store(user_id)
        if source_uri:
            existing = await col_store.find_doc_id_by_source_uri(source_uri)
            if existing:
                from claritymed.errors import DuplicateDocumentError

                raise DuplicateDocumentError(
                    source_uri=source_uri, existing_doc_id=existing
                )

        # 1. PHI scrub
        scrubbed = text if public else self._guard.scrub_free_text(text)[0]

        # 2. chunk
        doc = RawDocument(
            doc_id=doc_id,
            text=scrubbed,
            metadata=metadata or {},
        )
        chunked = self._chunker.chunk(doc)
        if not chunked.children:
            return IngestResult(written=0)

        # 3. embed children (do this BEFORE persisting parents so a
        # dedupe pass that drops every child also skips the parent
        # write — no orphans).
        child_texts = [c.text for c in chunked.children]
        dense_vecs = await self._embedder.embed_dense(child_texts)
        sparse_vecs = await self._embedder.embed_sparse(child_texts)

        # 4. per-chunk cosine-sim dedupe (shared with ingest_corpus)
        from claritymed import config as _cfg

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
            threshold=_cfg.upload_dedupe_cosine_threshold(),
        )

        if not kept_children:
            # Every chunk was a near-duplicate — skip the parent write
            # so the on-disk ParentStore JSON stays clean.
            return IngestResult(written=0, skipped_chunks=skipped_chunks)

        # 5. persist parents (the JSON ParentStore is sync; small write).
        # We persist all parents from the chunker output, even if some
        # of their children were dropped — surviving children still
        # reference them.
        parent_store = self._parent_store(user_id)
        parent_store.bulk_put(chunked.parents)
        parent_store.persist()

        # 6. write surviving children to Qdrant
        written = await col_store.upsert(
            children=kept_children,
            dense_vectors=kept_dense,
            sparse_vectors=kept_sparse,
            is_phi=not public,
            can_cloud=public,
        )
        return IngestResult(written=written, skipped_chunks=skipped_chunks)

    # --- search (independent path; HybridRetriever uses lower stores) --

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        *,
        only_cloud_safe: bool = False,
    ) -> list[RetrievedChunk]:
        """Hybrid search within one user's collection. Empty list when none.

        This is the single-user convenience path (e.g. ``claritymed rag
        search``). The full multi-collection AskService path goes through
        ``HybridRetriever`` which consumes the same underlying
        ``RagCollectionStore`` + ``ParentStore``.
        """
        col_store = self._collection_store(user_id)
        dense_vecs = await self._embedder.embed_dense([query])
        sparse_vecs = await self._embedder.embed_sparse([query])
        if not dense_vecs or not sparse_vecs:
            return []
        hits = await col_store.search_hybrid(
            dense_vecs[0],
            sparse_vecs[0],
            top_k,
            only_cloud_safe=only_cloud_safe,
        )
        parent_store = self._parent_store(user_id)
        return [self._to_retrieved_chunk(user_id, hit, parent_store) for hit in hits]

    # --- list / inspect ------------------------------------------------

    async def list_documents(self, user_id: str) -> list[dict]:
        """Return one summary dict per unique doc_id in the user's collection.

        Each dict has: doc_id, chunk_count, ingested_at, is_phi, can_cloud,
        source_uri, preview (first 120 chars of chunk 0 text).
        """
        coll = collection_name(user_id)
        if not await self._aclient.collection_exists(coll):
            return []

        docs: dict[str, dict] = {}
        offset = None
        while True:
            batch, offset = await self._aclient.scroll(
                collection_name=coll,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in batch:
                p = point.payload or {}
                doc_id = p.get("doc_id", "")
                if not doc_id:
                    continue
                if doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "chunk_count": 0,
                        "ingested_at": p.get("ingested_at", ""),
                        "is_phi": p.get("is_phi", True),
                        "can_cloud": p.get("can_cloud", False),
                        "source_uri": p.get("source_uri"),
                        "preview": "",
                    }
                docs[doc_id]["chunk_count"] += 1
                if p.get("chunk_index", 999) == 0:
                    docs[doc_id]["preview"] = (p.get("text") or "")[:120]
            if offset is None:
                break

        return sorted(docs.values(), key=lambda d: d["ingested_at"])

    async def get_chunks(self, user_id: str, doc_id: str) -> list[dict]:
        """Return all chunks for *doc_id*, ordered by chunk_index.

        Each dict has: chunk_index, text, parent_id, is_phi, can_cloud.
        """
        coll = collection_name(user_id)
        if not await self._aclient.collection_exists(coll):
            return []

        from qdrant_client.http import models as qm

        doc_filter = qm.Filter(
            must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
        )
        chunks = []
        offset = None
        while True:
            batch, offset = await self._aclient.scroll(
                collection_name=coll,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=doc_filter,
            )
            for point in batch:
                p = point.payload or {}
                chunks.append(
                    {
                        "chunk_index": p.get("chunk_index", 0),
                        "text": p.get("text", ""),
                        "parent_id": p.get("parent_id"),
                        "is_phi": p.get("is_phi", True),
                        "can_cloud": p.get("can_cloud", False),
                    }
                )
            if offset is None:
                break

        return sorted(chunks, key=lambda c: c["chunk_index"])

    # --- delete / drop -------------------------------------------------

    async def delete_document(self, user_id: str, doc_id: str) -> None:
        """Remove every chunk + parent record for ``doc_id``."""
        col_store = self._collection_store(user_id)
        await col_store.delete_by_doc_id(doc_id)
        parent_store = self._parent_store(user_id)
        if parent_store.delete_by_doc_id(doc_id) > 0:
            parent_store.persist()

    async def drop_user(self, user_id: str) -> bool:
        """PHI-clean erase: drop the Qdrant collection + parent JSON file.

        Returns True iff anything was actually removed.
        """
        col_store = self._collection_store(user_id)
        dropped_qdrant = await col_store.drop_collection()
        path = user_parent_docstore_path(user_id)
        dropped_parent = path.exists()
        if dropped_parent:
            path.unlink()
        return dropped_qdrant or dropped_parent

    # --- internals -----------------------------------------------------

    def _to_retrieved_chunk(
        self,
        user_id: str,
        hit,
        parent_store: ParentStore,
    ) -> RetrievedChunk:
        payload = hit.payload
        parent_id = payload.get("parent_id")
        parent_text = parent_store.get_text(parent_id) if parent_id else None
        return RetrievedChunk(
            text=hit.text,
            source="user_rag",
            score=hit.score,
            doc_id=str(payload.get("doc_id", "")),
            chunk_index=int(payload.get("chunk_index", 0)),
            is_phi=bool(payload.get("is_phi", True)),
            can_cloud=bool(payload.get("can_cloud", False)),
            user_id=None,
            source_uri=payload.get("source_uri"),
            ingested_at=None,
            collection_name=collection_name(user_id),
            parent_id=parent_id,
            parent_text=parent_text,
        )


def make_user_rag_store(user_id: str) -> UserRagStore:
    """Build a ``UserRagStore`` for one user (local-mode file storage).

    Opens ``AsyncQdrantClient(path=user_rag_qdrant_dir(user_id))`` —
    each user gets a separate on-disk SQLite under their data scope,
    holding the file lock for the process lifetime. System collections
    live on the shared Docker server (see ``build_hybrid_retriever``);
    only user_rag stays local because PHI isolation is worth the
    single-process-per-user concurrency limit.

    Caveat: the same user cannot run ``rag add`` and TUI at the same
    time (file lock is exclusive). Different users never conflict.
    Tests that need in-memory storage construct ``UserRagStore``
    directly with ``AsyncQdrantClient(":memory:")`` instead of going
    through this factory.
    """
    from claritymed.core.rag.chunking.factory import build_chunker
    from claritymed.core.rag.embedding.factory import build_embedder
    from claritymed.stores.paths import user_rag_qdrant_dir

    from claritymed.core.rag.qdrant_store import open_local_qdrant_client

    user_dir = user_rag_qdrant_dir(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    aclient = open_local_qdrant_client(user_dir)
    return UserRagStore(
        aclient=aclient,
        embedder=build_embedder(),
        chunker=build_chunker(),
        guard=PhiGuard.from_config(),
    )


def generate_doc_id() -> str:
    """Random doc id (UUID4 shortened to 12 chars)."""
    return f"doc-{uuid.uuid4().hex[:12]}"
