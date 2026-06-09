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

from qdrant_client import AsyncQdrantClient

from claritymed.core.rag.chunking.base import Chunker, RawDocument
from claritymed.core.rag.embedding.base import Embedder
from claritymed.core.rag.parent_store import ParentStore
from claritymed.core.rag.qdrant_store import RagCollectionStore
from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.core.phi.guard import PhiGuard
from claritymed.stores.paths import user_parent_docstore_path

logger = logging.getLogger(__name__)


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
    ) -> int:
        """Scrub, chunk, embed, persist. Returns child-chunk count written.

        When ``public=False`` (default) the raw text is PHI-scrubbed and
        the resulting chunks land with ``is_phi=True / can_cloud=False``.
        When ``public=True`` the scrub is skipped and chunks are marked
        ``is_phi=False / can_cloud=True`` — the user has explicitly
        consented to share this document with cloud models.

        Empty / whitespace-only input writes nothing and returns 0.
        """
        if not text or not text.strip():
            return 0

        # 0. dedup by source_uri (indexed — single lookup, not full scan)
        source_uri = (metadata or {}).get("source_uri")
        if source_uri:
            col_store = self._collection_store(user_id)
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
            return 0

        # 3. persist parents (the JSON ParentStore is sync; small write)
        parent_store = self._parent_store(user_id)
        parent_store.bulk_put(chunked.parents)
        parent_store.persist()

        # 4. embed children
        child_texts = [c.text for c in chunked.children]
        dense_vecs = await self._embedder.embed_dense(child_texts)
        sparse_vecs = await self._embedder.embed_sparse(child_texts)

        # 5. write children to Qdrant
        col_store = self._collection_store(user_id)
        written = await col_store.upsert(
            children=chunked.children,
            dense_vectors=dense_vecs,
            sparse_vectors=sparse_vecs,
            is_phi=not public,
            can_cloud=public,
        )
        return written

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

    user_dir = user_rag_qdrant_dir(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    aclient = AsyncQdrantClient(path=str(user_dir))
    return UserRagStore(
        aclient=aclient,
        embedder=build_embedder(),
        chunker=build_chunker(),
        guard=PhiGuard.from_config(),
    )


def generate_doc_id() -> str:
    """Random doc id (UUID4 shortened to 12 chars)."""
    return f"doc-{uuid.uuid4().hex[:12]}"
