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
the old embedder are dimension-incompatible; use ``rag migrate <user_id>``
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
from claritymed.orchestrator import PhiGuard
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

    # --- migration helper ---------------------------------------------

    async def migrate_user(self, user_id: str) -> None:
        """Drop the user's collection + docstore so the next ``add_document``
        rebuilds at the current embedder dimension.

        v1 does **not** auto-reembed: pre-Unit-8 vectors (384-dim fastembed)
        cannot be re-encoded without the original text, which lived only in
        the scrubbed payload. Users in the alpha window have effectively
        no real data; ``rag migrate`` is documented as destructive in the
        CLI help.
        """
        await self.drop_user(user_id)

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


def make_default_user_rag_store(
    qdrant_path: str | None = None,
) -> UserRagStore:
    """Build a ``UserRagStore`` from current config.

    Wires the active embedder + chunker + AsyncQdrantClient pointed at
    ``data/qdrant/user_rag/`` (or ``:memory:`` for tests when callers
    pass that explicitly).
    """
    from claritymed.core.rag.chunking.factory import build_chunker
    from claritymed.core.rag.embedding.factory import build_embedder
    from claritymed.stores.paths import user_rag_qdrant_dir

    path = qdrant_path or str(user_rag_qdrant_dir())
    aclient = (
        AsyncQdrantClient(path=path)
        if path != ":memory:"
        else AsyncQdrantClient(":memory:")
    )
    return UserRagStore(
        aclient=aclient,
        embedder=build_embedder(),
        chunker=build_chunker(),
        guard=PhiGuard.from_config(),
    )


def generate_doc_id() -> str:
    """Random doc id (UUID4 shortened to 12 chars)."""
    return f"doc-{uuid.uuid4().hex[:12]}"
