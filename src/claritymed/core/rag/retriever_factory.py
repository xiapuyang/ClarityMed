"""Assemble a fully-wired ``HybridRetriever`` from ``configs/retrieval.yaml``.

This is the bootstrap seam called by the CLI and TUI on startup when
``rag.enabled=true``. It wires every component the retriever needs:

* ``Embedder``       — from ``embedders.active`` (HTTP server at startup).
* ``Reranker``       — from ``rerankers.active``.
* ``TermService``    — from ``term_service.active``.
* ``Router``         — ``CollectionRouter`` populated with the
  ``system_rag.collections`` catalog.
* ``system_store``   — one ``AsyncQdrantClient`` pointed at
  ``shared_qdrant_dir()``; a per-collection wrapper per name.
* ``system_parent_store`` — single JSON KV at
  ``shared_parent_docstore_path()``.
* ``user_store``     — second ``AsyncQdrantClient`` pointed at
  ``user_rag_qdrant_dir()``; returns ``None`` when the per-user
  collection does not yet exist (so the retriever still falls back to
  system-only).
* ``user_parent_store`` — one ``ParentStore`` per user, lazily.

Fail-loud once opted in:

* Missing embedder server → ``EmbedderUnreachableError`` from the first
  call (we do **not** ping at startup; the cost of a real probe outweighs
  the benefit when the user can just re-run after starting the server).
* Unknown active id in any catalog → ``Unknown<X>Error`` immediately on
  ``load_retrieval_config().<section>.resolved()``.
* ``rag.enabled=false`` → callers should not even reach this module;
  ``build_hybrid_retriever`` itself does not re-check the flag (mirrors
  pattern: ``build_model`` does not check ``cloud_provider_opt_in``).

The two ``AsyncQdrantClient`` instances are created at startup and live
for the process lifetime; the underlying ``qdrant_client.local`` is
file-locked per directory, so the system and user Qdrant directories
must stay separate (they already are, via ``paths.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qdrant_client import AsyncQdrantClient

from claritymed.core.rag.embedding.factory import build_embedder
from claritymed.core.rag.parent_store import ParentStore
from claritymed.core.rag.qdrant_store import RagCollectionStore
from claritymed.core.rag.reranking.factory import build_reranker
from claritymed.core.rag.retriever import HybridRetriever
from claritymed.core.rag.routing.factory import build_router
from claritymed.core.rag.schemas import RetrievalConfig, load_retrieval_config
from claritymed.core.rag.terms.factory import build_term_service
from claritymed.stores.paths import (
    shared_parent_docstore_path,
    shared_qdrant_dir,
    user_parent_docstore_path,
    user_rag_qdrant_dir,
)

if TYPE_CHECKING:
    from claritymed.core.rag.embedding.base import Embedder


def _user_collection_name(user_id: str) -> str:
    # Mirror ``stores.user_rag.collection_name``; imported lazily would
    # create a circular dep (stores.user_rag → core.rag.chunking →
    # core.rag.__init__ → this module). One f-string is cheaper than the
    # indirection.
    return f"user_rag_{user_id}"


def build_hybrid_retriever(
    config: RetrievalConfig | None = None,
) -> HybridRetriever:
    """Wire every component into a HybridRetriever ready for async use.

    Args:
        config: Optional ``RetrievalConfig`` override. When ``None``,
            loads from ``configs/retrieval.yaml`` via the mtime cache.

    Returns:
        A HybridRetriever wired against the active embedder / reranker /
        term service / router and pointed at the on-disk Qdrant
        directories.
    """
    cfg = config or load_retrieval_config()

    embedder = build_embedder(cfg.embedders)
    reranker = build_reranker(cfg.rerankers)
    term_service = build_term_service(cfg.term_service)
    router = build_router(router_config=cfg.router, system_rag=cfg.system_rag)

    system_aclient = AsyncQdrantClient(path=str(shared_qdrant_dir()))
    user_aclient = AsyncQdrantClient(path=str(user_rag_qdrant_dir()))
    system_parent_store = ParentStore(shared_parent_docstore_path())

    return HybridRetriever(
        embedder=embedder,
        reranker=reranker,
        term_service=term_service,
        router=router,
        system_store_factory=_make_system_store_factory(system_aclient, embedder),
        system_parent_store=system_parent_store,
        user_store_factory=_make_user_store_factory(user_aclient, embedder),
        user_parent_store_factory=_make_user_parent_store_factory(),
        rerank_top_k=cfg.user_rag.rerank_k,
    )


def _make_system_store_factory(aclient: AsyncQdrantClient, embedder: "Embedder"):
    def factory(name: str) -> RagCollectionStore:
        return RagCollectionStore(
            aclient=aclient,
            collection_name=name,
            dense_dim=embedder.dimension,
        )

    return factory


def _make_user_store_factory(aclient: AsyncQdrantClient, embedder: "Embedder"):
    async def factory(user_id: str) -> RagCollectionStore | None:
        name = _user_collection_name(user_id)
        if not await aclient.collection_exists(name):
            return None
        return RagCollectionStore(
            aclient=aclient,
            collection_name=name,
            dense_dim=embedder.dimension,
        )

    return factory


def _make_user_parent_store_factory():
    def factory(user_id: str) -> ParentStore | None:
        path = user_parent_docstore_path(user_id)
        if not path.exists():
            return None
        return ParentStore(path)

    return factory
