"""Resolve the active ``Router`` from ``configs/retrieval.yaml``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from claritymed.core.rag.routing.centroid_router import CentroidRouter
from claritymed.core.rag.routing.centroid_store import CentroidStore
from claritymed.core.rag.routing.collection_router import CollectionRouter, Router
from claritymed.core.rag.schemas import (
    CollectionMetadata,
    RouterConfig,
    SystemRagConfig,
    load_retrieval_config,
)
from claritymed.errors import UnknownRouterError

if TYPE_CHECKING:
    from claritymed.core.rag.embedding.base import Embedder


def build_router(
    *,
    router_config: RouterConfig | None = None,
    system_rag: SystemRagConfig | None = None,
    embedder: "Embedder | None" = None,
    centroid_store: CentroidStore | None = None,
) -> Router:
    """Instantiate the active router with the system RAG catalog wired in.

    ``embedder`` and ``centroid_store`` are only required when the active
    router is ``centroid_classifier``; passing them unconditionally from
    ``build_hybrid_retriever`` is harmless because the rule-based branch
    ignores them.
    """
    cfg = (
        load_retrieval_config() if router_config is None or system_rag is None else None
    )
    router_cfg = router_config or (cfg.router if cfg else None)
    system_cfg = system_rag or (cfg.system_rag if cfg else None)
    if router_cfg is None or system_cfg is None:
        raise ValueError("build_router needs both router_config and system_rag")

    entry = router_cfg.resolved()
    catalog: list[CollectionMetadata] = list(system_cfg.collections)
    default_whitelist: list[str] = list(system_cfg.default_active)

    if entry.id == "rule_based":
        return CollectionRouter(
            catalog=catalog,
            config=entry,
            default_whitelist=default_whitelist,
        )
    if entry.id == "centroid_classifier":
        if embedder is None:
            raise ValueError(
                "centroid_classifier router requires an Embedder — "
                "build_router(..., embedder=...) must be passed when "
                "router.active is centroid_classifier"
            )
        if centroid_store is None:
            # Default location mirrors ``rag corpora refresh-centroid``.
            from claritymed.stores.paths import shared_root

            centroid_store = CentroidStore(shared_root() / "centroids")
        fallback = CollectionRouter(
            catalog=catalog,
            config=entry,
            default_whitelist=default_whitelist,
        )
        return CentroidRouter(
            catalog=catalog,
            config=entry,
            centroid_store=centroid_store,
            embedder=embedder,
            fallback=fallback,
            default_whitelist=default_whitelist,
        )
    raise UnknownRouterError(f"build_router has no factory branch for id={entry.id!r}")
