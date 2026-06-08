"""Resolve the active ``Router`` from ``configs/retrieval.yaml``."""

from __future__ import annotations

from claritymed.core.rag.routing.collection_router import CollectionRouter, Router
from claritymed.core.rag.schemas import (
    CollectionMetadata,
    RouterConfig,
    SystemRagConfig,
    load_retrieval_config,
)
from claritymed.errors import UnknownRouterError


def build_router(
    *,
    router_config: RouterConfig | None = None,
    system_rag: SystemRagConfig | None = None,
) -> Router:
    """Instantiate the active router with the system RAG catalog wired in."""
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
    raise UnknownRouterError(f"build_router has no factory branch for id={entry.id!r}")
