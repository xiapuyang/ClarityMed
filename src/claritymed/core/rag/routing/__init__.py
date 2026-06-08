"""Collection routing: per-query selection of active system collections.

v1 ships ``CollectionRouter`` (rule-based: language gate + topic overlap +
authority tier bias + user whitelist). Future plug: ``ClassifierRouter``
that uses a small embedding-similarity model — drop-in replacement of
the ``Router`` protocol.
"""

from claritymed.core.rag.routing.collection_router import (
    CollectionRouter,
    Router,
    RouterTrace,
    RoutingDecision,
)
from claritymed.core.rag.routing.factory import build_router

__all__ = [
    "CollectionRouter",
    "Router",
    "RouterTrace",
    "RoutingDecision",
    "build_router",
]
