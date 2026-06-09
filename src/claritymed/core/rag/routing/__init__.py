"""Collection routing: per-query selection of active system collections.

Ships two routers behind the same ``Router`` protocol:

* ``CollectionRouter`` — rule-based: language gate + topic overlap +
  authority tier bias + user whitelist.
* ``CentroidRouter`` — embedding-based: cosine similarity between query
  dense vector and per-collection centroid; falls back to the rule-based
  router per-collection when a centroid file is missing.
"""

from claritymed.core.rag.routing.centroid_router import CentroidRouter
from claritymed.core.rag.routing.centroid_store import (
    CentroidStore,
    compute_centroid,
    maybe_refresh,
)
from claritymed.core.rag.routing.collection_router import (
    CollectionRouter,
    Router,
    RouterTrace,
    RoutingDecision,
)
from claritymed.core.rag.routing.factory import build_router

__all__ = [
    "CentroidRouter",
    "CentroidStore",
    "CollectionRouter",
    "Router",
    "RouterTrace",
    "RoutingDecision",
    "build_router",
    "compute_centroid",
    "maybe_refresh",
]
