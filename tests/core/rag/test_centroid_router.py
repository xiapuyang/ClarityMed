"""Unit tests for ``CentroidRouter`` — embedding-based collection routing."""

from __future__ import annotations

import pytest

from claritymed.core.rag.routing.centroid_router import (
    CentroidRouter,
    _cosine_similarity,
)
from claritymed.core.rag.routing.centroid_store import CentroidStore
from claritymed.core.rag.routing.collection_router import CollectionRouter, Router
from claritymed.core.rag.routing.factory import build_router
from claritymed.core.rag.schemas import (
    CollectionMetadata,
    RouterConfig,
    RouterEntry,
    SystemRagConfig,
)
from claritymed.errors import UnknownRouterError

DENSE_DIM = 4


# --- fixtures & helpers -------------------------------------------------


def _meta(
    name: str,
    *,
    language: str = "en",
    cross_lingual: bool = False,
    tier: int = 1,
    topics=(),
) -> CollectionMetadata:
    return CollectionMetadata(
        name=name,
        language=language,
        cross_lingual=cross_lingual,
        authority_tier=tier,
        size_chunks=1000,
        topics=list(topics),
    )


class _StubEmbedder:
    """Embedder stub that returns a configured dense vector per query."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector if vector is not None else [1.0, 0.0, 0.0, 0.0]
        self.calls: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return len(self._vector)

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if not texts:
            return []
        return [list(self._vector)]

    async def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        return [{} for _ in texts]


def _entry(*, max_active: int = 3, min_similarity: float = 0.1) -> RouterEntry:
    return RouterEntry(
        id="centroid_classifier",
        max_active=max_active,
        min_similarity=min_similarity,
        authority_bias={1: 0.0, 2: 0.2, 3: 0.5},
    )


def _fallback(
    catalog: list[CollectionMetadata], *, default_whitelist=None
) -> CollectionRouter:
    return CollectionRouter(
        catalog=catalog,
        config=RouterEntry(id="rule_based", max_active=3, authority_bias={1: 0.0}),
        default_whitelist=default_whitelist,
    )


def _make_router(
    *,
    catalog: list[CollectionMetadata],
    centroids: dict[str, list[float]],
    qvec: list[float],
    min_similarity: float = 0.1,
    max_active: int = 3,
    default_whitelist=None,
) -> tuple[CentroidRouter, CentroidStore, _StubEmbedder]:
    """Construct a CentroidRouter wired against an in-memory CentroidStore."""

    class _InMemStore:
        def vector(self, name: str) -> list[float] | None:
            return centroids.get(name)

    store = _InMemStore()
    embedder = _StubEmbedder(vector=qvec)
    router = CentroidRouter(
        catalog=catalog,
        config=_entry(max_active=max_active, min_similarity=min_similarity),
        centroid_store=store,  # type: ignore[arg-type]
        embedder=embedder,
        fallback=_fallback(catalog, default_whitelist=default_whitelist),
        default_whitelist=default_whitelist,
    )
    return router, store, embedder  # type: ignore[return-value]


# --- cosine similarity -------------------------------------------------


def test_cosine_similarity_orthogonal_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_identical_is_one():
    assert _cosine_similarity([0.5, 0.5], [0.5, 0.5]) == pytest.approx(1.0)


def test_cosine_similarity_handles_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_similarity_handles_mismatched_dims():
    assert _cosine_similarity([1.0, 0.0], [1.0]) == 0.0


# --- protocol + happy path ---------------------------------------------


async def test_centroid_router_implements_protocol():
    catalog = [_meta("statpearls_en")]
    router, _, _ = _make_router(
        catalog=catalog,
        centroids={"statpearls_en": [1.0, 0.0, 0.0, 0.0]},
        qvec=[1.0, 0.0, 0.0, 0.0],
    )
    assert isinstance(router, Router)


async def test_routes_to_collection_closest_to_query_centroid():
    catalog = [
        _meta("statpearls_en"),
        _meta("textbooks_en"),
    ]
    router, _, embedder = _make_router(
        catalog=catalog,
        centroids={
            "statpearls_en": [1.0, 0.0, 0.0, 0.0],
            "textbooks_en": [0.0, 1.0, 0.0, 0.0],
        },
        qvec=[1.0, 0.0, 0.0, 0.0],
        min_similarity=0.5,
    )
    trace = await router.select_with_trace(
        "iron deficiency anemia", "en", user_whitelist=None
    )
    assert trace.selected == ["statpearls_en"]
    assert embedder.calls == [["iron deficiency anemia"]]
    decisions_by_name = {d.name: d for d in trace.considered}
    assert decisions_by_name["statpearls_en"].selected is True
    assert decisions_by_name["textbooks_en"].selected is False
    assert "centroid_score" in decisions_by_name["statpearls_en"].reason


async def test_threshold_filters_out_below_min_similarity():
    catalog = [_meta("statpearls_en")]
    # qvec is orthogonal to centroid → similarity ~0 < threshold 0.5
    router, _, _ = _make_router(
        catalog=catalog,
        centroids={"statpearls_en": [1.0, 0.0, 0.0, 0.0]},
        qvec=[0.0, 1.0, 0.0, 0.0],
        min_similarity=0.5,
    )
    trace = await router.select_with_trace("q", "en", user_whitelist=None)
    assert trace.selected == []
    decisions = {d.name: d for d in trace.considered}
    assert "min_similarity" in decisions["statpearls_en"].reason


async def test_max_active_caps_to_top_n_by_score():
    catalog = [
        _meta("col_a"),
        _meta("col_b"),
        _meta("col_c"),
    ]
    router, _, _ = _make_router(
        catalog=catalog,
        centroids={
            # Closest to qvec → best score
            "col_a": [1.0, 0.0, 0.0, 0.0],
            "col_b": [0.9, 0.1, 0.0, 0.0],
            "col_c": [0.8, 0.2, 0.0, 0.0],
        },
        qvec=[1.0, 0.0, 0.0, 0.0],
        max_active=2,
        min_similarity=0.0,
    )
    trace = await router.select_with_trace("q", "en", user_whitelist=None)
    assert trace.selected == ["col_a", "col_b"]
    capped = [d for d in trace.considered if not d.selected and "capped" in d.reason]
    assert len(capped) == 1
    assert capped[0].name == "col_c"


# --- fallback paths ----------------------------------------------------


async def test_collection_without_centroid_falls_back_to_rule_based():
    catalog = [
        _meta("statpearls_en", topics=["headache"]),
        _meta("textbooks_en", topics=["anatomy"]),
    ]
    router, _, _ = _make_router(
        catalog=catalog,
        # Only statpearls has a centroid; textbooks decision delegated.
        centroids={"statpearls_en": [1.0, 0.0, 0.0, 0.0]},
        qvec=[1.0, 0.0, 0.0, 0.0],
    )
    trace = await router.select_with_trace(
        "anatomy of the heart", "en", user_whitelist=None
    )
    decisions = {d.name: d for d in trace.considered}
    # textbooks_en has no centroid → rule-based fallback governs it.
    assert "centroid_absent" in decisions["textbooks_en"].reason
    assert "rule_based_fallback" in decisions["textbooks_en"].reason


async def test_all_centroids_absent_delegates_fully_to_fallback():
    catalog = [_meta("statpearls_en", topics=["headache"])]
    router, _, embedder = _make_router(
        catalog=catalog,
        centroids={},
        qvec=[1.0, 0.0, 0.0, 0.0],
    )
    trace = await router.select_with_trace("headache relief", "en", user_whitelist=None)
    # Falls back fully — selected matches what the rule-based router would do.
    assert trace.selected == ["statpearls_en"]
    # The embedder was never called — full delegation skips embedding entirely.
    assert embedder.calls == []


async def test_opted_out_returns_empty_without_embedding():
    catalog = [_meta("statpearls_en")]
    router, _, embedder = _make_router(
        catalog=catalog,
        centroids={"statpearls_en": [1.0, 0.0, 0.0, 0.0]},
        qvec=[1.0, 0.0, 0.0, 0.0],
    )
    trace = await router.select_with_trace("q", "en", user_whitelist=[])
    assert trace.selected == []
    # Opt-out short-circuit must not consume an embed call.
    assert embedder.calls == []


async def test_whitelist_gate_applies_to_centroid_collections():
    catalog = [
        _meta("statpearls_en"),
        _meta("textbooks_en"),
    ]
    router, _, _ = _make_router(
        catalog=catalog,
        centroids={
            "statpearls_en": [1.0, 0.0, 0.0, 0.0],
            "textbooks_en": [1.0, 0.0, 0.0, 0.0],
        },
        qvec=[1.0, 0.0, 0.0, 0.0],
    )
    trace = await router.select_with_trace("q", "en", user_whitelist=["statpearls_en"])
    assert trace.selected == ["statpearls_en"]
    decisions = {d.name: d for d in trace.considered}
    assert "not in whitelist" in decisions["textbooks_en"].reason


async def test_language_gate_blocks_mismatched_collection():
    catalog = [
        _meta("statpearls_en", language="en", cross_lingual=False),
    ]
    router, _, _ = _make_router(
        catalog=catalog,
        centroids={"statpearls_en": [1.0, 0.0, 0.0, 0.0]},
        qvec=[1.0, 0.0, 0.0, 0.0],
    )
    trace = await router.select_with_trace("q", "zh", user_whitelist=None)
    assert trace.selected == []
    decisions = {d.name: d for d in trace.considered}
    assert "language" in decisions["statpearls_en"].reason


# --- factory wiring ----------------------------------------------------


def test_build_router_factory_returns_centroid_router(tmp_path):
    cfg_router = RouterConfig(
        active="centroid_classifier",
        catalog=[
            {  # type: ignore[list-item]
                "id": "centroid_classifier",
                "max_active": 2,
                "min_similarity": 0.1,
                "authority_bias": {1: 0.0},
            }
        ],
    )
    cfg_system = SystemRagConfig(
        default_active=["statpearls_en"],
        collections=[
            _meta("statpearls_en", cross_lingual=True),  # type: ignore[list-item]
        ],
    )
    embedder = _StubEmbedder()
    r = build_router(
        router_config=cfg_router,
        system_rag=cfg_system,
        embedder=embedder,
        centroid_store=CentroidStore(tmp_path / "centroids"),
    )
    assert isinstance(r, CentroidRouter)


def test_build_router_factory_centroid_requires_embedder():
    cfg_router = RouterConfig(
        active="centroid_classifier",
        catalog=[
            {  # type: ignore[list-item]
                "id": "centroid_classifier",
                "max_active": 2,
                "min_similarity": 0.1,
                "authority_bias": {1: 0.0},
            }
        ],
    )
    cfg_system = SystemRagConfig(default_active=[], collections=[])
    with pytest.raises(ValueError, match="embedder"):
        build_router(router_config=cfg_router, system_rag=cfg_system)


def test_build_router_unknown_id_still_raises():
    cfg_router = RouterConfig.model_construct(
        active="classifier_v1",
        catalog=[
            RouterEntry.model_construct(
                id="classifier_v1",
                max_active=3,
                authority_bias={1: 0.0},
                min_similarity=0.1,
            )
        ],
    )
    cfg_system = SystemRagConfig(default_active=[], collections=[])
    with pytest.raises(UnknownRouterError):
        build_router(router_config=cfg_router, system_rag=cfg_system)
