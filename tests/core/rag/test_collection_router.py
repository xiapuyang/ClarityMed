"""Unit 5: CollectionRouter — rule-based per-query active set selector."""

from __future__ import annotations

import pytest

from claritymed.core.rag.routing import (
    CollectionRouter,
    Router,
    RouterTrace,
    build_router,
)
from claritymed.core.rag.schemas import (
    CollectionMetadata,
    RouterConfig,
    RouterEntry,
    SystemRagConfig,
)
from claritymed.errors import UnknownRouterError


# --- fixtures ---------------------------------------------------------


def _meta(
    name,
    *,
    language="en",
    cross_lingual=False,
    tier=1,
    topics=(),
    disease_codes=(),
) -> CollectionMetadata:
    return CollectionMetadata(
        name=name,
        language=language,
        cross_lingual=cross_lingual,
        authority_tier=tier,
        size_chunks=1000,
        topics=list(topics),
        disease_codes=list(disease_codes),
    )


def _router(
    catalog,
    *,
    max_active=3,
    authority_bias=None,
    default_whitelist=None,
) -> CollectionRouter:
    return CollectionRouter(
        catalog=catalog,
        config=RouterEntry(
            id="rule_based",
            max_active=max_active,
            authority_bias=authority_bias or {1: 0.0, 2: 0.2, 3: 0.5},
        ),
        default_whitelist=default_whitelist,
    )


# --- protocol + happy path -------------------------------------------


def test_router_implements_protocol():
    r = _router([_meta("statpearls_en")])
    assert isinstance(r, Router)


async def test_language_only_match_en_query():
    r = _router(
        [
            _meta("medcorp_en", language="en"),
            _meta("cmb_zh", language="zh"),
        ]
    )
    out = await r.select("aspirin side effects", "en", user_whitelist=None)
    assert out == ["medcorp_en"]


async def test_cross_lingual_collection_is_picked_for_other_language():
    r = _router(
        [
            _meta("statpearls_en", language="en", cross_lingual=True),
            _meta("medcorp_en", language="en"),  # not cross_lingual
            _meta("cmb_zh", language="zh"),
        ]
    )
    out = await r.select("头痛 怎么办", "zh", user_whitelist=None)
    assert "cmb_zh" in out
    assert "statpearls_en" in out  # cross_lingual saves it
    assert "medcorp_en" not in out


async def test_empty_whitelist_returns_empty():
    r = _router([_meta("statpearls_en")])
    assert await r.select("q", "en", user_whitelist=[]) == []


async def test_whitelist_restricts_candidates():
    r = _router(
        [
            _meta("statpearls_en", language="en"),
            _meta("medcorp_en", language="en"),
        ]
    )
    out = await r.select("q", "en", user_whitelist=["statpearls_en"])
    assert out == ["statpearls_en"]


async def test_max_active_caps_output():
    catalog = [_meta(f"col_en_{i}", language="en", tier=1) for i in range(5)]
    r = _router(catalog, max_active=2)
    out = await r.select("q", "en", user_whitelist=None)
    assert len(out) == 2


async def test_default_whitelist_used_when_user_whitelist_is_none():
    r = _router(
        [
            _meta("a_en", language="en"),
            _meta("b_en", language="en"),
        ],
        default_whitelist=["a_en"],
    )
    out = await r.select("q", "en", user_whitelist=None)
    assert out == ["a_en"]


# --- authority bias & topic overlap ---------------------------------


async def test_higher_tier_requires_topic_overlap():
    r = _router(
        [
            _meta(
                "low_quality_en",
                language="en",
                tier=3,
                topics=["uniquetopicword"],
            ),
        ],
        authority_bias={1: 0.0, 2: 0.5, 3: 1.0},
    )
    # Query doesn't contain "uniquetopicword" → score 0.0 < threshold 1.0 → out.
    assert await r.select("q with nothing matching", "en", user_whitelist=None) == []


async def test_topic_overlap_admits_high_tier_when_match():
    r = _router(
        [
            _meta(
                "low_quality_en",
                language="en",
                tier=3,
                topics=["headache"],
            ),
        ],
        authority_bias={1: 0.0, 2: 0.5, 3: 1.0},
    )
    assert await r.select("headache pain relief", "en", user_whitelist=None) == [
        "low_quality_en"
    ]


async def test_chinese_substring_topic_match():
    # Topic words don't tokenize in CJK; substring path must match.
    r = _router(
        [
            _meta(
                "diabetes_zh",
                language="zh",
                tier=2,
                topics=["糖尿病"],
            )
        ],
        authority_bias={1: 0.0, 2: 0.5, 3: 1.0},
    )
    assert await r.select("我患有糖尿病应该怎么办", "zh", user_whitelist=None) == [
        "diabetes_zh"
    ]


async def test_tier_one_admitted_without_topic_match():
    # tier=1, authority_bias[1]=0.0 → always passes language gate.
    r = _router([_meta("statpearls_en", language="en", tier=1, topics=["general"])])
    assert await r.select("anything goes here", "en", user_whitelist=None) == [
        "statpearls_en"
    ]


# --- trace ----------------------------------------------------------


async def test_select_with_trace_returns_per_collection_decisions():
    r = _router(
        [
            _meta("medcorp_en", language="en"),
            _meta("cmb_zh", language="zh"),
            _meta("statpearls_en", language="en", cross_lingual=True),
        ]
    )
    trace = await r.select_with_trace("aspirin", "en", user_whitelist=None)
    assert isinstance(trace, RouterTrace)
    decisions_by_name = {d.name: d for d in trace.considered}
    assert decisions_by_name["medcorp_en"].selected is True
    assert decisions_by_name["cmb_zh"].selected is False
    assert "language" in decisions_by_name["cmb_zh"].reason


async def test_capped_collections_appear_in_trace_with_reason():
    catalog = [_meta(f"c_en_{i}", language="en", tier=1) for i in range(4)]
    r = _router(catalog, max_active=2)
    trace = await r.select_with_trace("q", "en", user_whitelist=None)
    capped = [d for d in trace.considered if not d.selected and "capped" in d.reason]
    assert len(capped) == 2


# --- edge cases -----------------------------------------------------


async def test_empty_catalog_returns_empty():
    r = _router([])
    assert await r.select("q", "en", user_whitelist=None) == []


async def test_whitelist_with_unknown_collection_is_ignored():
    r = _router([_meta("statpearls_en", language="en")])
    out = await r.select("q", "en", user_whitelist=["statpearls_en", "does_not_exist"])
    assert out == ["statpearls_en"]


# --- factory --------------------------------------------------------


async def test_build_router_factory_happy_path():
    cfg_router = RouterConfig(
        active="rule_based",
        catalog=[
            {  # type: ignore[list-item]
                "id": "rule_based",
                "max_active": 2,
                "authority_bias": {1: 0.0},
            }
        ],
    )
    cfg_system = SystemRagConfig(
        default_active=["statpearls_en"],
        collections=[
            _meta("statpearls_en", language="en", cross_lingual=True)  # type: ignore[list-item]
        ],
    )
    r = build_router(router_config=cfg_router, system_rag=cfg_system)
    assert isinstance(r, CollectionRouter)
    assert await r.select("any", "en", user_whitelist=None) == ["statpearls_en"]


def test_build_router_unknown_id_raises():
    cfg_router = RouterConfig.model_construct(
        active="classifier_v1",
        catalog=[
            RouterEntry.model_construct(
                id="classifier_v1",
                max_active=3,
                authority_bias={1: 0.0},
            )
        ],
    )
    cfg_system = SystemRagConfig(default_active=[], collections=[])
    with pytest.raises(UnknownRouterError):
        build_router(router_config=cfg_router, system_rag=cfg_system)
