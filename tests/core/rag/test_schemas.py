"""Unit 1 of the RAG plan: contract + config-loader tests.

Covers ``CollectionMetadata`` / ``RetrievalTrace`` / ``EvidenceBundle`` /
``RetrievedChunk`` extensions / ``RetrievalConfig`` catalog resolution.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claritymed.core.rag import (
    CollectionMetadata,
    EvidenceBundle,
    GraderReport,
    RetrievalConfig,
    RetrievalTrace,
    load_retrieval_config,
)
from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.errors import (
    UnknownChunkerError,
    UnknownEmbedderError,
    UnknownRerankerError,
    UnknownRouterError,
    UnknownStrategyError,
    UnknownTermServiceError,
)


# --- CollectionMetadata -------------------------------------------------


def test_collection_metadata_happy_path():
    md = CollectionMetadata(
        name="statpearls_en",
        language="en",
        cross_lingual=True,
        authority_tier=1,
        size_chunks=12345,
        topics=["general clinical"],
        disease_codes=["I10"],
    )
    assert md.name == "statpearls_en"
    assert md.cross_lingual is True


def test_collection_metadata_is_frozen():
    md = CollectionMetadata(name="x_en", language="en", authority_tier=1)
    with pytest.raises(ValidationError):
        md.name = "y_en"  # type: ignore[misc]


def test_collection_metadata_rejects_extra_field():
    with pytest.raises(ValidationError):
        CollectionMetadata(
            name="x_en",
            language="en",
            authority_tier=1,
            unknown_field="boom",  # type: ignore[call-arg]
        )


def test_collection_metadata_rejects_invalid_name():
    # Uppercase / hyphen / leading digit all violate the pattern.
    for bad in ["BadName", "bad-name", "1leading"]:
        with pytest.raises(ValidationError):
            CollectionMetadata(name=bad, language="en", authority_tier=1)


def test_collection_metadata_rejects_invalid_authority_tier():
    for bad in [0, 4, -1]:
        with pytest.raises(ValidationError):
            CollectionMetadata(name="x_en", language="en", authority_tier=bad)


def test_collection_metadata_rejects_invalid_language():
    with pytest.raises(ValidationError):
        CollectionMetadata(
            name="x_ja",
            language="ja",  # type: ignore[arg-type]
            authority_tier=1,
        )


# --- RetrievalTrace / GraderReport / EvidenceBundle ---------------------


def test_retrieval_trace_happy_path():
    trace = RetrievalTrace(
        strategy="naive_hybrid", active_collections=["statpearls_en"]
    )
    assert trace.fallback_triggered is False
    assert trace.grader is None


def test_retrieval_trace_fallback_requires_grader():
    # Plan §Decisions: fallback_triggered=True is only legal when CRAG-lite
    # grader produced a "rewrite" decision.
    with pytest.raises(ValidationError):
        RetrievalTrace(
            strategy="naive_hybrid",
            fallback_triggered=True,
        )


def test_retrieval_trace_fallback_with_grader_ok():
    grader = GraderReport(threshold=0.5, mean_rerank_score=0.3, decision="rewrite")
    trace = RetrievalTrace(
        strategy="naive_hybrid",
        fallback_triggered=True,
        grader=grader,
    )
    assert trace.grader is not None and trace.grader.decision == "rewrite"


def test_grader_report_decision_literal():
    with pytest.raises(ValidationError):
        GraderReport(threshold=0.5, mean_rerank_score=0.4, decision="maybe")  # type: ignore[arg-type]


def test_evidence_bundle_packs_chunks_and_trace():
    chunk = RetrievedChunk(
        text="hello",
        source="system_rag",
        score=0.9,
        doc_id="d1",
        is_phi=False,
        collection_name="statpearls_en",
    )
    bundle = EvidenceBundle(
        chunks=[chunk],
        trace=RetrievalTrace(strategy="naive_hybrid"),
    )
    assert bundle.chunks[0].collection_name == "statpearls_en"
    assert bundle.trace.strategy == "naive_hybrid"


# --- RetrievedChunk backward compatibility ------------------------------


def test_retrieved_chunk_accepts_v1_payload():
    # The pre-extension shape (used by current UserRagStore.search) must
    # still validate after Unit 1's optional-field additions.
    chunk = RetrievedChunk(
        text="t",
        source="user_rag",
        score=0.8,
        doc_id="d1",
        chunk_index=0,
        is_phi=True,
        can_cloud=False,
    )
    assert chunk.collection_name is None
    assert chunk.parent_text is None
    assert chunk.dense_score is None


def test_retrieved_chunk_extended_payload_round_trip():
    chunk = RetrievedChunk(
        text="child text",
        source="system_rag",
        score=0.0,
        doc_id="statpearls/123",
        is_phi=False,
        collection_name="statpearls_en",
        parent_id="statpearls/123#p0",
        parent_text="long parent paragraph",
        dense_score=0.81,
        sparse_score=0.42,
        rerank_score=0.93,
    )
    dumped = chunk.model_dump()
    assert dumped["collection_name"] == "statpearls_en"
    assert dumped["rerank_score"] == 0.93


# --- RetrievalConfig from real configs/retrieval.yaml -------------------


def test_load_retrieval_config_parses_real_yaml():
    cfg = load_retrieval_config()
    assert cfg.strategies.active == "naive_hybrid"
    assert cfg.strategies.resolved().id == "naive_hybrid"
    assert cfg.embedders.active == "bge_m3_http"
    assert cfg.embedders.resolved().dense_dim == 1024
    assert cfg.rerankers.resolved().base_url.startswith("http")
    assert cfg.chunker.resolved().id == "parent_child"
    assert cfg.router.resolved().max_active >= 1
    assert any(c.name == "statpearls_en" for c in cfg.system_rag.collections)


def test_retrieval_config_unknown_embedder_active_fails_loud():
    raw = _base_raw_config()
    raw["embedders"]["active"] = "does_not_exist"
    with pytest.raises(UnknownEmbedderError):
        RetrievalConfig.model_validate(raw)


def test_retrieval_config_unknown_strategy_active_fails_loud():
    raw = _base_raw_config()
    raw["strategies"]["active"] = "totally_made_up"
    with pytest.raises(UnknownStrategyError):
        RetrievalConfig.model_validate(raw)


def test_retrieval_config_unknown_reranker_active_fails_loud():
    raw = _base_raw_config()
    raw["rerankers"]["active"] = "nope"
    with pytest.raises(UnknownRerankerError):
        RetrievalConfig.model_validate(raw)


def test_retrieval_config_unknown_chunker_active_fails_loud():
    raw = _base_raw_config()
    raw["chunker"]["active"] = "nope"
    with pytest.raises(UnknownChunkerError):
        RetrievalConfig.model_validate(raw)


def test_retrieval_config_unknown_term_service_active_fails_loud():
    raw = _base_raw_config()
    raw["term_service"]["active"] = "nope"
    with pytest.raises(UnknownTermServiceError):
        RetrievalConfig.model_validate(raw)


def test_retrieval_config_unknown_router_active_fails_loud():
    raw = _base_raw_config()
    raw["router"]["active"] = "nope"
    with pytest.raises(UnknownRouterError):
        RetrievalConfig.model_validate(raw)


def test_system_rag_default_active_must_be_known():
    raw = _base_raw_config()
    raw["system_rag"]["default_active"] = ["not_a_real_collection"]
    with pytest.raises(ValidationError):
        RetrievalConfig.model_validate(raw)


def test_authority_bias_keys_are_coerced_from_yaml_strings():
    raw = _base_raw_config()
    raw["router"]["catalog"][0]["authority_bias"] = {"1": 0.0, "2": 0.5}
    cfg = RetrievalConfig.model_validate(raw)
    bias = cfg.router.resolved().authority_bias
    assert bias[1] == 0.0
    assert bias[2] == 0.5


def test_resolved_raises_unknown_chunker_when_bypassing_validator():
    """model_construct skips _resolve_active; resolved() must still raise."""
    from claritymed.core.rag.schemas import ChunkerConfig, ParentChildChunkerConfig

    entry = ParentChildChunkerConfig.model_construct(
        id="parent_child", child_tok=128, parent_tok=512, overlap_tok=0
    )
    cfg = ChunkerConfig.model_construct(active="missing", catalog=[entry])
    with pytest.raises(UnknownChunkerError):
        cfg.resolved()


def test_resolved_raises_unknown_embedder_when_bypassing_validator():
    from claritymed.core.rag.schemas import EmbedderConfig, EmbedderEntry

    entry = EmbedderEntry.model_construct(
        id="bge",
        kind="http",
        base_url="http://x",
        dense_dim=1024,
        batch_size=32,
        timeout_s=30,
    )
    cfg = EmbedderConfig.model_construct(active="missing", catalog=[entry])
    with pytest.raises(UnknownEmbedderError):
        cfg.resolved()


def test_resolved_raises_unknown_reranker_when_bypassing_validator():
    from claritymed.core.rag.schemas import RerankerConfig, RerankerEntry

    entry = RerankerEntry.model_construct(
        id="bge", kind="http", base_url="http://x", batch_size=32, timeout_s=30
    )
    cfg = RerankerConfig.model_construct(active="missing", catalog=[entry])
    with pytest.raises(UnknownRerankerError):
        cfg.resolved()


def test_resolved_raises_unknown_term_service_when_bypassing_validator():
    from claritymed.core.rag.schemas import TermServiceConfig, TermServiceEntry

    entry = TermServiceEntry.model_construct(id="snomed", kind="snomed_ct")
    cfg = TermServiceConfig.model_construct(active="missing", catalog=[entry])
    with pytest.raises(UnknownTermServiceError):
        cfg.resolved()


def test_resolved_raises_unknown_router_when_bypassing_validator():
    from claritymed.core.rag.schemas import RouterConfig, RouterEntry

    entry = RouterEntry.model_construct(
        id="rule_based", max_active=3, authority_bias={1: 0.0}
    )
    cfg = RouterConfig.model_construct(active="missing", catalog=[entry])
    with pytest.raises(UnknownRouterError):
        cfg.resolved()


# --- helpers ------------------------------------------------------------


def _base_raw_config() -> dict:
    """Minimal valid raw config dict for unit-tests of validator branches."""
    return {
        "strategies": {
            "active": "naive_hybrid",
            "catalog": [{"id": "naive_hybrid", "grader": {"enabled": False}}],
        },
        "chunker": {
            "active": "parent_child",
            "catalog": [
                {
                    "id": "parent_child",
                    "child_tok": 200,
                    "parent_tok": 1000,
                    "overlap_tok": 20,
                }
            ],
        },
        "embedders": {
            "active": "bge_m3_http",
            "catalog": [
                {
                    "id": "bge_m3_http",
                    "kind": "http",
                    "base_url": "http://localhost:8082",
                    "dense_dim": 1024,
                }
            ],
        },
        "rerankers": {
            "active": "bge_v2_m3_http",
            "catalog": [
                {
                    "id": "bge_v2_m3_http",
                    "kind": "http",
                    "base_url": "http://localhost:8083",
                }
            ],
        },
        "term_service": {
            "active": "umls_cmekg_local",
            "catalog": [
                {
                    "id": "umls_cmekg_local",
                    "kind": "local",
                    "data_dir": "data/terminology",
                },
                {"id": "none", "kind": "noop"},
            ],
        },
        "router": {
            "active": "rule_based",
            "catalog": [
                {"id": "rule_based", "max_active": 3, "authority_bias": {1: 0.0}}
            ],
        },
        "system_rag": {
            "default_active": [],
            "collections": [
                {
                    "name": "statpearls_en",
                    "language": "en",
                    "cross_lingual": True,
                    "authority_tier": 1,
                }
            ],
        },
        "user_rag": {"top_k": 5, "rerank_k": 3, "score_threshold": 0.4},
        "qdrant": {"url": "http://localhost:6333", "api_key_env": None},
    }
