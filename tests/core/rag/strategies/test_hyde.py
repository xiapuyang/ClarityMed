"""Unit tests for ``HydeStrategy`` — Hypothetical Document Embeddings."""

from __future__ import annotations

from typing import Any

import pytest

from claritymed.core.rag.schemas import (
    EvidenceBundle,
    HydeStrategyConfig,
    NaiveHybridStrategyConfig,
    RetrievalTrace,
    StrategiesConfig,
)
from claritymed.core.rag.strategies import (
    HydeStrategy,
    RagStrategy,
    RetrievalContext,
    build_strategy,
)
from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.errors import UnknownStrategyError


class _FakeRetriever:
    """Records every ``retrieve`` call so tests can assert composed inputs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._bundle = EvidenceBundle(
            chunks=[],
            trace=RetrievalTrace(strategy="naive_hybrid", embed_ms=12),
        )

    def feed(self, bundle: EvidenceBundle) -> None:
        self._bundle = bundle

    async def retrieve(
        self,
        query: str,
        *,
        language: str,
        user_id: str,
        user_whitelist: list[str] | None = None,
        only_cloud_safe: bool = False,
        embedding_query_override: str | None = None,
    ) -> EvidenceBundle:
        self.calls.append(
            {
                "query": query,
                "language": language,
                "user_id": user_id,
                "user_whitelist": user_whitelist,
                "only_cloud_safe": only_cloud_safe,
                "embedding_query_override": embedding_query_override,
            }
        )
        return self._bundle


def _chunk(*, doc_id: str = "d0", text: str = "t") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source="system_rag",
        score=1.0,
        doc_id=doc_id,
        chunk_index=0,
        is_phi=False,
        can_cloud=True,
        collection_name="statpearls_en",
        parent_id=f"{doc_id}#p0",
        rerank_score=0.8,
    )


def _bundle(chunks: list[RetrievedChunk]) -> EvidenceBundle:
    return EvidenceBundle(
        chunks=chunks,
        trace=RetrievalTrace(strategy="naive_hybrid", embed_ms=10),
    )


def _ctx(
    query: str = "iron deficiency anemia", language: str = "en"
) -> RetrievalContext:
    return RetrievalContext(
        query=query,
        user_id="u1",
        language=language,  # type: ignore[arg-type]
    )


# --- protocol + happy path ---------------------------------------------


async def test_hyde_implements_strategy_protocol():
    async def hyp(q, lang):
        return "hypothetical passage"

    strategy = HydeStrategy(
        retriever=_FakeRetriever(),  # type: ignore[arg-type]
        config=HydeStrategyConfig(id="hyde"),
        hypothesizer=hyp,
    )
    assert isinstance(strategy, RagStrategy)


async def test_hyde_passes_hypothetical_doc_to_retriever_as_embedding_override():
    async def hyp(q, lang):
        return "Iron deficiency anemia is the most common cause of anemia worldwide..."

    fake = _FakeRetriever()
    fake.feed(_bundle([_chunk()]))
    strategy = HydeStrategy(
        retriever=fake,  # type: ignore[arg-type]
        config=HydeStrategyConfig(id="hyde", include_original=False),
        hypothesizer=hyp,
    )
    bundle = await strategy.retrieve(_ctx("what causes anemia"))
    assert len(fake.calls) == 1
    call = fake.calls[0]
    # The user's original question stays as ``query`` (so rerank uses it),
    # the hypothetical lands in the embedding override.
    assert call["query"] == "what causes anemia"
    assert call["embedding_query_override"].startswith("Iron deficiency anemia")
    assert "what causes anemia" not in call["embedding_query_override"]
    assert bundle.trace.strategy == "hyde"
    assert bundle.trace.hyde_fallback is False


async def test_hyde_include_original_concatenates_both():
    async def hyp(q, lang):
        return "Hepcidin regulates iron absorption."

    fake = _FakeRetriever()
    fake.feed(_bundle([_chunk()]))
    strategy = HydeStrategy(
        retriever=fake,  # type: ignore[arg-type]
        config=HydeStrategyConfig(id="hyde", include_original=True),
        hypothesizer=hyp,
    )
    await strategy.retrieve(_ctx("how does hepcidin work"))
    override = fake.calls[0]["embedding_query_override"]
    assert "Hepcidin regulates iron absorption." in override
    assert "how does hepcidin work" in override


async def test_hyde_fail_soft_on_llm_error_uses_original_query():
    async def hyp(q, lang):
        raise RuntimeError("LLM down")

    fake = _FakeRetriever()
    fake.feed(_bundle([_chunk()]))
    strategy = HydeStrategy(
        retriever=fake,  # type: ignore[arg-type]
        config=HydeStrategyConfig(id="hyde"),
        hypothesizer=hyp,
    )
    bundle = await strategy.retrieve(_ctx("anemia"))
    # No embedding override → retriever falls back to its own term expansion.
    assert fake.calls[0]["embedding_query_override"] is None
    assert bundle.trace.hyde_fallback is True
    assert bundle.trace.strategy == "hyde"


async def test_hyde_fail_soft_on_empty_hypothetical():
    async def hyp(q, lang):
        return "   "

    fake = _FakeRetriever()
    fake.feed(_bundle([_chunk()]))
    strategy = HydeStrategy(
        retriever=fake,  # type: ignore[arg-type]
        config=HydeStrategyConfig(id="hyde"),
        hypothesizer=hyp,
    )
    bundle = await strategy.retrieve(_ctx("q"))
    assert fake.calls[0]["embedding_query_override"] is None
    assert bundle.trace.hyde_fallback is True


async def test_hyde_caps_chunks_to_max_evidence():
    async def hyp(q, lang):
        return "anemia text"

    fake = _FakeRetriever()
    fake.feed(_bundle([_chunk(doc_id=f"d{i}") for i in range(5)]))
    strategy = HydeStrategy(
        retriever=fake,  # type: ignore[arg-type]
        config=HydeStrategyConfig(id="hyde"),
        hypothesizer=hyp,
        max_evidence=2,
    )
    bundle = await strategy.retrieve(_ctx())
    assert len(bundle.chunks) == 2


# --- factory wiring ----------------------------------------------------


def test_hyde_requires_either_model_or_hypothesizer():
    with pytest.raises(ValueError, match="hypothesizer"):
        HydeStrategy(
            retriever=_FakeRetriever(),  # type: ignore[arg-type]
            config=HydeStrategyConfig(id="hyde"),
        )


def test_build_strategy_hyde_returns_hyde_strategy():
    cfg = StrategiesConfig(
        active="hyde",
        catalog=[
            HydeStrategyConfig(id="hyde", include_original=False),
            NaiveHybridStrategyConfig(id="naive_hybrid"),
        ],
    )

    class _SentinelModel:
        pass

    strategy = build_strategy(
        _FakeRetriever(),  # type: ignore[arg-type]
        config=cfg,
        model=_SentinelModel(),  # type: ignore[arg-type]
    )
    assert isinstance(strategy, HydeStrategy)


def test_build_strategy_hyde_without_model_raises():
    cfg = StrategiesConfig(
        active="hyde",
        catalog=[
            HydeStrategyConfig(id="hyde"),
            NaiveHybridStrategyConfig(id="naive_hybrid"),
        ],
    )
    with pytest.raises(ValueError, match="Model"):
        build_strategy(_FakeRetriever(), config=cfg)  # type: ignore[arg-type]


def test_build_strategy_unknown_id_raises():
    # Skip ``_resolve_active`` validator via ``model_construct`` and point
    # at an entry whose id has no factory branch yet.
    cfg = StrategiesConfig.model_construct(
        active="naive_hybrid",
        catalog=[NaiveHybridStrategyConfig.model_construct(id="brand_new")],
    )
    with pytest.raises(UnknownStrategyError):
        build_strategy(_FakeRetriever(), config=cfg)  # type: ignore[arg-type]
