"""Unit 7: NaiveHybridStrategy + CRAG-lite grader paths."""

from __future__ import annotations

from typing import Any

import pytest

from claritymed.core.rag.schemas import (
    EvidenceBundle,
    GraderConfig,
    NaiveHybridStrategyConfig,
    RetrievalTrace,
    StrategiesConfig,
)
from claritymed.core.rag.strategies import (
    NaiveHybridStrategy,
    RagStrategy,
    RetrievalContext,
    build_strategy,
)
from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.errors import UnknownStrategyError


class FakeRetriever:
    """Programmable HybridRetriever stand-in for strategy tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._queue: list[EvidenceBundle] = []

    def feed(self, bundle: EvidenceBundle) -> None:
        self._queue.append(bundle)

    async def retrieve(
        self,
        query: str,
        *,
        language: str,
        user_id: str,
        user_whitelist: list[str] | None = None,
        only_cloud_safe: bool = False,
    ) -> EvidenceBundle:
        self.calls.append(query)
        if not self._queue:
            raise AssertionError(
                f"FakeRetriever has no more bundles queued (call #{len(self.calls)})"
            )
        return self._queue.pop(0)


def _chunk(
    *,
    doc_id: str,
    chunk_index: int = 0,
    collection: str = "statpearls_en",
    rerank_score: float | None = None,
    text: str = "text",
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source="system_rag",
        score=1.0,
        doc_id=doc_id,
        chunk_index=chunk_index,
        is_phi=False,
        can_cloud=True,
        collection_name=collection,
        parent_id=f"{doc_id}#p{chunk_index}",
        rerank_score=rerank_score,
    )


def _bundle(chunks: list[RetrievedChunk], **trace_kwargs: Any) -> EvidenceBundle:
    return EvidenceBundle(
        chunks=chunks,
        trace=RetrievalTrace(strategy="naive_hybrid", **trace_kwargs),
    )


def _ctx(query: str = "aspirin side effects", language: str = "en") -> RetrievalContext:
    return RetrievalContext(query=query, user_id="alice", language=language)


# --- protocol ----------------------------------------------------------


def test_strategy_implements_protocol():
    r = FakeRetriever()
    s = NaiveHybridStrategy(retriever=r)  # type: ignore[arg-type]
    assert isinstance(s, RagStrategy)


# --- grader OFF -------------------------------------------------------


async def test_grader_off_returns_first_pass_unchanged():
    r = FakeRetriever()
    r.feed(_bundle([_chunk(doc_id="d1")]))
    s = NaiveHybridStrategy(retriever=r)  # type: ignore[arg-type]
    bundle = await s.retrieve(_ctx())
    assert r.calls == ["aspirin side effects"]
    assert bundle.trace.fallback_triggered is False
    assert bundle.trace.grader is None
    assert len(bundle.chunks) == 1


# --- grader ON, pass --------------------------------------------------


async def test_grader_on_high_score_passes_no_second_pass():
    r = FakeRetriever()
    r.feed(_bundle([_chunk(doc_id=f"d{i}", rerank_score=0.9) for i in range(3)]))
    s = NaiveHybridStrategy(
        retriever=r,  # type: ignore[arg-type]
        grader=GraderConfig(enabled=True, threshold=0.5, top_n=3),
    )
    bundle = await s.retrieve(_ctx())
    assert len(r.calls) == 1
    assert bundle.trace.grader is not None
    assert bundle.trace.grader.decision == "pass"
    assert bundle.trace.fallback_triggered is False


# --- grader ON, low score, rewrite --------------------------------


async def test_grader_low_score_rewrites_when_stopwords_present():
    r = FakeRetriever()
    r.feed(_bundle([_chunk(doc_id="d1", rerank_score=0.1, text="low")]))
    r.feed(_bundle([_chunk(doc_id="d2", rerank_score=0.8, text="high")]))
    s = NaiveHybridStrategy(
        retriever=r,  # type: ignore[arg-type]
        grader=GraderConfig(enabled=True, threshold=0.5, top_n=1),
    )
    bundle = await s.retrieve(
        RetrievalContext(
            query="what are the side effects of aspirin",
            user_id="alice",
            language="en",
        )
    )
    assert len(r.calls) == 2
    assert r.calls[0] == "what are the side effects of aspirin"
    # Rewrite drops stopwords: what/are/the/of
    assert r.calls[1] == "side effects aspirin"
    assert bundle.trace.fallback_triggered is True
    assert bundle.trace.grader is not None
    assert bundle.trace.grader.decision == "rewrite"
    # Both chunks should appear, sorted by rerank score desc.
    assert bundle.chunks[0].rerank_score == 0.8


# --- rewrite no-op ----------------------------------------------------


async def test_rewrite_no_op_skips_second_pass_but_marks_decision():
    """If rewrite would equal the original (no stopwords / pure keywords),
    skip the second retrieval but still mark trace.fallback_triggered."""
    r = FakeRetriever()
    r.feed(_bundle([_chunk(doc_id="d1", rerank_score=0.1)]))
    s = NaiveHybridStrategy(
        retriever=r,  # type: ignore[arg-type]
        grader=GraderConfig(enabled=True, threshold=0.5, top_n=1),
    )
    bundle = await s.retrieve(_ctx(query="aspirin"))
    assert len(r.calls) == 1
    assert bundle.trace.fallback_triggered is True
    assert bundle.trace.grader is not None
    assert bundle.trace.grader.decision == "rewrite"


# --- merge dedup -----------------------------------------------------


async def test_merge_dedup_keeps_higher_rerank_score():
    r = FakeRetriever()
    r.feed(_bundle([_chunk(doc_id="dup", rerank_score=0.2, text="v_low")]))
    r.feed(_bundle([_chunk(doc_id="dup", rerank_score=0.9, text="v_high")]))
    s = NaiveHybridStrategy(
        retriever=r,  # type: ignore[arg-type]
        grader=GraderConfig(enabled=True, threshold=0.5, top_n=1),
    )
    bundle = await s.retrieve(
        RetrievalContext(query="what is the dosage", user_id="alice", language="en")
    )
    # One unique key (collection,doc_id,chunk_index) → one chunk
    assert len(bundle.chunks) == 1
    assert bundle.chunks[0].rerank_score == 0.9
    assert bundle.chunks[0].text == "v_high"


# --- max_evidence cap ------------------------------------------------


async def test_max_evidence_caps_returned_chunks():
    r = FakeRetriever()
    r.feed(
        _bundle(
            [_chunk(doc_id=f"d{i}", rerank_score=0.9 - i * 0.01) for i in range(10)]
        )
    )
    s = NaiveHybridStrategy(retriever=r, max_evidence=3)  # type: ignore[arg-type]
    bundle = await s.retrieve(_ctx())
    assert len(bundle.chunks) == 3


# --- chinese path -----------------------------------------------------


async def test_chinese_rewrite_collapses_whitespace_only():
    r = FakeRetriever()
    r.feed(_bundle([_chunk(doc_id="d1", rerank_score=0.1)]))
    r.feed(_bundle([_chunk(doc_id="d2", rerank_score=0.8)]))
    s = NaiveHybridStrategy(
        retriever=r,  # type: ignore[arg-type]
        grader=GraderConfig(enabled=True, threshold=0.5, top_n=1),
    )
    await s.retrieve(_ctx(query="阿司匹林  副作用", language="zh"))
    assert r.calls[0] == "阿司匹林  副作用"
    # Rewrite collapses double space.
    assert r.calls[1] == "阿司匹林 副作用"


# --- LLM rewrite NotImplemented in v1 --------------------------------


async def test_llm_rewrite_mode_raises():
    r = FakeRetriever()
    r.feed(_bundle([_chunk(doc_id="d1", rerank_score=0.1)]))
    s = NaiveHybridStrategy(
        retriever=r,  # type: ignore[arg-type]
        grader=GraderConfig(enabled=True, threshold=0.5, top_n=1, rewrite_mode="llm"),
    )
    with pytest.raises(NotImplementedError, match="llm"):
        await s.retrieve(
            RetrievalContext(
                query="what is aspirin used for",
                user_id="alice",
                language="en",
            )
        )


# --- empty chunks edge case -----------------------------------------


async def test_grader_handles_empty_first_pass():
    """No chunks → mean score 0 → triggers rewrite path."""
    r = FakeRetriever()
    r.feed(_bundle([]))
    r.feed(_bundle([_chunk(doc_id="d1", rerank_score=0.9)]))
    s = NaiveHybridStrategy(
        retriever=r,  # type: ignore[arg-type]
        grader=GraderConfig(enabled=True, threshold=0.5, top_n=3),
    )
    bundle = await s.retrieve(_ctx(query="what is aspirin"))
    assert len(r.calls) == 2
    assert bundle.chunks[0].doc_id == "d1"
    assert bundle.trace.fallback_triggered is True


# --- factory ---------------------------------------------------------


def test_build_strategy_factory_happy_path():
    class _Dummy:
        async def retrieve(self, *a, **k):
            raise NotImplementedError

    cfg = StrategiesConfig(
        active="naive_hybrid",
        catalog=[
            NaiveHybridStrategyConfig(
                id="naive_hybrid",
                grader=GraderConfig(enabled=True),
            )
        ],
    )
    s = build_strategy(_Dummy(), cfg)  # type: ignore[arg-type]
    assert isinstance(s, NaiveHybridStrategy)


def test_build_strategy_unknown_id_raises():
    class _Dummy:
        async def retrieve(self, *a, **k):
            raise NotImplementedError

    class FakeFutureConfig:
        id = "graph"

    # Use model_construct to synthesize a still-reserved id ("graph")
    # the factory has no branch for yet — pydantic's own validator would
    # reject it because the catalog union does not include GraphConfig.
    cfg = StrategiesConfig.model_construct(active="graph", catalog=[FakeFutureConfig()])
    with pytest.raises(UnknownStrategyError):
        build_strategy(_Dummy(), cfg)  # type: ignore[arg-type]


def test_build_strategy_agentic_returns_naive_hybrid_with_flag():
    """Agentic mode reuses NaiveHybridStrategy under the hood — the
    factory just tags the instance so AskService can detect the rollout."""
    from claritymed.core.rag.schemas import AgenticStrategyConfig

    class _Dummy:
        async def retrieve(self, *a, **k):
            raise NotImplementedError

    cfg = StrategiesConfig(
        active="agentic",
        catalog=[
            AgenticStrategyConfig(id="agentic"),
            NaiveHybridStrategyConfig(id="naive_hybrid"),
        ],
    )
    s = build_strategy(_Dummy(), cfg)  # type: ignore[arg-type]
    assert isinstance(s, NaiveHybridStrategy)
    assert getattr(s, "is_agentic", False) is True
