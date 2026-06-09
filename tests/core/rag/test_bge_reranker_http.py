"""HTTP contract tests for ``BgeRerankerV2M3HttpReranker``."""

from __future__ import annotations

import json

import httpx
import pytest

from claritymed.core.rag.reranking import (
    BgeRerankerV2M3HttpReranker,
    RerankHit,
    build_reranker,
)
from claritymed.core.rag.schemas import RerankerConfig
from claritymed.errors import (
    MissingApiKeyError,
    RerankerUnreachableError,
    UnknownRerankerError,
)


def _reranker(handler, **overrides) -> BgeRerankerV2M3HttpReranker:
    kwargs = {
        "base_url": "http://rerank.test",
        "transport": httpx.MockTransport(handler),
    }
    kwargs.update(overrides)
    return BgeRerankerV2M3HttpReranker(**kwargs)


# --- happy path ---------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_returns_top_k_sorted_desc():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank"
        body = json.loads(request.content)
        assert body["query"] == "headache"
        assert body["texts"] == ["a", "b", "c"]
        return httpx.Response(
            200,
            json=[
                {"index": 1, "score": 0.95},
                {"index": 0, "score": 0.55},
                {"index": 2, "score": 0.10},
            ],
        )

    rr = _reranker(handler)
    hits = await rr.rerank("headache", ["a", "b", "c"], top_k=2)
    assert hits == [RerankHit(index=1, score=0.95), RerankHit(index=0, score=0.55)]


@pytest.mark.asyncio
async def test_rerank_empty_docs_returns_empty_without_http():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP must not be called for empty docs")

    rr = _reranker(handler)
    assert await rr.rerank("q", [], top_k=5) == []


@pytest.mark.asyncio
async def test_rerank_batches_and_merges_indices():
    """Caller-space index translation across batches."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # Each batch returns local indices in arbitrary order; index 0
        # within the batch corresponds to (offset + 0) caller-space.
        result = [
            {"index": i, "score": 0.5 + i * 0.01} for i in range(len(body["texts"]))
        ]
        return httpx.Response(200, json=result)

    rr = _reranker(handler, batch_size=2)
    docs = ["d0", "d1", "d2", "d3", "d4"]
    hits = await rr.rerank("q", docs, top_k=5)
    # Highest score wins; with our handler, score = 0.5 + local_idx * 0.01.
    # Batches: [d0, d1] -> local 0,1 mapped to global 0,1
    #          [d2, d3] -> local 0,1 mapped to global 2,3
    #          [d4]     -> local 0 mapped to global 4
    # Scores: d0=.50 d1=.51 d2=.50 d3=.51 d4=.50
    # Top 5 sorted desc: any tiebreak by index is acceptable, but score
    # ordering must hold.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert {h.index for h in hits} == {0, 1, 2, 3, 4}


# --- fail-loud paths ---------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_500_raises_reranker_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    rr = _reranker(handler)
    with pytest.raises(RerankerUnreachableError, match="500"):
        await rr.rerank("q", ["a"], top_k=1)


@pytest.mark.asyncio
async def test_rerank_connection_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    rr = _reranker(handler)
    with pytest.raises(RerankerUnreachableError):
        await rr.rerank("q", ["a"], top_k=1)


@pytest.mark.asyncio
async def test_rerank_response_not_list_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"oops": True})

    rr = _reranker(handler)
    with pytest.raises(RerankerUnreachableError):
        await rr.rerank("q", ["a"], top_k=1)


@pytest.mark.asyncio
async def test_rerank_index_out_of_range_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"index": 99, "score": 0.9}])

    rr = _reranker(handler)
    with pytest.raises(RerankerUnreachableError, match="out of range"):
        await rr.rerank("q", ["a"], top_k=1)


@pytest.mark.asyncio
async def test_rerank_missing_index_field_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"score": 0.9}])

    rr = _reranker(handler)
    with pytest.raises(RerankerUnreachableError, match="invalid"):
        await rr.rerank("q", ["a"], top_k=1)


@pytest.mark.asyncio
async def test_rerank_top_k_zero_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    rr = _reranker(handler)
    with pytest.raises(ValueError):
        await rr.rerank("q", ["a"], top_k=0)


def test_api_key_env_unset_raises_missing_api_key():
    with pytest.raises(MissingApiKeyError):
        BgeRerankerV2M3HttpReranker(
            base_url="http://x",
            api_key_env="RERANK_KEY",
        )


def test_no_base_url_rejects():
    with pytest.raises(ValueError):
        BgeRerankerV2M3HttpReranker(base_url="")


# --- factory ----------------------------------------------------------


def test_build_reranker_factory_happy_path():
    cfg = RerankerConfig(
        active="bge_v2_m3_http",
        catalog=[
            {  # type: ignore[list-item]
                "id": "bge_v2_m3_http",
                "kind": "http",
                "base_url": "http://rerank.test",
            }
        ],
    )
    rr = build_reranker(cfg)
    assert isinstance(rr, BgeRerankerV2M3HttpReranker)


def test_build_reranker_factory_v2_gemma_uses_same_client():
    """v2-gemma shares the TEI /rerank wire format with v2-m3, so the
    same async HTTP client serves it — only the server-side model and
    optional query instruction prefix differ."""
    cfg = RerankerConfig(
        active="bge_v2_gemma_http",
        catalog=[
            {  # type: ignore[list-item]
                "id": "bge_v2_gemma_http",
                "kind": "http",
                "base_url": "http://rerank.test",
                "batch_size": 8,
            }
        ],
    )
    rr = build_reranker(cfg)
    assert isinstance(rr, BgeRerankerV2M3HttpReranker)


def test_build_reranker_unknown_id_raises():
    from claritymed.core.rag.schemas import RerankerConfig, RerankerEntry

    entry = RerankerEntry.model_construct(
        id="qwen3_rerank_http",
        kind="http",
        base_url="http://x",
        batch_size=32,
        timeout_s=30,
        api_key_env=None,
    )
    cfg = RerankerConfig.model_construct(active="qwen3_rerank_http", catalog=[entry])
    with pytest.raises(UnknownRerankerError):
        build_reranker(cfg)
