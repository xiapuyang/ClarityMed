"""HTTP contract tests for ``BgeM3HttpEmbedder``.

Uses ``httpx.MockTransport`` to intercept requests so no real server is
needed and no new test dependency is added.
"""

from __future__ import annotations

import json

import httpx
import pytest

from claritymed.core.rag.embedding import BgeM3HttpEmbedder, build_embedder
from claritymed.core.rag.schemas import EmbedderConfig
from claritymed.errors import (
    EmbedderUnreachableError,
    MissingApiKeyError,
    UnknownEmbedderError,
)

DENSE_DIM = 1024


def _dense_vec(seed: float = 0.1) -> list[float]:
    return [seed] * DENSE_DIM


def _make_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _embedder(handler, **overrides) -> BgeM3HttpEmbedder:
    kwargs = {
        "base_url": "http://embed.test",
        "dense_dim": DENSE_DIM,
        "transport": _make_transport(handler),
    }
    kwargs.update(overrides)
    return BgeM3HttpEmbedder(**kwargs)


# --- happy path ---------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_dense_returns_one_vector_per_input():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        body = json.loads(request.content)
        assert body == {"inputs": ["a", "b"]}
        return httpx.Response(200, json=[_dense_vec(0.1), _dense_vec(0.2)])

    emb = _embedder(handler)
    out = await emb.embed_dense(["a", "b"])
    assert len(out) == 2
    assert len(out[0]) == DENSE_DIM
    assert out[0][0] == 0.1


@pytest.mark.asyncio
async def test_embed_sparse_flat_dict_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed_sparse"
        return httpx.Response(200, json=[{"7": 0.9, "13": 0.4}])

    emb = _embedder(handler)
    out = await emb.embed_sparse(["hello"])
    assert out == [{7: 0.9, 13: 0.4}]


@pytest.mark.asyncio
async def test_embed_sparse_indices_values_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"indices": [3, 7], "values": [0.8, 0.5]}])

    emb = _embedder(handler)
    out = await emb.embed_sparse(["q"])
    assert out == [{3: 0.8, 7: 0.5}]


@pytest.mark.asyncio
async def test_embed_dense_empty_input_skips_http():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP must not be called for empty input")

    emb = _embedder(handler)
    assert await emb.embed_dense([]) == []
    assert await emb.embed_sparse([]) == []


@pytest.mark.asyncio
async def test_embed_dense_batches_when_over_batch_size():
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        body = json.loads(request.content)
        return httpx.Response(200, json=[_dense_vec(0.1) for _ in body["inputs"]])

    emb = _embedder(handler, batch_size=2)
    out = await emb.embed_dense(["x"] * 5)
    assert call_count["n"] == 3  # 2+2+1
    assert len(out) == 5


# --- fail-loud paths ---------------------------------------------------


@pytest.mark.asyncio
async def test_dense_503_raises_embedder_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="server overloaded")

    emb = _embedder(handler)
    with pytest.raises(EmbedderUnreachableError, match="503"):
        await emb.embed_dense(["a"])


@pytest.mark.asyncio
async def test_dense_connection_error_raises_embedder_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    emb = _embedder(handler)
    with pytest.raises(EmbedderUnreachableError):
        await emb.embed_dense(["a"])


@pytest.mark.asyncio
async def test_dense_dim_mismatch_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[[0.0] * 512])  # wrong dim

    emb = _embedder(handler)
    with pytest.raises(EmbedderUnreachableError, match="dim"):
        await emb.embed_dense(["a"])


@pytest.mark.asyncio
async def test_dense_response_count_mismatch_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[_dense_vec()])  # only 1 for 2 inputs

    emb = _embedder(handler)
    with pytest.raises(EmbedderUnreachableError, match="expected list"):
        await emb.embed_dense(["a", "b"])


@pytest.mark.asyncio
async def test_sparse_invalid_entry_shape_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not a dict"])

    emb = _embedder(handler)
    with pytest.raises(EmbedderUnreachableError):
        await emb.embed_sparse(["q"])


@pytest.mark.asyncio
async def test_api_key_env_unset_raises_missing_api_key():
    with pytest.raises(MissingApiKeyError, match="EMBED_API_KEY"):
        BgeM3HttpEmbedder(
            base_url="http://x",
            dense_dim=DENSE_DIM,
            api_key_env="EMBED_API_KEY",
        )


def test_no_base_url_rejects():
    with pytest.raises(ValueError):
        BgeM3HttpEmbedder(base_url="", dense_dim=DENSE_DIM)


# --- factory -----------------------------------------------------------


def test_build_embedder_factory_happy_path():
    cfg = EmbedderConfig(
        active="bge_m3_http",
        catalog=[
            {  # type: ignore[list-item]
                "id": "bge_m3_http",
                "kind": "http",
                "base_url": "http://embed.test",
                "dense_dim": DENSE_DIM,
            }
        ],
    )
    emb = build_embedder(cfg)
    assert isinstance(emb, BgeM3HttpEmbedder)
    assert emb.dimension == DENSE_DIM


def test_build_embedder_unknown_id_raises():
    """Catalog id the factory has no branch for must fail loud."""
    # Inject by constructing a config with a synthesized entry, bypass
    # validator with model_construct to simulate "added to YAML but
    # factory not yet taught".
    from claritymed.core.rag.schemas import EmbedderEntry, EmbedderConfig

    entry = EmbedderEntry.model_construct(
        id="qwen3_emb_http",
        kind="http",
        base_url="http://x",
        dense_dim=DENSE_DIM,
        batch_size=32,
        timeout_s=30,
        api_key_env=None,
    )
    cfg = EmbedderConfig.model_construct(active="qwen3_emb_http", catalog=[entry])
    with pytest.raises(UnknownEmbedderError):
        build_embedder(cfg)


# --- dimension property ------------------------------------------------


def test_dimension_property():
    emb = BgeM3HttpEmbedder(base_url="http://x", dense_dim=DENSE_DIM)
    assert emb.dimension == DENSE_DIM


# --- aclose / api_key / request_id injection ---------------------------


@pytest.mark.asyncio
async def test_aclose_closes_client_and_allows_reuse():
    """aclose() drains the async client; calling it twice is safe."""
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.path)
        return httpx.Response(200, json=[_dense_vec()])

    emb = _embedder(handler)
    await emb.embed_dense(["a"])  # forces client creation
    assert emb._async_client is not None
    await emb.aclose()
    assert emb._async_client is None
    await emb.aclose()  # second call must not raise


@pytest.mark.asyncio
async def test_api_key_env_set_sends_auth_header(monkeypatch):
    """When api_key_env resolves to a value, Bearer header is sent."""
    monkeypatch.setenv("EMBED_TEST_KEY", "sk-test-token")
    received: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json=[_dense_vec()])

    import httpx as _httpx

    emb = BgeM3HttpEmbedder(
        base_url="http://embed.test",
        dense_dim=DENSE_DIM,
        api_key_env="EMBED_TEST_KEY",
        transport=_httpx.MockTransport(handler),
    )
    await emb.embed_dense(["hello"])
    assert received[0] == "Bearer sk-test-token"
    await emb.aclose()


@pytest.mark.asyncio
async def test_request_id_injected_when_context_var_set():
    """X-Request-ID header is injected from request_id_ctx when set."""
    from claritymed.context import request_id_ctx

    received: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request.headers.get("x-request-id", ""))
        return httpx.Response(200, json=[_dense_vec()])

    emb = _embedder(handler)
    token = request_id_ctx.set("20260615120000ABCDEF12")
    try:
        await emb.embed_dense(["hello"])
    finally:
        request_id_ctx.reset(token)
        await emb.aclose()

    assert received[0] == "20260615120000ABCDEF12"


@pytest.mark.asyncio
async def test_sparse_connection_error_raises_embedder_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    emb = _embedder(handler)
    with pytest.raises(EmbedderUnreachableError):
        await emb.embed_sparse(["a"])
    await emb.aclose()
