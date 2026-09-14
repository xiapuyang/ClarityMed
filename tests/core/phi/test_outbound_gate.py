"""Unit tests for OutboundTextGate and PhiOutboundGate."""

from __future__ import annotations

import json

import httpx
import pytest

from claritymed.core.phi.outbound_gate import (
    PhiOutboundGate,
    make_outbound_gate,
    resolve_phi_kind,
)
from claritymed.core.rag.embedding.bge_m3 import BgeM3HttpEmbedder
from claritymed.core.rag.reranking.bge_v2_m3 import BgeRerankerV2M3HttpReranker

DENSE_DIM = 1024


# --- helpers ----------------------------------------------------------------


class _StubScrubService:
    """Replaces every non-empty token with '[SCRUBBED]' for deterministic tests."""

    def scrub(self, text: str) -> tuple[str, object]:
        scrubbed = " ".join(
            "[SCRUBBED]" if tok.strip() else tok for tok in text.split(" ")
        )
        return scrubbed, None


def _stub_gate() -> PhiOutboundGate:
    gate = PhiOutboundGate.__new__(PhiOutboundGate)
    gate._scrub = _StubScrubService()  # type: ignore[attr-defined]
    return gate


def _dense_vec(val: float = 0.1) -> list[float]:
    return [val] * DENSE_DIM


# --- PhiOutboundGate --------------------------------------------------------


def test_gate_scrub_delegates_to_service():
    gate = _stub_gate()
    result = gate.scrub("hello world")
    assert result == "[SCRUBBED] [SCRUBBED]"


def test_gate_scrub_batch_maps_each_text():
    gate = _stub_gate()
    results = gate.scrub_batch(["a b", "c d"])
    assert results == ["[SCRUBBED] [SCRUBBED]", "[SCRUBBED] [SCRUBBED]"]


def test_gate_scrub_batch_empty_returns_empty():
    gate = _stub_gate()
    assert gate.scrub_batch([]) == []


def test_make_outbound_gate_returns_none_for_local():
    assert make_outbound_gate("local") is None


def test_make_outbound_gate_returns_none_for_unknown():
    assert make_outbound_gate("") is None


# --- resolve_phi_kind -------------------------------------------------------


@pytest.mark.parametrize(
    "phi_kind, base_url, expected",
    [
        # Explicit config always wins.
        ("local", "http://remote.example.com/", "local"),
        ("cloud", "http://localhost:8080/", "cloud"),
        # None (not configured): auto-detect from URL.
        (None, "http://localhost:8080/", "local"),
        (None, "http://127.0.0.1:8080/", "local"),
        (None, "http://[::1]:8080/", "local"),
        (None, "http://embed.internal.company.com/", "cloud"),
        (None, "https://api.huggingface.co/", "cloud"),
    ],
)
def test_resolve_phi_kind(phi_kind, base_url, expected):
    assert resolve_phi_kind(phi_kind, base_url) == expected


# --- BgeM3HttpEmbedder with scrub_gate ------------------------------------


@pytest.mark.asyncio
async def test_embedder_scrubs_texts_before_post_when_gate_set():
    received_inputs: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        received_inputs.append(body["inputs"])
        return httpx.Response(200, json=[_dense_vec(0.5)])

    gate = _stub_gate()
    emb = BgeM3HttpEmbedder(
        base_url="http://embed.test",
        dense_dim=DENSE_DIM,
        transport=httpx.MockTransport(handler),
        scrub_gate=gate,
    )
    await emb.embed_dense(["hello world"])
    assert received_inputs == [["[SCRUBBED] [SCRUBBED]"]]


@pytest.mark.asyncio
async def test_embedder_no_scrub_when_gate_is_none():
    received_inputs: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        received_inputs.append(body["inputs"])
        return httpx.Response(200, json=[_dense_vec(0.5)])

    emb = BgeM3HttpEmbedder(
        base_url="http://embed.test",
        dense_dim=DENSE_DIM,
        transport=httpx.MockTransport(handler),
        scrub_gate=None,
    )
    await emb.embed_dense(["hello world"])
    assert received_inputs == [["hello world"]]


@pytest.mark.asyncio
async def test_embedder_scrubs_sparse_texts():
    received_inputs: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        received_inputs.append(body["inputs"])
        return httpx.Response(200, json=[{"0": 0.9}])

    gate = _stub_gate()
    emb = BgeM3HttpEmbedder(
        base_url="http://embed.test",
        dense_dim=DENSE_DIM,
        transport=httpx.MockTransport(handler),
        scrub_gate=gate,
    )
    await emb.embed_sparse(["patient name"])
    assert received_inputs == [["[SCRUBBED] [SCRUBBED]"]]


# --- BgeRerankerV2M3HttpReranker with scrub_gate ---------------------------


@pytest.mark.asyncio
async def test_reranker_scrubs_query_and_docs_before_post():
    received: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return httpx.Response(200, json=[{"index": 0, "score": 0.9}])

    gate = _stub_gate()
    rr = BgeRerankerV2M3HttpReranker(
        base_url="http://rerank.test",
        transport=httpx.MockTransport(handler),
        scrub_gate=gate,
    )
    await rr.rerank("patient allergies", ["doc one"], top_k=1)
    assert received[0]["query"] == "[SCRUBBED] [SCRUBBED]"
    assert received[0]["texts"] == ["[SCRUBBED] [SCRUBBED]"]


@pytest.mark.asyncio
async def test_reranker_no_scrub_when_gate_is_none():
    received: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return httpx.Response(200, json=[{"index": 0, "score": 0.9}])

    rr = BgeRerankerV2M3HttpReranker(
        base_url="http://rerank.test",
        transport=httpx.MockTransport(handler),
        scrub_gate=None,
    )
    await rr.rerank("patient allergies", ["doc one"], top_k=1)
    assert received[0]["query"] == "patient allergies"
    assert received[0]["texts"] == ["doc one"]


# --- LLMTranslationProvider with scrub_gate --------------------------------


@pytest.mark.asyncio
async def test_translation_provider_scrubs_input_before_agent_call():
    """Gate scrubs text before it reaches the pydantic-ai Agent."""
    from contextlib import contextmanager
    from unittest.mock import AsyncMock, MagicMock, patch

    gate = _stub_gate()

    from claritymed.core.translation.llm_provider import LLMTranslationProvider

    provider = LLMTranslationProvider(model=MagicMock(), scrub_gate=gate)

    mock_result = MagicMock()
    mock_result.output = "translated"
    captured_inputs: list[str] = []

    async def fake_run(text, *args, **kwargs):
        captured_inputs.append(text)
        return mock_result

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=fake_run)

    @contextmanager
    def fake_step(*args, **kwargs):
        s = MagicMock()
        yield s

    with patch("pydantic_ai.Agent", return_value=mock_agent):
        with patch("claritymed.core.prompts.registry.get_default_registry") as mock_reg:
            mock_reg.return_value.get.return_value = "system prompt"
            with patch("claritymed.core.observability.steps.step", fake_step):
                await provider.translate("hello world", target_lang="zh")

    assert captured_inputs == ["[SCRUBBED] [SCRUBBED]"]


@pytest.mark.asyncio
async def test_translation_provider_no_scrub_when_gate_is_none():
    from contextlib import contextmanager
    from unittest.mock import AsyncMock, MagicMock, patch

    from claritymed.core.translation.llm_provider import LLMTranslationProvider

    provider = LLMTranslationProvider(model=MagicMock(), scrub_gate=None)

    mock_result = MagicMock()
    mock_result.output = "translated"
    captured_inputs: list[str] = []

    async def fake_run(text, *args, **kwargs):
        captured_inputs.append(text)
        return mock_result

    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(side_effect=fake_run)

    @contextmanager
    def fake_step(*args, **kwargs):
        s = MagicMock()
        yield s

    with patch("pydantic_ai.Agent", return_value=mock_agent):
        with patch("claritymed.core.prompts.registry.get_default_registry") as mock_reg:
            mock_reg.return_value.get.return_value = "system prompt"
            with patch("claritymed.core.observability.steps.step", fake_step):
                await provider.translate("hello world", target_lang="zh")

    assert captured_inputs == ["hello world"]
