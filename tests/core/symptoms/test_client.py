"""Tests for :class:`SymptomsServerClient` using ``httpx.MockTransport``."""

from __future__ import annotations

import json

import httpx
import pytest

from claritymed.core.symptoms.client import SymptomsServerClient
from claritymed.errors import SymptomsServerUnreachableError


def _ok(body: dict) -> httpx.Response:
    return httpx.Response(
        200,
        content=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def _err(status: int, body: dict | None = None) -> httpx.Response:
    payload = body or {"detail": "error"}
    return httpx.Response(
        status,
        content=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def _question_payload() -> dict:
    return {
        "question": "Do you have a fever?",
        "header": "E_91",
        "options": [
            {"label": "Yes", "description": "Yes  ·yes"},
            {"label": "No", "description": "No  ·no"},
        ],
        "multi_select": False,
        "numeric": None,
    }


# --- happy paths ---------------------------------------------------------


@pytest.mark.asyncio
async def test_health_round_trips() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return _ok(
            {"status": "ok", "datasets_loaded": ["ddxplus"], "models_loaded": ["m1"]}
        )

    transport = httpx.MockTransport(handler)
    async with SymptomsServerClient("http://test", transport=transport) as client:
        health = await client.health()
    assert health.status == "ok"
    assert health.datasets_loaded == ["ddxplus"]


@pytest.mark.asyncio
async def test_start_session_posts_payload() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/datasets/ddxplus/sessions"
        captured["body"] = json.loads(request.content)
        return _ok({"session_id": "sid-1", "first_question": _question_payload()})

    transport = httpx.MockTransport(handler)
    async with SymptomsServerClient("http://test", transport=transport) as client:
        resp = await client.start_session(
            "ddxplus",
            "chest pain",
            {"age_years": 45, "sex": "M"},
            language="en",
        )
    assert resp.session_id == "sid-1"
    assert resp.first_question.question == "Do you have a fever?"
    assert captured["body"]["complaint"] == "chest pain"
    assert captured["body"]["profile"]["age_years"] == 45
    assert captured["body"]["language"] == "en"


@pytest.mark.asyncio
async def test_turn_with_raw_answer_value() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/datasets/ddxplus/sessions/sid-1/turn"
        captured["body"] = json.loads(request.content)
        return _ok(
            {
                "done": False,
                "next_question": _question_payload(),
                "turn_count": 1,
                "hit_cap": False,
            }
        )

    transport = httpx.MockTransport(handler)
    async with SymptomsServerClient("http://test", transport=transport) as client:
        resp = await client.turn(
            "ddxplus",
            "sid-1",
            answer="Yes",
            answer_value="yes",
            language="zh",
        )
    assert resp.done is False
    assert captured["body"]["answer"] == "Yes"
    assert captured["body"]["answer_value"] == "yes"
    assert captured["body"]["language"] == "zh"


@pytest.mark.asyncio
async def test_cancel_returns_partial_outcome() -> None:
    body = {
        "cancelled": True,
        "partial_differential": [],
        "evidence_collected": [],
        "turn_count": 2,
        "partial_confidence": 0.3,
        "meets_confidence_threshold": False,
        "severity_override": False,
        "max_low_severity_seen": None,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/v1/datasets/ddxplus/sessions/sid-1"
        return _ok(body)

    transport = httpx.MockTransport(handler)
    async with SymptomsServerClient("http://test", transport=transport) as client:
        resp = await client.cancel("ddxplus", "sid-1")
    assert resp.cancelled is True
    assert resp.partial_confidence == 0.3
    assert resp.severity_override is False


# --- fail-loud transport paths -------------------------------------------


@pytest.mark.asyncio
async def test_connect_error_raises_unreachable() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    async with SymptomsServerClient("http://test", transport=transport) as client:
        with pytest.raises(SymptomsServerUnreachableError, match="unreachable"):
            await client.health()


@pytest.mark.asyncio
async def test_timeout_raises_unreachable() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("server too slow")

    transport = httpx.MockTransport(handler)
    async with SymptomsServerClient("http://test", transport=transport) as client:
        with pytest.raises(SymptomsServerUnreachableError, match="timeout"):
            await client.health()


@pytest.mark.asyncio
async def test_5xx_response_raises_unreachable() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return _err(503, {"detail": "model not loaded"})

    transport = httpx.MockTransport(handler)
    async with SymptomsServerClient("http://test", transport=transport) as client:
        with pytest.raises(SymptomsServerUnreachableError, match="503"):
            await client.health()


@pytest.mark.asyncio
async def test_404_propagates_as_http_status_error() -> None:
    """Application-level 404 (unknown session) is NOT a transport failure —
    let the plugin branch on it via httpx.HTTPStatusError."""

    def handler(_: httpx.Request) -> httpx.Response:
        return _err(404, {"detail": "session expired or unknown"})

    transport = httpx.MockTransport(handler)
    async with SymptomsServerClient("http://test", transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc:
            await client.turn("ddxplus", "missing", answer="Yes")
    assert exc.value.response.status_code == 404


@pytest.mark.asyncio
async def test_422_propagates_as_http_status_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return _err(422, {"detail": "answer did not match values"})

    transport = httpx.MockTransport(handler)
    async with SymptomsServerClient("http://test", transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc:
            await client.turn("ddxplus", "sid-1", answer="garbage")
    assert exc.value.response.status_code == 422


# --- base_url normalization ---------------------------------------------


@pytest.mark.asyncio
async def test_trailing_slash_in_base_url_is_normalized() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return _ok({"status": "ok", "datasets_loaded": [], "models_loaded": []})

    transport = httpx.MockTransport(handler)
    async with SymptomsServerClient("http://test/", transport=transport) as client:
        await client.health()
    # No double-slash before /health.
    assert "//health" not in captured["url"]
