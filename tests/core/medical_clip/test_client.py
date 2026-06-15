"""Unit tests for the medical-clip async HTTP client.

Uses ``httpx.MockTransport`` so the client exercises every code path
without binding a port. The transport receives the request, asserts
shape (headers, body), and returns canned responses.
"""

from __future__ import annotations

import base64
import hashlib

import httpx
import pytest

from claritymed.core.medical_clip.client import MedicalClipClient
from claritymed.errors import MedicalClipUnreachableError

_HEX64 = "a" * 64


def _ok_modality_response(request: httpx.Request) -> httpx.Response:
    body = {
        "request_id": "req_42",
        "modality": "ultrasound",
        "confidence": 0.93,
        "is_medical": True,
        "scores": [
            {"label": "ultrasound", "score": 0.93},
            {"label": "ct", "score": 0.04},
            {"label": "xray", "score": 0.02},
            {"label": "dermoscopy", "score": 0.005},
            {"label": "photo", "score": 0.003},
            {"label": "document", "score": 0.002},
        ],
        "elapsed_ms": 48,
    }
    return httpx.Response(200, json=body)


# --- happy path ----------------------------------------------------------


async def test_classify_modality_round_trips_modality_response() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = request.url.path
        captured["headers"] = dict(request.headers)
        captured["body"] = request.content
        return _ok_modality_response(request)

    transport = httpx.MockTransport(handler)
    async with MedicalClipClient(transport=transport) as client:
        resp = await client.classify_modality(
            image_bytes=b"\xff\xd8\xff\xe0fake jpeg",
            request_id="req_42",
        )

    assert resp.modality == "ultrasound"
    assert resp.confidence == pytest.approx(0.93)
    assert resp.is_medical is True
    assert captured["url"] == "/v1/classify_modality"
    assert captured["headers"]["x-request-id"] == "req_42"


async def test_classify_modality_defaults_sha256_to_payload_hash() -> None:
    """Callers that haven't pre-hashed get the digest computed for them."""
    image = b"some bytes"
    expected_sha = hashlib.sha256(image).hexdigest()
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return _ok_modality_response(request)

    transport = httpx.MockTransport(handler)
    async with MedicalClipClient(transport=transport) as client:
        await client.classify_modality(image_bytes=image, request_id="req_1")

    assert captured["body"]["image"]["sha256"] == expected_sha
    assert base64.b64decode(captured["body"]["image"]["data_b64"]) == image


async def test_classify_modality_respects_explicit_sha256() -> None:
    """Callers that store the digest on the blob can skip the re-hash."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return _ok_modality_response(request)

    transport = httpx.MockTransport(handler)
    async with MedicalClipClient(transport=transport) as client:
        await client.classify_modality(
            image_bytes=b"xxx", request_id="req_1", sha256=_HEX64
        )

    assert captured["body"]["image"]["sha256"] == _HEX64


async def test_health_round_trips_health_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "model_id": "microsoft/BiomedCLIP",
                "model_revision": "abc",
                "tasks_loaded": ["modality"],
                "device": "mps",
                "uptime_s": 12,
            },
        )

    transport = httpx.MockTransport(handler)
    async with MedicalClipClient(transport=transport) as client:
        resp = await client.health()
    assert resp.status == "ok"
    assert resp.model_revision == "abc"


# --- failure paths -------------------------------------------------------


async def test_classify_modality_raises_unreachable_on_connect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handler)
    async with MedicalClipClient(transport=transport) as client:
        with pytest.raises(MedicalClipUnreachableError) as exc:
            await client.classify_modality(image_bytes=b"x", request_id="req_1")
        assert "unreachable" in str(exc.value)


async def test_classify_modality_raises_unreachable_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    transport = httpx.MockTransport(handler)
    async with MedicalClipClient(transport=transport) as client:
        with pytest.raises(MedicalClipUnreachableError) as exc:
            await client.classify_modality(image_bytes=b"x", request_id="req_1")
        assert "timeout" in str(exc.value)


async def test_classify_modality_raises_unreachable_on_5xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"code": "boom"}})

    transport = httpx.MockTransport(handler)
    async with MedicalClipClient(transport=transport) as client:
        with pytest.raises(MedicalClipUnreachableError):
            await client.classify_modality(image_bytes=b"x", request_id="req_1")


async def test_classify_modality_propagates_4xx_as_http_status_error() -> None:
    """4xx is an expected application error; let the caller inspect the code."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "image_hash_mismatch",
                    "message": "sha mismatch",
                    "request_id": "req_1",
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with MedicalClipClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError) as exc:
            await client.classify_modality(image_bytes=b"x", request_id="req_1")
        body = exc.value.response.json()
        assert body["error"]["code"] == "image_hash_mismatch"


async def test_aclose_releases_underlying_client() -> None:
    """Use as a non-context-manager to verify aclose paths."""

    client = MedicalClipClient(transport=httpx.MockTransport(_ok_modality_response))
    try:
        resp = await client.classify_modality(image_bytes=b"x", request_id="req_1")
        assert resp.modality == "ultrasound"
    finally:
        await client.aclose()
