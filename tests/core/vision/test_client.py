"""Unit tests for the vision async HTTP client.

Uses ``httpx.MockTransport`` so the client exercises every code path
without binding a port. The transport receives the request, asserts
shape (headers, body), and returns canned responses.
"""

from __future__ import annotations

import hashlib

import httpx
import pytest

from claritymed.core.vision.client import VisionHttpClient
from claritymed.errors import VisionServerUnreachableError


_HEX64 = "a" * 64


def _ok_health(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "status": "ok",
            "models_loaded": [
                {
                    "disease_id": "breast_cancer_ultrasound",
                    "model_id": "breast_busi_unet_v1",
                }
            ],
            "uptime_s": 42,
        },
    )


def _ok_catalog(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "models": [
                {
                    "disease_id": "breast_cancer_ultrasound",
                    "model_id": "breast_busi_unet_v1",
                    "model_version": "v1.0.0",
                    "framework": "pytorch",
                    "task": "classification+segmentation",
                    "labels": ["benign", "malignant", "normal"],
                    "cancer_class": True,
                    "accepted_modality": "ultrasound",
                    "manifest_sha": _HEX64,
                    "expected_ms": 800,
                    "supports_saliency": False,
                    "supports_tta": True,
                }
            ]
        },
    )


def _ok_detect(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "request_id": "req_42",
            "disease_id": "breast_cancer_ultrasound",
            "model_id": "breast_busi_unet_v1",
            "model_version": "v1.0.0",
            "elapsed_ms": 412,
            "input_quality": {
                "passed": True,
                "checks": [{"name": "modality_match", "score": 0.97, "passed": True}],
            },
            "classification": {
                "labels": ["benign", "malignant", "normal"],
                "probabilities": [0.12, 0.81, 0.07],
                "top1": "malignant",
                "top1_prob": 0.81,
                "confidence_tier": "high",
            },
            "cancer_status": "malignant",
            "clinical_action": "urgent_specialist",
            "labels_meta": {
                "benign": {
                    "description": "benign mass",
                    "cancer_status": "benign",
                    "clinical_action": "routine_followup",
                },
                "malignant": {
                    "description": "malignant lesion",
                    "cancer_status": "malignant",
                    "clinical_action": "urgent_specialist",
                },
                "normal": {
                    "description": "no lesion",
                    "cancer_status": "normal",
                    "clinical_action": "no_action",
                },
            },
            "warnings": [],
            "model_card_url": None,
        },
    )


# --- happy path ----------------------------------------------------------


async def test_health_round_trips() -> None:
    transport = httpx.MockTransport(_ok_health)
    async with VisionHttpClient("http://127.0.0.1:8085", transport=transport) as c:
        resp = await c.health()
    assert resp.status == "ok"
    assert resp.uptime_s == 42


async def test_catalog_round_trips() -> None:
    transport = httpx.MockTransport(_ok_catalog)
    async with VisionHttpClient("http://127.0.0.1:8085", transport=transport) as c:
        resp = await c.catalog()
    assert len(resp.models) == 1
    entry = resp.models[0]
    assert entry.model_id == "breast_busi_unet_v1"
    assert entry.cancer_class is True


async def test_detect_round_trips_with_request_id_header() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = request.url.path
        captured["headers"] = dict(request.headers)
        captured["body"] = request.content
        return _ok_detect(request)

    transport = httpx.MockTransport(handler)
    async with VisionHttpClient("http://127.0.0.1:8085", transport=transport) as c:
        resp = await c.detect(
            request_id="req_42",
            disease_id="breast_cancer_ultrasound",
            model_id=None,
            image_bytes=b"\xff\xd8fake",
        )
    assert resp.classification.top1 == "malignant"
    assert captured["url"] == "/v1/detect"
    assert captured["headers"]["x-request-id"] == "req_42"
    # Body carried the sha so the server can re-verify.
    expected_sha = hashlib.sha256(b"\xff\xd8fake").hexdigest()
    assert expected_sha in captured["body"].decode("utf-8")


# --- error paths ---------------------------------------------------------


async def test_5xx_raises_vision_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    async with VisionHttpClient("http://127.0.0.1:8085", transport=transport) as c:
        with pytest.raises(VisionServerUnreachableError) as ei:
            await c.detect(
                request_id="req_42",
                disease_id="breast_cancer_ultrasound",
                model_id=None,
                image_bytes=b"x",
            )
    assert "500" in str(ei.value)


async def test_connect_error_raises_vision_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = httpx.MockTransport(handler)
    async with VisionHttpClient("http://127.0.0.1:8085", transport=transport) as c:
        with pytest.raises(VisionServerUnreachableError) as ei:
            await c.health()
    assert "unreachable" in str(ei.value)


async def test_4xx_propagates_as_http_status_error_so_caller_can_branch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # The shape mirrors the server's uniform error envelope so a
        # real caller could decode the code field; the test only needs
        # status_code+body propagation.
        return httpx.Response(
            422,
            json={
                "error": {
                    "code": "modality_mismatch",
                    "message": "model accepts ultrasound, image is ct",
                    "request_id": "req_42",
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with VisionHttpClient("http://127.0.0.1:8085", transport=transport) as c:
        with pytest.raises(httpx.HTTPStatusError) as ei:
            await c.detect(
                request_id="req_42",
                disease_id="breast_cancer_ultrasound",
                model_id=None,
                image_bytes=b"x",
            )
    err_body = ei.value.response.json()
    assert err_body["error"]["code"] == "modality_mismatch"
