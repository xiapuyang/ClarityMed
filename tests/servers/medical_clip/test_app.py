"""End-to-end-ish tests for the medical-clip FastAPI app.

Uses ``fastapi.TestClient`` so the app runs in-process — no socket
binding, no subprocess. ``_state["engine"]`` is replaced with a stub
that returns deterministic scores so the lifecycle (health → classify →
gating → error envelopes) can be exercised without loading BiomedCLIP.

The lifespan is bypassed via ``CLARITYMED_MEDICAL_CLIP_SKIP_LOAD=1``
which the conftest at the bottom of this file sets per-test.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sys
from typing import Any

import pytest

from claritymed.core.medical_clip.schemas import Modality, ModalityScore
from claritymed.servers.medical_clip.app import (
    HOST,
    _NON_MEDICAL_LABELS,
    _apply_gating,
    app,
)
from claritymed.servers.medical_clip.biomed_clip import ImageDecodeError

# ``import claritymed.servers.medical_clip.app as app_mod`` falls afoul
# of attribute access (the FastAPI instance shadows the submodule), so
# resolve through sys.modules — same shape as the symptoms test file.
app_mod = sys.modules["claritymed.servers.medical_clip.app"]


# --- stub engine ----------------------------------------------------------


class _StubEngine:
    """Deterministic BiomedClipEngine substitute for app tests."""

    model_id = "stub-biomedclip"
    model_revision = "stub-revision"
    device = "cpu"
    labels: list[Modality] = [
        "ultrasound",
        "ct",
        "xray",
        "dermoscopy",
        "photo",
        "document",
    ]

    def __init__(
        self,
        scores: dict[Modality, float] | None = None,
        *,
        raise_on_classify: ImageDecodeError | None = None,
    ) -> None:
        # default: confident ultrasound
        self._scores = scores or {
            "ultrasound": 0.93,
            "ct": 0.03,
            "xray": 0.02,
            "dermoscopy": 0.01,
            "photo": 0.005,
            "document": 0.005,
        }
        self._raise = raise_on_classify
        self.classify_calls: list[bytes] = []

    def classify(self, image_bytes: bytes) -> list[ModalityScore]:
        self.classify_calls.append(image_bytes)
        if self._raise is not None:
            raise self._raise
        ranked = sorted(self._scores.items(), key=lambda pair: pair[1], reverse=True)
        return [ModalityScore(label=label, score=score) for label, score in ranked]


# --- fixtures ------------------------------------------------------------


@pytest.fixture(autouse=True)
def _skip_load(monkeypatch: pytest.MonkeyPatch):
    """Bypass the BiomedCLIP lifespan load on every test in this file."""
    monkeypatch.setenv("CLARITYMED_MEDICAL_CLIP_SKIP_LOAD", "1")
    # Reset module state so tests don't leak engines across runs.
    app_mod._state["engine"] = None
    app_mod._state["started_at"] = 0.0
    app_mod._state["tasks_loaded"] = []
    app_mod._state["gating"] = {
        "min_confidence": 0.55,
        "min_medical_confidence": 0.70,
    }
    yield
    app_mod._state["engine"] = None


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def _png_bytes() -> bytes:
    """Tiny 1×1 PNG used as a payload across tests."""
    # 67-byte minimal PNG — generated once by Pillow, hardcoded here to
    # keep the test free of image-encoding dependencies.
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgAAIAAAUAAeImBZsAAAAASUVORK5CYII="
    )


def _payload(
    *,
    image_bytes: bytes | None = None,
    request_id: str = "req_1",
    sha_override: str | None = None,
) -> dict[str, Any]:
    raw = image_bytes if image_bytes is not None else _png_bytes()
    sha = sha_override or hashlib.sha256(raw).hexdigest()
    return {
        "request_id": request_id,
        "image": {
            "sha256": sha,
            "data_b64": base64.b64encode(raw).decode("ascii"),
        },
    }


# --- happy path ----------------------------------------------------------


def test_loopback_host_constant_is_127001() -> None:
    """KTD-V8 — server must refuse non-loopback binds at the constant level."""
    assert HOST == "127.0.0.1"


def test_health_reports_loading_when_engine_absent(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "loading"
    assert body["device"] == "-"
    assert body["tasks_loaded"] == []


def test_health_reports_ok_with_engine(client) -> None:
    app_mod._state["engine"] = _StubEngine()
    app_mod._state["tasks_loaded"] = ["modality"]
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_id"] == "stub-biomedclip"
    assert body["model_revision"] == "stub-revision"
    assert body["tasks_loaded"] == ["modality"]


def test_classify_modality_happy_path_ultrasound(client) -> None:
    app_mod._state["engine"] = _StubEngine()
    resp = client.post("/v1/classify_modality", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "ultrasound"
    assert body["is_medical"] is True
    assert body["confidence"] >= 0.7
    assert body["request_id"] == "req_1"
    assert body["scores"][0]["label"] == "ultrasound"  # descending order


def test_classify_modality_emits_unknown_below_min_confidence(client) -> None:
    """Plan edge case: top1 below min_confidence collapses to unknown."""
    app_mod._state["engine"] = _StubEngine(
        scores={
            "ultrasound": 0.40,
            "ct": 0.20,
            "xray": 0.20,
            "dermoscopy": 0.10,
            "photo": 0.05,
            "document": 0.05,
        }
    )
    resp = client.post("/v1/classify_modality", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "unknown"
    assert body["is_medical"] is False
    # Raw confidence preserved on the wire even after gating collapse
    assert body["confidence"] == pytest.approx(0.40, abs=0.001)


def test_classify_modality_is_medical_false_for_photo_top1(client) -> None:
    """Plan edge case: top1 in {photo, document} is never is_medical=true."""
    app_mod._state["engine"] = _StubEngine(
        scores={
            "photo": 0.92,
            "ultrasound": 0.04,
            "ct": 0.02,
            "xray": 0.01,
            "dermoscopy": 0.005,
            "document": 0.005,
        }
    )
    resp = client.post("/v1/classify_modality", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "photo"
    assert body["is_medical"] is False


def test_classify_modality_is_medical_false_below_min_medical(client) -> None:
    """Plan edge case: medical top1 above min_confidence but below min_medical."""
    app_mod._state["engine"] = _StubEngine(
        scores={
            "ultrasound": 0.60,  # above min_confidence (0.55), below min_medical (0.70)
            "ct": 0.20,
            "xray": 0.10,
            "dermoscopy": 0.05,
            "photo": 0.03,
            "document": 0.02,
        }
    )
    resp = client.post("/v1/classify_modality", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["modality"] == "ultrasound"
    assert body["is_medical"] is False


# --- error envelopes -----------------------------------------------------


def test_classify_modality_503_when_engine_missing(client) -> None:
    """No engine loaded → 503 with the uniform error envelope."""
    resp = client.post("/v1/classify_modality", json=_payload())
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "service_unavailable"
    assert body["error"]["request_id"] == "req_1"


def test_classify_modality_400_on_invalid_base64(client) -> None:
    app_mod._state["engine"] = _StubEngine()
    payload = _payload()
    payload["image"]["data_b64"] = "not!base64@@@"
    resp = client.post("/v1/classify_modality", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "image_decode_failed"


def test_classify_modality_400_on_sha_mismatch(client) -> None:
    """Plan edge case (Unit 4 mirror): decoded sha must equal claimed sha."""
    app_mod._state["engine"] = _StubEngine()
    payload = _payload(sha_override="b" * 64)
    resp = client.post("/v1/classify_modality", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "image_hash_mismatch"
    assert body["error"]["details"]["claimed"] == "b" * 64


def test_classify_modality_400_on_engine_decode_failure(client) -> None:
    """Engine raises ImageDecodeError → 400 image_decode_failed."""
    app_mod._state["engine"] = _StubEngine(
        raise_on_classify=ImageDecodeError("corrupt"),
    )
    resp = client.post("/v1/classify_modality", json=_payload())
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "image_decode_failed"


def test_classify_modality_validation_error_becomes_400(client) -> None:
    """Plan contract: Pydantic 422s normalize to 400 bad_request."""
    app_mod._state["engine"] = _StubEngine()
    # Missing required field `request_id`
    resp = client.post(
        "/v1/classify_modality",
        json={"image": {"sha256": "a" * 64, "data_b64": "AAAA"}},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "bad_request"


# --- internal helpers ----------------------------------------------------


def test_apply_gating_collapses_low_confidence_to_unknown() -> None:
    top1 = ModalityScore(label="ultrasound", score=0.40)
    gating = {"min_confidence": 0.55, "min_medical_confidence": 0.70}
    label, is_medical = _apply_gating(top1, gating)
    assert label == "unknown"
    assert is_medical is False


def test_apply_gating_photo_never_medical() -> None:
    top1 = ModalityScore(label="photo", score=0.99)
    gating = {"min_confidence": 0.55, "min_medical_confidence": 0.70}
    label, is_medical = _apply_gating(top1, gating)
    assert label == "photo"
    assert is_medical is False


def test_apply_gating_medical_above_threshold() -> None:
    top1 = ModalityScore(label="ct", score=0.80)
    gating = {"min_confidence": 0.55, "min_medical_confidence": 0.70}
    label, is_medical = _apply_gating(top1, gating)
    assert label == "ct"
    assert is_medical is True


def test_non_medical_labels_set_is_frozen() -> None:
    """Guard against accidental mutation of the structural rule."""
    assert "photo" in _NON_MEDICAL_LABELS
    assert "document" in _NON_MEDICAL_LABELS
    assert "unknown" in _NON_MEDICAL_LABELS
    assert "ultrasound" not in _NON_MEDICAL_LABELS


# --- env scrub -----------------------------------------------------------


def test_skip_load_env_active_for_this_module() -> None:
    """Sanity: every test in this file ran under skip-load."""
    assert os.environ.get("CLARITYMED_MEDICAL_CLIP_SKIP_LOAD") == "1"
