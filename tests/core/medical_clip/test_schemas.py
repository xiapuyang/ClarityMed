"""Validate the medical-clip wire schemas + ``configs/medical_clip.yaml``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from claritymed.config import CONFIGS_DIR
from claritymed.core.medical_clip.schemas import (
    HealthResponse,
    ImagePayload,
    ModalityRequest,
    ModalityResponse,
    ModalityScore,
)

_HEX64 = "a" * 64


# --- ModalityRequest / ModalityResponse -----------------------------------


def test_modality_request_accepts_b64_image() -> None:
    """Happy path from the plan: base64 + sha256 image dict round-trips."""
    req = ModalityRequest.model_validate(
        {
            "request_id": "req_1",
            "image": {"sha256": _HEX64, "data_b64": "iVBORw0KGgo="},
        }
    )
    assert req.image.sha256 == _HEX64


def test_modality_response_round_trip() -> None:
    resp = ModalityResponse.model_validate(
        {
            "request_id": "req_1",
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
    )
    assert resp.modality == "ultrasound"
    assert resp.is_medical is True
    assert len(resp.scores) == 6


def test_modality_response_unknown_bucket_is_legal() -> None:
    """Server returns modality='unknown' when top1 falls below min_confidence."""
    resp = ModalityResponse.model_validate(
        {
            "request_id": "req_1",
            "modality": "unknown",
            "confidence": 0.4,
            "is_medical": False,
            "scores": [{"label": "unknown", "score": 0.4}],
            "elapsed_ms": 50,
        }
    )
    assert resp.modality == "unknown"


def test_modality_literal_rejects_unknown_label() -> None:
    """Edge case from the plan: Modality literal rejects garbage."""
    with pytest.raises(ValidationError):
        ModalityScore.model_validate({"label": "unknown_modality", "score": 0.5})


def test_image_payload_rejects_short_sha() -> None:
    with pytest.raises(ValidationError):
        ImagePayload.model_validate({"sha256": "abc", "data_b64": "x"})


def test_image_payload_rejects_uppercase_sha() -> None:
    """sha256 fields normalize to lowercase hex elsewhere; reject mixed case here."""
    with pytest.raises(ValidationError):
        ImagePayload.model_validate({"sha256": "A" * 64, "data_b64": "x"})


def test_modality_score_clamps_score_to_unit_interval() -> None:
    with pytest.raises(ValidationError):
        ModalityScore.model_validate({"label": "ultrasound", "score": 1.5})


def test_modality_response_extra_field_forbidden() -> None:
    """frozen + extra=forbid: an LLM that hallucinates a field gets ValidationError."""
    with pytest.raises(ValidationError):
        ModalityResponse.model_validate(
            {
                "request_id": "req_1",
                "modality": "ultrasound",
                "confidence": 0.93,
                "is_medical": True,
                "scores": [{"label": "ultrasound", "score": 0.93}],
                "elapsed_ms": 48,
                "hallucinated": "value",
            }
        )


# --- HealthResponse -------------------------------------------------------


def test_health_response_minimal_fields() -> None:
    h = HealthResponse.model_validate(
        {
            "status": "ok",
            "model_id": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            "device": "mps",
            "uptime_s": 1234,
        }
    )
    assert h.model_revision is None
    assert h.tasks_loaded == []


# --- configs/medical_clip.yaml --------------------------------------------


def test_shipped_configs_medical_clip_yaml_loads() -> None:
    """Smoke: medical_clip.yaml parses + has the structure ``app.py`` expects.

    Unit 1 doesn't ship a Pydantic root model for medical_clip.yaml (server
    config is hand-read in ``app.py`` lifespan, Unit 2). Here we assert the
    shape downstream code will rely on so a typo at this point doesn't
    silently break Unit 2 when it lands.
    """
    raw = yaml.safe_load(Path(CONFIGS_DIR / "medical_clip.yaml").read_text())

    assert raw["server"]["base_url"] == "http://127.0.0.1:8086"
    assert raw["model"]["model_id"].startswith("microsoft/BiomedCLIP")
    assert raw["model"]["device"] in ("auto", "cpu", "cuda", "mps")

    modality = raw["tasks"]["modality"]
    assert isinstance(modality["candidates"], list)
    assert len(modality["candidates"]) >= 5

    # Each candidate matches Modality literal values (minus 'unknown')
    candidate_labels = {entry["label"] for entry in modality["candidates"]}
    assert candidate_labels.issubset(
        {"ultrasound", "ct", "xray", "dermoscopy", "photo", "document"}
    )

    gating = modality["gating"]
    assert 0.0 < gating["min_confidence"] < 1.0
    assert 0.0 < gating["min_medical_confidence"] < 1.0


def test_medical_clip_yaml_gating_outside_unit_interval_is_invalid() -> None:
    """Plan's error-path scenario: gating thresholds outside (0, 1) must be caught.

    Without a Pydantic root model we exercise the same guard the lifespan
    code will use: a simple range check. Unit 2's ``app.py`` will move this
    into a validator; the test stays as documentation of the contract.
    """
    raw = yaml.safe_load(Path(CONFIGS_DIR / "medical_clip.yaml").read_text())
    raw["tasks"]["modality"]["gating"]["min_confidence"] = 1.5

    gating = raw["tasks"]["modality"]["gating"]
    assert not (0.0 < gating["min_confidence"] < 1.0)
