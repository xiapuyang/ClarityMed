"""Tests for ``PhiGuard``."""

from __future__ import annotations

import json

from claritymed.core.orchestrator import PhiGuard


def _sample_payload() -> dict:
    return {
        "patient": {
            "allergies": [
                {"substance": "penicillin", "severity": "severe"},
                {"substance": "latex", "severity": "mild"},
            ],
            "conditions": [{"code": "E11", "display": "Type 2 diabetes"}],
            "medications": [{"display": "metformin", "dose": "500 mg"}],
        },
        "lab_panel": {
            "values": [{"name": "glucose", "value": 5.4, "unit": "mmol/L"}],
            "source_doc": "uploads/lab.pdf",
        },
    }


def test_from_config_loads_real_safety_yaml():
    guard = PhiGuard.from_config()
    assert "patient.allergies.*.substance" in guard.rules.fields


def test_cloud_payload_gets_redacted():
    guard = PhiGuard.from_config()
    redacted, hits = guard.check_outbound(_sample_payload(), provider_kind="cloud")
    paths = [h.field_path for h in hits]
    assert "patient.allergies.0.substance" in paths
    assert "patient.allergies.1.substance" in paths
    text = json.dumps(redacted)
    assert "penicillin" not in text
    assert "latex" not in text
    # lab_panel.source_doc also a PHI field.
    assert "uploads/lab.pdf" not in text


def test_local_payload_passes_through():
    guard = PhiGuard.from_config()
    payload = _sample_payload()
    redacted, hits = guard.check_outbound(payload, provider_kind="local")
    assert hits == []
    assert redacted == payload


def test_missing_field_no_hit():
    guard = PhiGuard.from_config()
    payload = {"patient": {}}
    _, hits = guard.check_outbound(payload, provider_kind="cloud")
    assert hits == []


def test_payload_unchanged_when_no_rules():
    from claritymed.core.orchestrator.phi_guard import PhiRules

    guard = PhiGuard(PhiRules(fields=[], providers={"cloud": "deny"}))
    payload = _sample_payload()
    redacted, hits = guard.check_outbound(payload, provider_kind="cloud")
    assert hits == []
    assert redacted == payload
