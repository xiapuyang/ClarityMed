"""E2E: privacy-filter model exercised against each supported entity type.

What these tests prove end-to-end (no mocks anywhere):

* The ONNX model runs inference successfully via _OnnxNerPipeline.
* Each entity_group label the model was trained on is either correctly
  redacted (PERSON, ADDRESS, EMAIL, PHONE, URL, ACCOUNT, SECRET) or
  correctly preserved (DATE — in _SKIP_LABELS).
* _apply_spans replaces detected spans with the right [REDACTED:X] placeholder.
* ScrubReport.model_hit_types is populated with per-label counts.

Run:
    uv run pytest tests/e2e/scrub -v --no-cov

Required services:
    openai/privacy-filter ONNX model cached locally (see conftest.py).
"""

from __future__ import annotations

import pytest

from claritymed.core.scrub.service import ScrubService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_redacted(scrubbed: str, original_fragment: str, placeholder: str) -> None:
    assert original_fragment not in scrubbed, (
        f"Expected {original_fragment!r} to be redacted, but it survived in: {scrubbed!r}"
    )
    assert placeholder in scrubbed, (
        f"Expected {placeholder!r} in output, got: {scrubbed!r}"
    )


# ---------------------------------------------------------------------------
# One test per supported label
# ---------------------------------------------------------------------------


def test_private_person_redacted(scrub_service: ScrubService) -> None:
    text = "Patient John Smith was admitted on Monday."
    scrubbed, report = scrub_service.scrub(text)
    _assert_redacted(scrubbed, "John Smith", "[REDACTED:PERSON]")
    assert report.model_hit_types.get("private_person", 0) >= 1


def test_private_address_redacted(scrub_service: ScrubService) -> None:
    text = "He resides at 742 Evergreen Terrace, Springfield, IL 62704."
    scrubbed, report = scrub_service.scrub(text)
    assert "[REDACTED:ADDRESS]" in scrubbed, (
        f"Expected address to be redacted, got: {scrubbed!r}"
    )
    assert report.model_hit_types.get("private_address", 0) >= 1


def test_private_email_redacted(model_only_scrub_service: ScrubService) -> None:
    # Use model_only_scrub_service: the regex pre-pass would otherwise catch the
    # address first, leaving model_hits=0 and defeating the e2e assertion.
    text = "Send results to john.smith@hospital.org for follow-up."
    scrubbed, report = model_only_scrub_service.scrub(text)
    _assert_redacted(scrubbed, "john.smith@hospital.org", "[REDACTED:EMAIL]")
    assert report.model_hit_types.get("private_email", 0) >= 1


def test_private_phone_redacted(model_only_scrub_service: ScrubService) -> None:
    # Same reason: regex layer has a phone rule that fires before the model.
    text = "Call the patient at +1-555-867-5309 to confirm the appointment."
    scrubbed, report = model_only_scrub_service.scrub(text)
    _assert_redacted(scrubbed, "555-867-5309", "[REDACTED:PHONE]")
    assert report.model_hit_types.get("private_phone", 0) >= 1


def test_private_url_redacted(scrub_service: ScrubService) -> None:
    text = "Portal access: https://patient.hospital.org/records/john-smith"
    scrubbed, report = scrub_service.scrub(text)
    assert "[REDACTED:URL]" in scrubbed, (
        f"Expected URL to be redacted, got: {scrubbed!r}"
    )
    assert report.model_hit_types.get("private_url", 0) >= 1


def test_account_number_redacted(scrub_service: ScrubService) -> None:
    text = "Insurance account number: 4532015112830366."
    scrubbed, report = scrub_service.scrub(text)
    _assert_redacted(scrubbed, "4532015112830366", "[REDACTED:ACCOUNT]")
    assert report.model_hit_types.get("account_number", 0) >= 1


def test_secret_redacted(scrub_service: ScrubService) -> None:
    text = "API key: tok_FAKESECRET00000000000000000000"
    scrubbed, report = scrub_service.scrub(text)
    assert "[REDACTED:SECRET]" in scrubbed or "[REDACTED]" in scrubbed, (
        f"Expected secret to be redacted, got: {scrubbed!r}"
    )
    assert report.model_hits >= 1


# ---------------------------------------------------------------------------
# private_date must NOT be redacted
# ---------------------------------------------------------------------------


def test_private_date_preserved(scrub_service: ScrubService) -> None:
    """Dates are clinical context — the model may detect them but _SKIP_LABELS prevents redaction."""
    text = "Symptoms started on March 15, 2024 and resolved by April 2."
    scrubbed, _ = scrub_service.scrub(text)
    # The exact dates should survive in the output unchanged.
    assert "March 15, 2024" in scrubbed, (
        f"Date was incorrectly redacted from: {scrubbed!r}"
    )
    assert "April 2" in scrubbed, f"Date was incorrectly redacted from: {scrubbed!r}"
    assert "[REDACTED:DATE]" not in scrubbed


# ---------------------------------------------------------------------------
# hit_types in audit payload — verified via caplog
# ---------------------------------------------------------------------------


def test_audit_hit_types_populated(
    model_only_scrub_service: ScrubService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """scrub.privacy_filter audit entry contains hit_types, captured via caplog.

    In test mode _in_test_mode() leaves propagate=True on claritymed.audit so
    caplog can intercept records emitted by audit_event() without any file I/O.
    """
    import json
    import logging

    text = "Patient Jane Doe, email jane@example.com"
    with caplog.at_level(logging.INFO, logger="claritymed.audit"):
        model_only_scrub_service.scrub(text)

    audit_records = [r for r in caplog.records if r.name == "claritymed.audit"]
    assert audit_records, "No claritymed.audit records captured by caplog"

    ok_events = []
    for record in audit_records:
        try:
            event = json.loads(record.getMessage())
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            event.get("kind") == "scrub.privacy_filter"
            and event.get("payload", {}).get("status") == "ok"
        ):
            ok_events.append(event)

    assert ok_events, (
        f"No scrub.privacy_filter ok events in captured audit records: "
        f"{[r.getMessage() for r in audit_records]}"
    )

    payload = ok_events[-1]["payload"]
    assert "hit_types" in payload, f"hit_types missing from audit payload: {payload}"
    assert isinstance(payload["hit_types"], dict)
    detected_labels = set(payload["hit_types"].keys())
    assert detected_labels & {"private_person", "private_email"}, (
        f"Expected person or email in hit_types, got: {detected_labels}"
    )
