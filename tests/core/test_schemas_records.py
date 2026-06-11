"""Tests for ``records.Manifest`` / ``Attachment`` / ``ExtractedLab``."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from claritymed.core.schemas.records import Attachment, ExtractedLab, Manifest


def _attachment(**overrides) -> Attachment:
    payload = {
        "sha256": "a" * 64,
        "filename": "report.pdf",
        "mime": "application/pdf",
        "size": 12345,
    }
    payload.update(overrides)
    return Attachment(**payload)


def _manifest(**overrides) -> Manifest:
    payload = {
        "revision": 1,
        "kind": "exam-report",
        "category": "exam-reports",
        "slug": "2026-06-11-ab12cd34",
        "title": "Annual checkup",
        "date": date(2026, 6, 11),
        "attachments": [_attachment()],
    }
    payload.update(overrides)
    return Manifest(**payload)


def test_manifest_minimum_revision_is_one():
    with pytest.raises(ValidationError):
        _manifest(revision=0)


def test_manifest_default_embed_status_ok():
    m = _manifest()
    assert m.embed_status == "ok"


def test_manifest_embed_status_alias_writes_underscore_key():
    """``_embed_status`` keeps the on-disk YAML's underscore-prefix convention
    while the Python attribute stays underscore-free."""
    m = _manifest()
    dumped = m.model_dump(by_alias=True)
    assert "_embed_status" in dumped
    assert "embed_status" not in dumped


def test_manifest_date_alias_round_trips():
    """YAML uses ``date:`` even though the Python field is ``event_date``."""
    m = Manifest(
        revision=1,
        kind="exam-report",
        category="exam-reports",
        slug="2026-06-11-ab12cd34",
        title="t",
        date="2026-06-11",
    )
    assert m.event_date == date(2026, 6, 11)
    dumped = m.model_dump(by_alias=True)
    assert dumped["date"] == date(2026, 6, 11)
    assert "event_date" not in dumped


def test_attachment_rejects_bad_sha():
    with pytest.raises(ValidationError):
        _attachment(sha256="not-hex")


def test_attachment_default_ocr_status_pending():
    a = _attachment()
    assert a.ocr_status == "pending"


def test_extracted_lab_accepts_numeric_or_qualitative_value():
    ExtractedLab(name="HGB", value=105, unit="g/L")
    ExtractedLab(name="result", value="positive")


def test_manifest_rejects_extra_field():
    with pytest.raises(ValidationError):
        _manifest(rogue_field="oops")


def test_manifest_library_metadata_optional():
    """Records leave ``authors``/``year``/``public`` at defaults."""
    m = _manifest()
    assert m.authors == []
    assert m.year is None
    assert m.public is False
