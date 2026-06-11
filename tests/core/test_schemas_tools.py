"""Tests for the tool-args pydantic schemas (``core/schemas/tools.py``)."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from claritymed.core.schemas.tools import (
    TOOL_ARG_SCHEMAS,
    AttachmentRef,
    DeleteRecordArgs,
    SaveAllergyArgs,
    SaveConditionArgs,
    SaveMedicationArgs,
    SaveRecordArgs,
    SaveToLibraryArgs,
    UpdateProfileFieldArgs,
)


def test_registry_lists_seven_tools():
    assert sorted(TOOL_ARG_SCHEMAS) == [
        "delete_record",
        "save_allergy",
        "save_condition",
        "save_medication",
        "save_record",
        "save_to_library",
        "update_profile_field",
    ]


def test_attachment_ref_requires_sha256_hex():
    AttachmentRef(sha256="a" * 64, filename="r.pdf")
    with pytest.raises(ValidationError):
        AttachmentRef(sha256="not-hex", filename="r.pdf")


def test_save_record_args_date_alias():
    """LLM submits ``date`` in JSON; field name stays ``event_date``."""
    args = SaveRecordArgs(
        category="exam-reports",
        kind="exam-report",
        title="Annual checkup",
        date="2026-06-11",
        attachments=[AttachmentRef(sha256="a" * 64, filename="r.pdf")],
    )
    assert args.event_date == date(2026, 6, 11)


def test_save_record_args_attachments_optional():
    args = SaveRecordArgs(category="exam-reports", kind="exam-report", title="t")
    assert args.attachments == []


def test_save_medication_args_minimum_required():
    SaveMedicationArgs(name="metformin")
    with pytest.raises(ValidationError):
        SaveMedicationArgs()


def test_save_allergy_args_severity_enforced():
    SaveAllergyArgs(substance="penicillin", severity="severe", source="self_report")
    with pytest.raises(ValidationError):
        SaveAllergyArgs(substance="x", severity="bogus", source="self_report")


def test_save_condition_args_optional_date():
    SaveConditionArgs(display="Hypertension")
    SaveConditionArgs(display="Hypertension", onset_date=date(2020, 1, 1))


def test_update_profile_field_whitelist():
    UpdateProfileFieldArgs(field="weight_kg", value=72.5)
    UpdateProfileFieldArgs(field="sex", value="female")
    with pytest.raises(ValidationError):
        UpdateProfileFieldArgs(field="created_at", value="x")


def test_save_to_library_args_public_defaults_false():
    args = SaveToLibraryArgs(title="Paper")
    assert args.public is False


def test_delete_record_args_requires_confirm_kind():
    DeleteRecordArgs(record_path="/some/path", confirm_kind="exam-report")
    with pytest.raises(ValidationError):
        DeleteRecordArgs(record_path="/some/path")


def test_save_record_args_rejects_extra_fields():
    """LLM-hallucinated extra field becomes a ValidationError, not a silent
    dropped key — keeps drift between prompt YAML and schema visible."""
    with pytest.raises(ValidationError):
        SaveRecordArgs(
            category="x",
            kind="y",
            title="t",
            rogue="oops",
        )
