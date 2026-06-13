"""Tests for the seven ingest tools and the toolset wiring."""

from __future__ import annotations

import os

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.errors import RecordNotFound
from claritymed.orchestrator.features.ingest_tools_plugin import (
    INGEST_TOOLS,
    build_ingest_toolset,
    delete_record,
    save_allergy,
    save_condition,
    save_medication,
    save_record,
    save_to_library,
    update_profile_field,
)
from claritymed.orchestrator.services.tool_dispatcher import ToolDispatcher
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.manifest_store import ManifestStore
from claritymed.stores.paths import user_audit_payload_path
from claritymed.stores.profile import ProfileStore


@pytest.fixture
def _ctx():
    tokens = apply_context("20260611000000ABCDEF12", "alice", "en")
    yield
    reset_context(tokens)


@pytest.fixture
def dispatcher() -> ToolDispatcher:
    return ToolDispatcher()


def test_seven_tools_registered():
    assert sorted(INGEST_TOOLS) == [
        "delete_record",
        "save_allergy",
        "save_condition",
        "save_medication",
        "save_record",
        "save_to_library",
        "update_profile_field",
    ]


def test_save_allergy_persists_and_audits(dispatcher, _ctx):
    out = save_allergy(
        {"substance": "penicillin", "severity": "severe", "source": "self_report"},
        dispatcher=dispatcher,
    )
    assert out == {"ok": True}
    rows = ProfileStore("alice").list_allergies()
    assert any(a.substance == "penicillin" for a in rows)


def test_save_medication_persists(dispatcher, _ctx):
    out = save_medication(
        {"name": "metformin", "dose": "500mg", "frequency": "BID"},
        dispatcher=dispatcher,
    )
    assert out == {"ok": True}
    rows = ProfileStore("alice").list_medications()
    assert any(m.display == "metformin" for m in rows)


def test_save_medication_round_trips_dates(dispatcher, _ctx):
    save_medication(
        {
            "name": "aspirin",
            "onset_date": "2020-01-01",
            "end_date": "2022-06-15",
        },
        dispatcher=dispatcher,
    )
    [asp] = [
        m for m in ProfileStore("alice").list_medications() if m.display == "aspirin"
    ]
    assert asp.onset_date.isoformat() == "2020-01-01"
    assert asp.end_date.isoformat() == "2022-06-15"


def test_save_allergy_round_trips_dates(dispatcher, _ctx):
    save_allergy(
        {
            "substance": "shellfish",
            "severity": "moderate",
            "source": "self_report",
            "onset_date": "2015-07-04",
        },
        dispatcher=dispatcher,
    )
    [sh] = [
        a for a in ProfileStore("alice").list_allergies() if a.substance == "shellfish"
    ]
    assert sh.onset_date.isoformat() == "2015-07-04"
    assert sh.end_date is None


def test_save_condition_persists(dispatcher, _ctx):
    out = save_condition({"display": "Hypertension"}, dispatcher=dispatcher)
    assert out == {"ok": True}
    rows = ProfileStore("alice").list_conditions()
    assert any(c.display == "Hypertension" for c in rows)


def test_save_condition_persists_end_date(dispatcher, _ctx):
    """end_date null = still ongoing; populated = resolved."""
    save_condition(
        {
            "display": "bronchitis",
            "onset_date": "2024-01-01",
            "end_date": "2024-02-15",
        },
        dispatcher=dispatcher,
    )
    rows = ProfileStore("alice").list_conditions()
    [bron] = [c for c in rows if c.display == "bronchitis"]
    assert bron.onset_date.isoformat() == "2024-01-01"
    assert bron.end_date.isoformat() == "2024-02-15"


def test_update_profile_field_changes_weight(dispatcher, _ctx):
    out = update_profile_field(
        {"field": "weight_kg", "value": 72.5}, dispatcher=dispatcher
    )
    assert out == {"ok": True}
    profile = ProfileStore("alice").get_profile()
    assert profile.weight_kg == 72.5


def test_update_profile_field_rejects_invalid_field(dispatcher, _ctx):
    with pytest.raises(ValueError):
        update_profile_field(
            {"field": "created_at", "value": "x"}, dispatcher=dispatcher
        )


def test_update_profile_field_writes_passive_geo_field(dispatcher, _ctx):
    """Residence (proactive) and current_occupation (passive) both round-trip."""
    update_profile_field(
        {"field": "residence", "value": "Shanghai"}, dispatcher=dispatcher
    )
    update_profile_field(
        {"field": "current_occupation", "value": "nurse"}, dispatcher=dispatcher
    )
    p = ProfileStore("alice").get_profile()
    assert p.residence == "Shanghai"
    assert p.current_occupation == "nurse"


def test_update_profile_field_audit_tags_solicitation(dispatcher, _ctx, monkeypatch):
    """Every update_profile_field audit row carries proactive|passive."""
    captured: list[tuple[str, dict]] = []

    def _capture(kind, payload):
        captured.append((kind, payload))

    monkeypatch.setattr(
        "claritymed.orchestrator.features.ingest_tools_plugin.audit_event",
        _capture,
    )
    update_profile_field({"field": "weight_kg", "value": 70.0}, dispatcher=dispatcher)
    update_profile_field(
        {"field": "marital_status", "value": "married"}, dispatcher=dispatcher
    )
    by_field = {p["field"]: p for _, p in captured}
    assert by_field["weight_kg"]["solicitation"] == "proactive"
    assert by_field["marital_status"]["solicitation"] == "passive"


def test_save_record_writes_manifest(dispatcher, _ctx):
    bs = BlobStore("alice")
    sha = bs.store(b"report bytes", "pdf")
    out = save_record(
        {
            "category": "exam-reports",
            "kind": "exam-report",
            "title": "Annual",
            "date": "2026-06-11",
            "attachments": [{"sha256": sha, "filename": "r.pdf"}],
        },
        dispatcher=dispatcher,
    )
    assert "record_path" in out
    assert out["record_path"].startswith("exam-reports/2026-06-11-")
    # Manifest exists on disk.
    cat, slug = out["record_path"].split("/")
    m = ManifestStore("alice", "records").read(cat, slug)
    assert m.title == "Annual"


def test_save_record_unknown_sha_raises(dispatcher, _ctx):
    from claritymed.errors import UnknownSha256

    with pytest.raises(UnknownSha256):
        save_record(
            {
                "category": "exam-reports",
                "kind": "exam-report",
                "title": "x",
                "attachments": [{"sha256": "b" * 64, "filename": "x.pdf"}],
            },
            dispatcher=dispatcher,
        )


def test_save_to_library_writes_manifest(dispatcher, _ctx):
    bs = BlobStore("alice")
    sha = bs.store(b"paper", "pdf")
    out = save_to_library(
        {
            "title": "Paper Title",
            "attachments": [{"sha256": sha, "filename": "p.pdf"}],
            "public": False,
        },
        dispatcher=dispatcher,
    )
    assert "library_path" in out
    cat, slug = out["library_path"].split("/")
    m = ManifestStore("alice", "library").read(cat, slug)
    assert m.title == "Paper Title"
    assert m.public is False


def test_delete_record_round_trip(dispatcher, _ctx):
    bs = BlobStore("alice")
    sha = bs.store(b"x", "pdf")
    saved = save_record(
        {
            "category": "exam-reports",
            "kind": "exam-report",
            "title": "to-delete",
            "attachments": [{"sha256": sha, "filename": "x.pdf"}],
        },
        dispatcher=dispatcher,
    )
    record_path = saved["record_path"]
    out = delete_record(
        {"record_path": record_path, "confirm_kind": "exam-report"},
        dispatcher=dispatcher,
    )
    assert out == {"deleted": record_path}
    cat, slug = record_path.split("/")
    with pytest.raises(RecordNotFound):
        ManifestStore("alice", "records").read(cat, slug)


def test_delete_record_confirm_kind_mismatch(dispatcher, _ctx):
    bs = BlobStore("alice")
    sha = bs.store(b"y", "pdf")
    saved = save_record(
        {
            "category": "exam-reports",
            "kind": "exam-report",
            "title": "keep",
            "attachments": [{"sha256": sha, "filename": "y.pdf"}],
        },
        dispatcher=dispatcher,
    )
    with pytest.raises(ValueError, match="confirm_kind"):
        delete_record(
            {
                "record_path": saved["record_path"],
                "confirm_kind": "wrong-kind",
            },
            dispatcher=dispatcher,
        )


def test_audit_payload_written_with_owner_only_mode(dispatcher, _ctx):
    save_allergy(
        {"substance": "peanut", "severity": "mild", "source": "self_report"},
        dispatcher=dispatcher,
    )
    payload_path = user_audit_payload_path("alice", "20260611000000ABCDEF12")
    assert payload_path.exists()
    # Owner-only mode on POSIX. Check the low 9 bits are 0o600.
    mode = os.stat(payload_path).st_mode & 0o777
    assert mode == 0o600


def test_build_ingest_toolset_returns_toolset(dispatcher):
    ts = build_ingest_toolset(dispatcher)
    # Default path (no approval_required_func) → inner FunctionToolset.
    from pydantic_ai.toolsets import FunctionToolset

    assert isinstance(ts, FunctionToolset)


def test_build_ingest_toolset_wraps_when_approval_func(dispatcher):
    from pydantic_ai.toolsets import ApprovalRequiredToolset

    ts = build_ingest_toolset(dispatcher, approval_required_func=lambda *a, **kw: True)
    assert isinstance(ts, ApprovalRequiredToolset)


# --- no-op guard tests (items 6+7) -----------------------------------


def test_save_allergy_no_change_on_duplicate(dispatcher, _ctx):
    save_allergy(
        {"substance": "penicillin", "severity": "severe", "source": "self_report"},
        dispatcher=dispatcher,
    )
    out = save_allergy(
        {"substance": "penicillin", "severity": "mild", "source": "clinical_record"},
        dispatcher=dispatcher,
    )
    assert out == {"ok": False, "reason": "no_change"}
    # Only one row written.
    assert len(ProfileStore("alice").list_allergies()) == 1


def test_save_allergy_no_change_case_insensitive(dispatcher, _ctx):
    save_allergy(
        {"substance": "Shellfish", "severity": "mild", "source": "self_report"},
        dispatcher=dispatcher,
    )
    out = save_allergy(
        {"substance": "shellfish", "severity": "mild", "source": "self_report"},
        dispatcher=dispatcher,
    )
    assert out == {"ok": False, "reason": "no_change"}


def test_save_allergy_allows_when_resolved(dispatcher, _ctx):
    """A resolved allergy (end_date set) does not block a new active record."""
    save_allergy(
        {
            "substance": "penicillin",
            "severity": "mild",
            "source": "self_report",
            "end_date": "2024-01-01",
        },
        dispatcher=dispatcher,
    )
    out = save_allergy(
        {"substance": "penicillin", "severity": "severe", "source": "clinical_record"},
        dispatcher=dispatcher,
    )
    assert out == {"ok": True}


def test_save_condition_no_change_on_duplicate(dispatcher, _ctx):
    save_condition({"display": "Hypertension"}, dispatcher=dispatcher)
    out = save_condition({"display": "Hypertension"}, dispatcher=dispatcher)
    assert out == {"ok": False, "reason": "no_change"}


def test_save_condition_no_change_case_insensitive(dispatcher, _ctx):
    save_condition({"display": "Type 2 Diabetes"}, dispatcher=dispatcher)
    out = save_condition({"display": "type 2 diabetes"}, dispatcher=dispatcher)
    assert out == {"ok": False, "reason": "no_change"}


def test_save_condition_allows_when_resolved(dispatcher, _ctx):
    save_condition(
        {"display": "bronchitis", "end_date": "2023-03-01"}, dispatcher=dispatcher
    )
    out = save_condition({"display": "bronchitis"}, dispatcher=dispatcher)
    assert out == {"ok": True}


def test_save_medication_no_change_on_duplicate(dispatcher, _ctx):
    save_medication({"name": "metformin", "dose": "500mg"}, dispatcher=dispatcher)
    out = save_medication(
        {"name": "metformin", "dose": "1000mg"}, dispatcher=dispatcher
    )
    assert out == {"ok": False, "reason": "no_change"}


def test_save_medication_allows_when_discontinued(dispatcher, _ctx):
    save_medication(
        {"name": "aspirin", "end_date": "2022-06-01"}, dispatcher=dispatcher
    )
    out = save_medication({"name": "aspirin"}, dispatcher=dispatcher)
    assert out == {"ok": True}


def test_update_profile_field_no_change_same_value(dispatcher, _ctx):
    update_profile_field({"field": "weight_kg", "value": 72.5}, dispatcher=dispatcher)
    out = update_profile_field(
        {"field": "weight_kg", "value": 72.5}, dispatcher=dispatcher
    )
    assert out == {"ok": False, "reason": "no_change"}


def test_update_profile_field_no_change_string_coercion(dispatcher, _ctx):
    """'72.5' and 72.5 coerce to the same float → no_change."""
    update_profile_field({"field": "weight_kg", "value": 72.5}, dispatcher=dispatcher)
    out = update_profile_field(
        {"field": "weight_kg", "value": "72.5"}, dispatcher=dispatcher
    )
    assert out == {"ok": False, "reason": "no_change"}


def test_update_profile_field_no_change_birth_date_downgrade(dispatcher, _ctx):
    """Year-only approximation must not overwrite a precise full date."""
    update_profile_field(
        {"field": "birth_date", "value": "1989-05-02"}, dispatcher=dispatcher
    )
    out = update_profile_field(
        {"field": "birth_date", "value": "1989-01-01"}, dispatcher=dispatcher
    )
    assert out == {"ok": False, "reason": "no_change"}


def test_update_profile_field_allows_more_specific_date(dispatcher, _ctx):
    """Replacing a year-only approximation with a precise date is allowed."""
    update_profile_field(
        {"field": "birth_date", "value": "1989-01-01"}, dispatcher=dispatcher
    )
    out = update_profile_field(
        {"field": "birth_date", "value": "1989-05-02"}, dispatcher=dispatcher
    )
    assert out == {"ok": True}


# --- format helper tests (item 5) ------------------------------------


def test_format_record_embed_text_title_only():
    from claritymed.orchestrator.features.ingest_tools_plugin import (
        _format_record_embed_text,
    )
    from claritymed.core.schemas.tools import SaveRecordArgs

    parsed = SaveRecordArgs(category="labs", kind="lab_report", title="Lipid panel")
    text = _format_record_embed_text(parsed)
    assert "Lipid panel" in text


def test_format_record_embed_text_includes_labs_and_notes():
    from claritymed.orchestrator.features.ingest_tools_plugin import (
        _format_record_embed_text,
    )
    from claritymed.core.schemas.tools import SaveRecordArgs
    from claritymed.core.schemas.records import ExtractedLab

    parsed = SaveRecordArgs(
        category="labs",
        kind="lab_report",
        title="Annual labs",
        notes="Fasting sample",
        extracted_labs=[ExtractedLab(name="LDL", value="120", unit="mg/dL")],
        tags=["fasting"],
    )
    text = _format_record_embed_text(parsed)
    assert "Fasting sample" in text
    assert "LDL: 120 mg/dL" in text
    assert "fasting" in text


def test_format_library_embed_text_fields():
    from claritymed.orchestrator.features.ingest_tools_plugin import (
        _format_library_embed_text,
    )
    from claritymed.core.schemas.tools import SaveToLibraryArgs

    parsed = SaveToLibraryArgs(
        title="Hypertension Guideline 2024",
        authors=["Smith", "Jones"],
        year=2024,
        tags=["cardiology"],
    )
    text = _format_library_embed_text(parsed)
    assert "Hypertension Guideline 2024" in text
    assert "Smith" in text
    assert "2024" in text
    assert "cardiology" in text
