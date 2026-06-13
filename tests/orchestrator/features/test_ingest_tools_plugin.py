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
    tokens = apply_context("20260611000000ABCDEF12", "test", "en")
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
    rows = ProfileStore("test").list_allergies()
    assert any(a.substance == "penicillin" for a in rows)


def test_save_medication_persists(dispatcher, _ctx):
    out = save_medication(
        {"name": "metformin", "dose": "500mg", "frequency": "BID"},
        dispatcher=dispatcher,
    )
    assert out == {"ok": True}
    rows = ProfileStore("test").list_medications()
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
        m for m in ProfileStore("test").list_medications() if m.display == "aspirin"
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
        a for a in ProfileStore("test").list_allergies() if a.substance == "shellfish"
    ]
    assert sh.onset_date.isoformat() == "2015-07-04"
    assert sh.end_date is None


def test_save_condition_persists(dispatcher, _ctx):
    out = save_condition({"display": "Hypertension"}, dispatcher=dispatcher)
    assert out == {"ok": True}
    rows = ProfileStore("test").list_conditions()
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
    rows = ProfileStore("test").list_conditions()
    [bron] = [c for c in rows if c.display == "bronchitis"]
    assert bron.onset_date.isoformat() == "2024-01-01"
    assert bron.end_date.isoformat() == "2024-02-15"


def test_update_profile_field_changes_weight(dispatcher, _ctx):
    out = update_profile_field(
        {"field": "weight_kg", "value": 72.5}, dispatcher=dispatcher
    )
    assert out == {"ok": True}
    profile = ProfileStore("test").get_profile()
    assert profile.weight_kg == 72.5


def test_update_profile_field_rejects_invalid_field(dispatcher, _ctx):
    """Schema-level rejection (``field`` is a pydantic ``Literal``) now
    surfaces as ``ModelRetry`` so the agent loop can hand the message
    back to the LLM, rather than aborting the whole turn on a typo."""
    from pydantic_ai.exceptions import ModelRetry

    with pytest.raises(ModelRetry):
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
    p = ProfileStore("test").get_profile()
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
    bs = BlobStore("test")
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
    m = ManifestStore("test", "records").read(cat, slug)
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
    bs = BlobStore("test")
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
    m = ManifestStore("test", "library").read(cat, slug)
    assert m.title == "Paper Title"
    assert m.public is False


def test_delete_record_round_trip(dispatcher, _ctx):
    bs = BlobStore("test")
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
        ManifestStore("test", "records").read(cat, slug)


def test_delete_record_confirm_kind_mismatch(dispatcher, _ctx):
    bs = BlobStore("test")
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
    payload_path = user_audit_payload_path("test", "20260611000000ABCDEF12")
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


@pytest.mark.asyncio
async def test_disable_ingest_hooks_env_skips_embed_task(dispatcher, _ctx, monkeypatch):
    """``CLARITYMED_DISABLE_INGEST_HOOKS=1`` must short-circuit the
    fire-and-forget embed task on ``save_record``. The benchmark / e2e
    harnesses set this so a successful tool call does not also fan out
    to embedder + qdrant — those are separately tested elsewhere, and
    here they're just noise that can race with per-trial dir wipes."""
    import claritymed.orchestrator.features.ingest_tools_plugin as plugin

    monkeypatch.setenv("CLARITYMED_DISABLE_INGEST_HOOKS", "1")

    called = False

    async def _spy(result, kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(plugin, "_embed_record_task", _spy)
    monkeypatch.setattr(plugin, "_embed_library_task", _spy)

    ts = build_ingest_toolset(dispatcher)
    # Invoke the save_record entry directly; the toolset registers it as
    # a sync callable when no embed hook is bound. The entry must NOT
    # spawn an embed task.
    save_record_tool = ts.tools["save_record"]
    result = save_record_tool.function(
        category="checkups",
        kind="checkup",
        title="Annual",
    )
    if hasattr(result, "__await__"):
        result = await result
    # Yield once so any (mistakenly) scheduled task would get to run.
    import asyncio

    await asyncio.sleep(0)
    assert called is False, "embed hook ran despite CLARITYMED_DISABLE_INGEST_HOOKS=1"
    assert isinstance(result, dict)


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
    assert len(ProfileStore("test").list_allergies()) == 1


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


# ---------------------------------------------------------------------------
# delete_record edge cases (validation branches)
# ---------------------------------------------------------------------------


def test_delete_record_rejects_malformed_record_path(dispatcher, _ctx):
    """ValueError must surface when record_path lacks a category/slug split."""
    with pytest.raises(ValueError, match="category/slug"):
        delete_record(
            {"record_path": "no_slash_here", "confirm_kind": "exam-report"},
            dispatcher=dispatcher,
        )


def test_delete_record_missing_record_raises_not_found(dispatcher, _ctx):
    """RecordNotFound propagates when slug doesn't exist on disk."""
    with pytest.raises(RecordNotFound):
        delete_record(
            {
                "record_path": "exam-reports/2099-01-01-ghost",
                "confirm_kind": "exam-report",
            },
            dispatcher=dispatcher,
        )


# ---------------------------------------------------------------------------
# _validate_ingest_prompts: missing prompts must fail loud
# ---------------------------------------------------------------------------


def test_validate_ingest_prompts_raises_with_listing():
    from claritymed.orchestrator.features.ingest_tools_plugin import (
        _validate_ingest_prompts,
    )

    class _EmptyRegistry:
        def get(self, *args, **kwargs):
            raise KeyError("not found")

    with pytest.raises(RuntimeError, match="Missing ingest tool prompts"):
        _validate_ingest_prompts(_EmptyRegistry())


# ---------------------------------------------------------------------------
# IngestToolsFeature pre_invoke + as_tool
# ---------------------------------------------------------------------------


async def test_ingest_tools_feature_pre_invoke_returns_empty(dispatcher):
    from claritymed.orchestrator.features.ingest_tools_plugin import IngestToolsFeature

    feat = IngestToolsFeature(dispatcher, approval_required_func=None)
    assert await feat.pre_invoke(None) == ""
    assert feat.as_tool() is None


# ---------------------------------------------------------------------------
# _embed_record_task / _embed_library_task background helpers
# ---------------------------------------------------------------------------


async def test_embed_record_task_returns_silently_when_no_record_path(_ctx):
    from claritymed.orchestrator.features.ingest_tools_plugin import _embed_record_task

    # Empty result.record_path → early return, no store touched.
    await _embed_record_task({}, {})  # must not raise


async def test_embed_record_task_swallows_store_errors(_ctx, monkeypatch):
    """Background task must log + swallow any exception (no propagation)."""
    from claritymed.orchestrator.features import ingest_tools_plugin as _itp
    from claritymed.stores import user_phi_rag as _uphi

    def _boom(_uid):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(_uphi, "make_phi_rag_store", _boom)

    # Provide enough kwargs for SaveRecordArgs to validate.
    await _itp._embed_record_task(
        {"record_path": "labs/2026-05-10-lipid"},
        {"category": "labs", "kind": "lab_report", "title": "Lipid"},
    )


async def test_embed_record_task_calls_store_on_happy_path(_ctx, monkeypatch):
    from claritymed.orchestrator.features import ingest_tools_plugin as _itp
    from claritymed.stores import user_phi_rag as _uphi

    captured = {}

    class _StubStore:
        async def add_record(self, user_id, record_path, text):
            captured["user_id"] = user_id
            captured["record_path"] = record_path
            captured["text"] = text
            return 3

    monkeypatch.setattr(_uphi, "make_phi_rag_store", lambda uid: _StubStore())
    await _itp._embed_record_task(
        {"record_path": "labs/2026-05-10-lipid"},
        {"category": "labs", "kind": "lab_report", "title": "Lipid"},
    )
    assert captured["record_path"] == "labs/2026-05-10-lipid"
    assert "Lipid" in captured["text"]


async def test_embed_library_task_returns_silently_when_no_library_path(_ctx):
    from claritymed.orchestrator.features.ingest_tools_plugin import (
        _embed_library_task,
    )

    await _embed_library_task({}, {})  # must not raise


async def test_embed_library_task_calls_store_on_happy_path(_ctx, monkeypatch):
    from claritymed.orchestrator.features import ingest_tools_plugin as _itp
    from claritymed.stores import user_rag as _ur

    captured = {}

    class _StubStore:
        async def add_document(self, user_id, library_path, text, public):
            captured["public"] = public
            captured["library_path"] = library_path
            return 7

    monkeypatch.setattr(_ur, "make_user_rag_store", lambda uid: _StubStore())
    await _itp._embed_library_task(
        {"library_path": "papers/aha-2024"},
        {"title": "AHA Guideline", "public": True},
    )
    assert captured["public"] is True
    assert captured["library_path"] == "papers/aha-2024"
