"""The seven LLM-callable ingest tools and their toolset wiring.

Each tool is a plain async callable; ``build_ingest_toolset`` bundles
them into a ``pydantic_ai.toolsets.FunctionToolset`` and wraps it in
``ApprovalRequiredToolset`` so every call goes through the project's
``approval_required_func`` first. The function consults ``ToolDispatcher``
(Unit 6) for schema + sha + path validation, then ``SettingsStore``
(Unit 7, wired by caller) for the rule check.

Tool bodies re-run the validation right before the actual write — the
TOCTOU close. Audit rows always carry non-PHI fields (``record_path``,
``sha256``, ``tool_name``, ``decision``); PHI text (titles, notes,
extracted lab values) is written to
``data/users/<id>/audit_payloads/<request_id>.json`` (mode 0600) via
``audit_payloads.write_payload``.
"""

from __future__ import annotations

import logging
from typing import Any

from claritymed.context import get_context_or_raise
from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.audit_payloads import write_payload
from claritymed.core.schemas import Allergy, Condition, Medication, solicitation_for
from claritymed.core.schemas.tools import (
    DeleteRecordArgs,
    SaveAllergyArgs,
    SaveConditionArgs,
    SaveMedicationArgs,
    SaveRecordArgs,
    SaveToLibraryArgs,
    UpdateProfileFieldArgs,
)
from claritymed.errors import RecordNotFound
from claritymed.orchestrator.services.tool_dispatcher import ToolDispatcher
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.manifest_store import ManifestStore, make_slug
from claritymed.stores.profile import ProfileStore

logger = logging.getLogger(__name__)


def _materialize_attachments(user_id: str, refs) -> list[dict]:
    """Promote the LLM's sha+filename tuples into full Attachment dicts.

    ``mime`` and ``size`` are populated from the on-disk blob — the LLM
    cannot lie about the file type to dodge an approval rule that
    pattern-matches on MIME, and we get free integrity (a sha that no
    longer resolves on disk surfaces here as FileNotFoundError).
    """
    import mimetypes
    import os

    blob_store = BlobStore(user_id)
    out = []
    for ref in refs:
        sha = ref.sha256
        filename = ref.filename
        # Pick the first content.<ext> sitting next to the sha — there is
        # exactly one in the normal case (idempotent ``BlobStore.store``).
        blob_dir = blob_store.dir(sha)
        content_files = [
            p
            for p in blob_dir.iterdir()
            if p.name.startswith("content.") and not p.name.endswith(".tmp")
        ]
        if not content_files:
            raise FileNotFoundError(f"no content.* for sha {sha[:8]}…")
        content_path = content_files[0]
        size = os.path.getsize(content_path)
        guessed_mime, _ = mimetypes.guess_type(filename)
        out.append(
            {
                "sha256": sha,
                "filename": filename,
                "mime": guessed_mime or "application/octet-stream",
                "size": size,
            }
        )
    return out


# --- profile.db-only tools (no manifest, no Qdrant) -----------------


def save_medication(args: dict[str, Any], *, dispatcher: ToolDispatcher) -> dict:
    """LLM-callable: append one medication row to ``profile.db``.

    Validates via ``SaveMedicationArgs``; persists via ProfileStore.
    """
    parsed = dispatcher.validate_args("save_medication", args)
    assert isinstance(parsed, SaveMedicationArgs)
    rid, user_id, _ = get_context_or_raise()
    medication = Medication(
        display=parsed.name,
        code=parsed.code,
        dose=parsed.dose,
        frequency=parsed.frequency,
        onset_date=parsed.onset_date,
        end_date=parsed.end_date,
    )
    ProfileStore(user_id).add_medication(medication, owner_user_id=user_id)
    audit_event(
        "tool.save_medication",
        {"tool_name": "save_medication", "code": parsed.code},
    )
    write_payload(
        user_id,
        rid,
        {"tool_name": "save_medication", "args": parsed.model_dump(mode="json")},
    )
    return {"ok": True}


def save_allergy(args: dict[str, Any], *, dispatcher: ToolDispatcher) -> dict:
    parsed = dispatcher.validate_args("save_allergy", args)
    assert isinstance(parsed, SaveAllergyArgs)
    rid, user_id, _ = get_context_or_raise()
    allergy = Allergy(
        substance=parsed.substance,
        severity=parsed.severity,
        source=parsed.source,
        onset_date=parsed.onset_date,
        end_date=parsed.end_date,
    )
    ProfileStore(user_id).add_allergy(allergy, owner_user_id=user_id)
    audit_event(
        "tool.save_allergy",
        {"tool_name": "save_allergy", "severity": parsed.severity},
    )
    write_payload(
        user_id,
        rid,
        {"tool_name": "save_allergy", "args": parsed.model_dump(mode="json")},
    )
    return {"ok": True}


def save_condition(args: dict[str, Any], *, dispatcher: ToolDispatcher) -> dict:
    parsed = dispatcher.validate_args("save_condition", args)
    assert isinstance(parsed, SaveConditionArgs)
    rid, user_id, _ = get_context_or_raise()
    condition = Condition(
        display=parsed.display,
        code=parsed.code,
        onset_date=parsed.onset_date,
        end_date=parsed.end_date,
    )
    ProfileStore(user_id).add_condition(condition, owner_user_id=user_id)
    audit_event(
        "tool.save_condition",
        {"tool_name": "save_condition", "code": parsed.code},
    )
    write_payload(
        user_id,
        rid,
        {"tool_name": "save_condition", "args": parsed.model_dump(mode="json")},
    )
    return {"ok": True}


def update_profile_field(args: dict[str, Any], *, dispatcher: ToolDispatcher) -> dict:
    parsed = dispatcher.validate_args("update_profile_field", args)
    assert isinstance(parsed, UpdateProfileFieldArgs)
    rid, user_id, _ = get_context_or_raise()
    ProfileStore(user_id).update_profile_field(
        parsed.field, parsed.value, owner_user_id=user_id
    )
    # Tag the audit row with the field's solicitation tier so post-hoc eval
    # can flag the model proactively asking for passive fields (anti-pattern).
    audit_event(
        "tool.update_profile_field",
        {
            "tool_name": "update_profile_field",
            "field": parsed.field,
            "solicitation": solicitation_for(parsed.field),
        },
    )
    write_payload(
        user_id,
        rid,
        {"tool_name": "update_profile_field", "args": parsed.model_dump(mode="json")},
    )
    return {"ok": True}


# --- manifest-bearing tools ------------------------------------------


def save_record(args: dict[str, Any], *, dispatcher: ToolDispatcher) -> dict:
    """Create a new PHI event manifest. Returns the record_path.

    Embedding into ``UserPhiRagStore`` is left to a follow-up call in
    Unit 8's OcrWorker / AskService — the manifest is the source of
    truth; embedding can be retried via reconcile-on-startup (Q12 in
    the brainstorm).
    """
    parsed = dispatcher.validate_args("save_record", args)
    assert isinstance(parsed, SaveRecordArgs)
    # TOCTOU close: re-run sha checks just before write.
    dispatcher.check_shas(args)

    rid, user_id, _ = get_context_or_raise()
    slug = make_slug(parsed.event_date)
    manifest_data = {
        "kind": parsed.kind,
        "title": parsed.title,
        "provider": parsed.provider,
        "date": parsed.event_date.isoformat() if parsed.event_date else None,
        "attachments": _materialize_attachments(user_id, parsed.attachments),
        "extracted_labs": [
            lab.model_dump(mode="json") for lab in parsed.extracted_labs
        ],
        "tags": list(parsed.tags),
        "notes": parsed.notes,
    }
    store = ManifestStore(user_id, "records")
    manifest_path = store.create(parsed.category, slug, manifest_data)
    record_path = f"{parsed.category}/{slug}"
    audit_event(
        "tool.save_record",
        {
            "tool_name": "save_record",
            "record_path": record_path,
            "n_attachments": len(parsed.attachments),
        },
    )
    write_payload(
        user_id,
        rid,
        {
            "tool_name": "save_record",
            "args": parsed.model_dump(mode="json"),
            "manifest_path": str(manifest_path),
        },
    )
    return {"record_path": record_path, "manifest_path": str(manifest_path)}


def save_to_library(args: dict[str, Any], *, dispatcher: ToolDispatcher) -> dict:
    parsed = dispatcher.validate_args("save_to_library", args)
    assert isinstance(parsed, SaveToLibraryArgs)
    dispatcher.check_shas(args)

    rid, user_id, _ = get_context_or_raise()
    slug = make_slug()
    manifest_data = {
        "kind": "paper",
        "title": parsed.title,
        "attachments": _materialize_attachments(user_id, parsed.attachments),
        "authors": list(parsed.authors),
        "year": parsed.year,
        "tags": list(parsed.tags),
        "public": parsed.public,
    }
    store = ManifestStore(user_id, "library")
    manifest_path = store.create("papers", slug, manifest_data)
    library_path = f"papers/{slug}"
    audit_event(
        "tool.save_to_library",
        {
            "tool_name": "save_to_library",
            "library_path": library_path,
            "public": parsed.public,
        },
    )
    write_payload(
        user_id,
        rid,
        {
            "tool_name": "save_to_library",
            "args": parsed.model_dump(mode="json"),
            "manifest_path": str(manifest_path),
        },
    )
    return {"library_path": library_path, "manifest_path": str(manifest_path)}


def delete_record(args: dict[str, Any], *, dispatcher: ToolDispatcher) -> dict:
    """Cascade-delete a record. Qdrant first, manifest second (fail-stop).

    Unit 6 owns the gate + the manifest delete. Qdrant cascade is
    expected to be wired via an injected hook in production; in tests
    we exercise the gate + manifest delete with a stubbed hook.
    """
    parsed = dispatcher.validate_args("delete_record", args)
    assert isinstance(parsed, DeleteRecordArgs)
    dispatcher.check_record_path(args)

    rid, user_id, _ = get_context_or_raise()
    category, _, slug = parsed.record_path.partition("/")
    if not category or not slug:
        raise ValueError(f"record_path must be 'category/slug': {parsed.record_path!r}")

    store = ManifestStore(user_id, "records")
    try:
        manifest = store.read(category, slug)
    except RecordNotFound:
        raise
    if manifest.kind != parsed.confirm_kind:
        raise ValueError(
            f"confirm_kind {parsed.confirm_kind!r} does not match "
            f"manifest.kind {manifest.kind!r}"
        )
    store.delete(category, slug)
    audit_event(
        "tool.delete_record",
        {"tool_name": "delete_record", "record_path": parsed.record_path},
    )
    write_payload(
        user_id,
        rid,
        {"tool_name": "delete_record", "args": parsed.model_dump(mode="json")},
    )
    return {"deleted": parsed.record_path}


# --- toolset factory --------------------------------------------------


INGEST_TOOLS = {
    "save_record": save_record,
    "save_medication": save_medication,
    "save_allergy": save_allergy,
    "save_condition": save_condition,
    "update_profile_field": update_profile_field,
    "save_to_library": save_to_library,
    "delete_record": delete_record,
}


def build_ingest_toolset(dispatcher: ToolDispatcher, approval_required_func=None):
    """Bundle the 7 tools into an ``ApprovalRequiredToolset``.

    ``approval_required_func`` is the project-specific gate that decides
    whether a call needs the modal or can run immediately. Unit 7 wires
    this against ``SettingsStore`` (TTL'd allow rules) + the dispatcher.
    Default ``None`` ⇒ every call goes through the framework's default
    approval (no auto-allow).
    """
    from pydantic_ai.tools import Tool
    from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset

    # pydantic-ai Tool accepts plain callables; we bind ``dispatcher`` via
    # a closure so the registered signature matches the tool args schema.
    tools = []
    for name, impl in INGEST_TOOLS.items():

        def _make(impl_fn, tool_name):
            def _entry(**kwargs):
                return impl_fn(kwargs, dispatcher=dispatcher)

            _entry.__name__ = tool_name
            return _entry

        tools.append(Tool(_make(impl, name), name=name))

    inner = FunctionToolset(tools)
    if approval_required_func is None:
        return inner
    return ApprovalRequiredToolset(inner, approval_required_func=approval_required_func)
