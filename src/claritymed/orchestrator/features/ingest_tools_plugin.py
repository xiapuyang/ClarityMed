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

``IngestToolsFeature`` is the orchestrator-facing wrapper that
exposes the toolset as a ``FeaturePlugin.as_toolset()``. AskService
collects all plugin toolsets per turn and hands them to the agent.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import TYPE_CHECKING, Any, Callable

from claritymed.context import get_context_or_raise
from claritymed.core.features.base import FeatureMode, TurnContext
from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.audit_payloads import write_payload
from claritymed.core.schemas import (
    Allergy,
    Condition,
    Medication,
    Profile,
    solicitation_for,
)
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

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

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


# --- RAG embed helpers -----------------------------------------------


def _format_record_embed_text(parsed: SaveRecordArgs) -> str:
    """Build the text to embed for a newly saved PHI record."""
    parts = [parsed.title]
    if parsed.notes:
        parts.append(parsed.notes)
    if parsed.extracted_labs:
        lab_strs: list[str] = []
        for lab in parsed.extracted_labs:
            s = f"{lab.name}: {lab.value}"
            if lab.unit:
                s += f" {lab.unit}"
            lab_strs.append(s)
        parts.append("Labs: " + ", ".join(lab_strs))
    if parsed.tags:
        parts.append("Tags: " + ", ".join(parsed.tags))
    return "\n".join(parts)


def _format_library_embed_text(parsed: SaveToLibraryArgs) -> str:
    """Build the text to embed for a newly saved library entry."""
    parts = [parsed.title]
    if parsed.authors:
        parts.append("Authors: " + ", ".join(parsed.authors))
    if parsed.year:
        parts.append(f"Year: {parsed.year}")
    if parsed.tags:
        parts.append("Tags: " + ", ".join(parsed.tags))
    return "\n".join(parts)


async def _embed_record_task(result: dict, kwargs: dict) -> None:
    """Background task: embed record metadata text into ``UserPhiRagStore``."""
    from claritymed.stores.user_phi_rag import make_phi_rag_store

    try:
        _, user_id, _ = get_context_or_raise()
        record_path = result.get("record_path", "")
        if not record_path:
            return
        parsed = SaveRecordArgs.model_validate(kwargs)
        text = _format_record_embed_text(parsed)
        if not text.strip():
            return
        store = make_phi_rag_store(user_id)
        n = await store.add_record(user_id, record_path, text)
        logger.debug("embed_record: %s → %d chunks", record_path, n)
    except Exception:  # noqa: BLE001
        logger.warning("embed_record: background task failed", exc_info=True)


async def _embed_library_task(result: dict, kwargs: dict) -> None:
    """Background task: embed library metadata text into ``UserRagStore``."""
    from claritymed.stores.user_rag import make_user_rag_store

    try:
        _, user_id, _ = get_context_or_raise()
        library_path = result.get("library_path", "")
        if not library_path:
            return
        parsed = SaveToLibraryArgs.model_validate(kwargs)
        text = _format_library_embed_text(parsed)
        if not text.strip():
            return
        store = make_user_rag_store(user_id)
        n = await store.add_document(user_id, library_path, text, public=parsed.public)
        logger.debug("embed_library: %s → %d chunks", library_path, n)
    except Exception:  # noqa: BLE001
        logger.warning("embed_library: background task failed", exc_info=True)


# --- profile.db-only tools (no manifest, no Qdrant) -----------------


def save_medication(args: dict[str, Any], *, dispatcher: ToolDispatcher) -> dict:
    """LLM-callable: append one medication row to ``profile.db``.

    Validates via ``SaveMedicationArgs``; persists via ProfileStore.
    """
    parsed = dispatcher.validate_args("save_medication", args)
    assert isinstance(parsed, SaveMedicationArgs)
    rid, user_id, _ = get_context_or_raise()
    existing = ProfileStore(user_id).list_medications()
    if any(
        m.end_date is None and m.display.lower() == parsed.name.lower()
        for m in existing
    ):
        return {"ok": False, "reason": "no_change"}
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
    existing = ProfileStore(user_id).list_allergies()
    if any(
        a.end_date is None and a.substance.lower() == parsed.substance.lower()
        for a in existing
    ):
        return {"ok": False, "reason": "no_change"}
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
    existing = ProfileStore(user_id).list_conditions()
    if any(
        c.end_date is None and c.display.lower() == parsed.display.lower()
        for c in existing
    ):
        return {"ok": False, "reason": "no_change"}
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
    try:
        current = ProfileStore(user_id).get_profile()
        if current is not None:
            current_val = getattr(current, parsed.field, None)
            if current_val is not None:
                # Coerce the incoming value through the same path ProfileStore uses
                # so we compare date↔date, float↔float, bool↔bool.
                data = current.model_dump(mode="python")
                data[parsed.field] = parsed.value
                coerced_val = getattr(Profile.model_validate(data), parsed.field, None)
                if coerced_val == current_val:
                    return {"ok": False, "reason": "no_change"}
                # Downgrade guard for birth_date: refuse to replace a precise
                # date (month≠1 or day≠1) with a year-only approximation (01-01).
                if parsed.field == "birth_date" and isinstance(coerced_val, date):
                    if (
                        coerced_val.month == 1
                        and coerced_val.day == 1
                        and isinstance(current_val, date)
                        and (current_val.month != 1 or current_val.day != 1)
                    ):
                        return {"ok": False, "reason": "no_change"}
    except Exception:  # noqa: BLE001
        pass  # Let the write proceed if the guard itself fails
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


def _validate_ingest_prompts(registry) -> None:
    """Assert every ingest tool has a bilingual prompt YAML loaded.

    Fail-loud at toolset build time rather than at first model call —
    a missing YAML is a deployment misconfiguration, not a per-turn
    error to retry. Surfaces as ``RuntimeError`` listing the gaps so
    the TUI can show it once instead of seven warning lines.
    """
    missing: list[str] = []
    for tool_name in INGEST_TOOLS:
        prompt_name = f"{tool_name}_tool"
        for lang in ("en", "zh"):
            try:
                registry.get(prompt_name, language=lang)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                missing.append(f"{prompt_name}.{lang}")
    if missing:
        raise RuntimeError(
            "Missing ingest tool prompts: "
            + ", ".join(missing)
            + ". Add the corresponding YAML(s) under core/prompts/store/ "
            "before wiring ingest_factory."
        )


def build_ingest_toolset(
    dispatcher: ToolDispatcher,
    approval_required_func=None,
    *,
    language: str = "en",
):
    """Bundle the 7 tools into an ``ApprovalRequiredToolset``.

    ``approval_required_func`` is the project-specific gate that decides
    whether a call needs the modal or can run immediately. Unit 7 wires
    this against ``SettingsStore`` (TTL'd allow rules) + the dispatcher.
    Default ``None`` ⇒ raw ``FunctionToolset`` with no approval gate
    (used by ``cli tool --auto-approve``, eval harnesses, and tests).

    Each tool's ``description=`` is loaded from
    ``core/prompts/store/<tool_name>_tool.yaml`` so the LLM sees prose
    semantics (WHY/WHEN) alongside pydantic-ai's auto-injected JSON
    schema (WHAT). The YAMLs are immutable per-version; the registry
    handles language selection.
    """
    from pydantic_ai.tools import Tool
    from pydantic_ai.toolsets import ApprovalRequiredToolset, FunctionToolset

    from claritymed.core.prompts.registry import PromptRegistry

    registry = PromptRegistry()

    # Fail-loud check: each of the seven tools must have a prompt YAML
    # in both EN and ZH. A missing YAML at runtime would silently
    # degrade the LLM to schema-only descriptions; instead we abort at
    # toolset build time so the regression is caught on app start, not
    # mid-turn. Cheap (registry already in memory).
    _validate_ingest_prompts(registry)

    def _description_for(tool_name: str) -> str:
        # ``_validate_ingest_prompts`` already asserted presence, so a
        # failure here is structural (e.g. registry corruption) — let
        # the exception propagate rather than swallow it.
        return registry.get(f"{tool_name}_tool", language=language)  # type: ignore[arg-type]

    # pydantic-ai Tool accepts plain callables; we bind ``dispatcher`` via
    # a closure so the registered signature matches the tool args schema.
    # ``save_record`` and ``save_to_library`` get async wrappers so they
    # can fire-and-forget a RAG embed task without blocking the agent loop.
    _EMBED_HOOKS: dict[str, Any] = {
        "save_record": _embed_record_task,
        "save_to_library": _embed_library_task,
    }

    tools = []
    for name, impl in INGEST_TOOLS.items():
        embed_hook = _EMBED_HOOKS.get(name)

        def _make(impl_fn, tool_name, hook=None):
            if hook is not None:

                async def _entry(**kwargs):
                    result = impl_fn(kwargs, dispatcher=dispatcher)
                    if isinstance(result, dict) and result.get("ok") is not False:
                        asyncio.create_task(hook(result, kwargs))
                    return result
            else:

                def _entry(**kwargs):  # type: ignore[misc]
                    return impl_fn(kwargs, dispatcher=dispatcher)

            _entry.__name__ = tool_name
            return _entry

        tools.append(
            Tool(
                _make(impl, name, hook=embed_hook),
                name=name,
                description=_description_for(name) or None,
            )
        )

    inner = FunctionToolset(tools)
    if approval_required_func is None:
        return inner
    return ApprovalRequiredToolset(inner, approval_required_func=approval_required_func)


# --- FeaturePlugin wrapper -------------------------------------------


class IngestToolsFeature:
    """Plugin exposing the seven write tools behind the approval gate.

    The plugin owns the dispatcher + the ``approval_required_func``
    closure; both are constructed once and reused across turns. The
    LLM-facing toolset is rebuilt on each ``as_toolset()`` call because
    pydantic-ai's ``FunctionToolset`` does not document re-entrancy
    across concurrent agent runs (cheap to rebuild — just rewires the
    seven closures) and the AskService is per-turn anyway.

    ``approval_required_func`` only consults the allow-rule store; it
    must never call into a UI (pydantic-ai invokes it from the tool-
    execution path, where blocking on a modal would deadlock the agent
    loop). Deny rules and the modal flow are handled by AskService
    after the ``DeferredToolRequests`` materializes.
    """

    name = "ingest_tools"
    mode: FeatureMode = "tool"

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        approval_required_func: Callable[[Any, Any, dict[str, Any]], bool] | None,
        *,
        language: str = "en",
    ) -> None:
        self._dispatcher = dispatcher
        self._approval_required_func = approval_required_func
        self._language = language

    async def pre_invoke(self, ctx: TurnContext) -> str:
        return ""

    def as_tool(self) -> Callable | None:
        return None

    def as_toolset(self) -> "AbstractToolset[Any] | None":
        return build_ingest_toolset(
            self._dispatcher,
            approval_required_func=self._approval_required_func,
            language=self._language,
        )
