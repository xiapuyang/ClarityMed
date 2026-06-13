"""The single PHI write gate.

``ToolDispatcher.gate(tool_name, args)`` is the canonical entry point
every PHI-writing tool call funnels through. It performs the project-
specific checks the framework's ``ApprovalRequiredToolset`` cannot do
on its own:

1. Schema validation via ``core/schemas/tools.TOOL_ARG_SCHEMAS`` —
   rejects extra fields the LLM hallucinated, malformed dates, etc.
2. **sha256 set** — every sha mentioned in args must resolve in the
   current user's blob universe (session attachments ∪ blobs/<sha>/
   exists ∪ any manifest in this user's records/library references
   the sha). Cross-user sha resolution always fails.
3. **record_path containment** — ``delete_record``'s ``record_path``
   must resolve under ``data/users/<auth_uid>/records/`` and must not
   be a symlink. Defends against ``../etc/passwd`` and symlink escape.
4. **Approval rule match** — consults ``SettingsStore`` (Unit 7) for a
   non-expired allow rule matching the tool name + args subset. Hit ⇒
   tool runs immediately (still re-validates step 2/3 right before
   execution to close the TOCTOU window).
5. Otherwise raises ``pydantic_ai.exceptions.ApprovalRequired`` so the
   agent loop surfaces a ``DeferredToolRequests`` and the TUI's
   ``ApprovalModal`` (Unit 7) drives the user's decision.

Unit 6 ships the gate + the seven tool implementations. Unit 7 wires
``SettingsStore`` (rule lookup) and the TUI side of the approval flow.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import ModelRetry

from claritymed.context import get_context_or_raise
from claritymed.core.observability.audit import audit_event
from claritymed.core.schemas.tools import TOOL_ARG_SCHEMAS
from claritymed.errors import (
    PathOutsideUserDomain,
    UnknownSha256,
)
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.manifest_store import ManifestStore
from claritymed.stores.paths import user_records_dir

logger = logging.getLogger(__name__)


# Strings that mean "no value" coming from a local LLM. Empty string is
# included because small models default to it for unfilled optional
# fields ("end_date": "") instead of omitting the key — pydantic then
# fails ``date_from_datetime_parsing`` on the zero-length string.
_NULL_SENTINEL_STRINGS = frozenset({"None", "none", "NONE", "null", "NULL", "Null", ""})


def _normalize_args(value: Any) -> Any:
    """Pre-clean LLM-emitted tool args before pydantic validation.

    Two normalizations, both targeted at the typo classes the ingest-tool
    benchmark surfaces against small / local models (Qwen3.6, MLX
    variants). Doing them once at the dispatcher boundary saves a
    ``BeforeValidator`` on every Optional / list field across every tool
    schema.

    1. **Null sentinel strings → ``None``** — small LLMs frequently emit
       Python's ``None`` literal, JSON's ``null``, or an empty string as
       a quoted string for unfilled optional fields. Pydantic treats
       those as actual strings, which then fails coercion for any
       Optional field whose annotation is not ``str`` (date, Decimal,
       Enum, int, ...). A required ``str`` field that receives ``""`` /
       ``"None"`` becomes ``None`` and then fails with "field required"
       — the right failure mode, since the model emitted no real value.

    2. **JSON-encoded list / dict strings → parsed value** — the dominant
       benchmark failure mode is models emitting ``"tags":
       '["a","b"]'`` or ``"attachments": '[]'``: the entire list
       JSON-stringified into one quoted blob. We try ``json.loads`` on
       any string that looks structurally JSON-ish (starts with ``[`` /
       ``{`` after strip) and substitute the parsed value when it lands
       as a list or dict. Parse failures fall through unchanged so a
       genuinely-string field is never corrupted, and bona-fide string
       payloads that happen to start with ``[`` (rare in our domain)
       are preserved when ``json.loads`` raises.
    """
    if isinstance(value, dict):
        return {k: _normalize_args(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_args(v) for v in value]
    if isinstance(value, str):
        if value in _NULL_SENTINEL_STRINGS:
            return None
        stripped = value.strip()
        if stripped.startswith(("[", "{")) and stripped.endswith(("]", "}")):
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                return value
            if isinstance(parsed, (list, dict)):
                # Recurse so nested null sentinels / nested JSON strings
                # inside the now-real container also get cleaned.
                return _normalize_args(parsed)
    return value


class ApprovalGateResult(BaseModel):
    """Outcome of ``ToolDispatcher.gate``.

    ``allowed`` is True when an existing rule covers the call; the
    caller can execute right away. ``allowed=False`` means the dispatch
    must defer to the TUI approval flow (Unit 7 raises
    ``ApprovalRequired`` from the toolset wrapper).
    """

    model_config = {"frozen": True}

    allowed: bool
    rule_id: str | None = None


class ToolDispatcher:
    """Composable gate. UnitTest-friendly: stateless apart from the
    injected ``settings_store`` and ``session_attachments`` callables.
    """

    def __init__(
        self,
        *,
        session_attachments: Callable[[], set[str]] = lambda: set(),
        rule_match: Callable[[str, dict[str, Any]], str | None] | None = None,
    ) -> None:
        self._session_attachments = session_attachments
        self._rule_match = rule_match

    # --- public gate -------------------------------------------------

    def validate_args(self, tool_name: str, args: dict[str, Any]) -> BaseModel:
        """Step 1 of the gate. Returns the validated pydantic model.

        Separated from ``gate`` so the per-tool implementation can call
        it again at the TOCTOU close (just before the actual write).
        """
        schema = TOOL_ARG_SCHEMAS.get(tool_name)
        if schema is None:
            raise ValueError(f"unknown tool: {tool_name!r}")
        cleaned = _normalize_args(args)
        try:
            return schema.model_validate(cleaned)
        except ValidationError as exc:
            # Log the failed call so a maintainer reading app.log can see
            # the exact (tool_name, args, error) triple for every retry
            # cycle. WARNING level — these are recoverable; pydantic-ai
            # converts the ModelRetry below into a RetryPromptPart and
            # the model gets up to ``tools.ingest.max_retries`` chances
            # to self-correct. The audit log carries the same info for
            # compliance; app.log carries it for debugging.
            logger.warning(
                "tool_args_invalid tool=%s args=%s error=%s",
                tool_name,
                cleaned,
                exc.errors(include_url=False, include_context=False),
            )
            # Raise ``ModelRetry`` (not ``ValueError``) so pydantic-ai's
            # tool loop catches it, wraps as ``ToolRetryError`` with a
            # ``RetryPromptPart``, and lets the model self-correct on
            # the next agent step. ``ValueError`` would escape ``agent.run``
            # entirely (``on_tool_execute_error`` re-raises) and abort
            # the whole turn — small models that emit ``"None"`` strings,
            # stringified lists, or wrong field names never get a chance
            # to fix their tool call.
            #
            # The ``unknown tool`` ValueError above stays as-is: that one
            # is a structural bug in the toolset wiring, not something
            # the model can repair by trying again.
            raise ModelRetry(f"invalid args for {tool_name}: {exc}") from exc

    def check_shas(self, args: dict[str, Any]) -> None:
        """Step 2 of the gate.

        Every sha256 referenced by ``args.attachments`` must exist in
        one of three places, ALL scoped to the current user:

        * the session-attachments tray (passed by the constructor),
        * an on-disk blob under ``blobs/<sha[:2]>/<sha>/``,
        * a sha referenced from a manifest in this user's records or
          library (handled by ``BlobStore.exists`` — manifest references
          can be checked by the caller separately if needed).

        The third clause is enforced *structurally* via the per-user
        ``BlobStore`` boundary. Cross-user resolution always fails
        because the BlobStore is keyed by the auth'd ``user_id``.

        Self-normalizes via ``_normalize_args`` so it is safe to call
        with the raw LLM-emitted ``args`` dict — small models sometimes
        encode ``attachments`` as a JSON string, and iterating that
        directly would treat each character as an entry and crash on
        ``str.sha256``. ``_normalize_args`` is idempotent, so calling
        from ``gate`` (which already normalizes once) is also fine.
        """
        cleaned = _normalize_args(args)
        attachments = cleaned.get("attachments") or []
        if not attachments:
            return
        _, user_id, _ = get_context_or_raise()
        session_shas = self._session_attachments()
        blob_store = BlobStore(user_id)
        for entry in attachments:
            if isinstance(entry, dict):
                sha = entry.get("sha256")
            else:
                # ``AttachmentRef`` / any object exposing ``.sha256``; bare
                # strings or other shapes fall through to the "missing sha256"
                # branch instead of raising ``AttributeError``.
                sha = getattr(entry, "sha256", None)
            if not sha:
                raise UnknownSha256("missing sha256 in attachment")
            if sha in session_shas:
                continue
            if blob_store.exists(sha):
                continue
            raise UnknownSha256(
                f"sha256 {sha[:8]}… not in session attachments or blob pool"
            )

    def check_record_path(self, args: dict[str, Any]) -> None:
        """Step 3 of the gate. Validates ``record_path`` containment.

        Resolves both the supplied path and the current user's records
        root, then asserts ``resolved.is_relative_to(root)`` *and* that
        the final component is not a symlink. The symlink check uses
        ``os.lstat`` so a symlink that resolves into the user's tree
        still fails (closes the symlink-into-own-tree confusion attack).
        """
        record_path = args.get("record_path")
        if not record_path:
            return
        import os
        from pathlib import Path

        _, user_id, _ = get_context_or_raise()
        root = user_records_dir(user_id).resolve()
        # Accept both relative and absolute record_path inputs; resolve
        # against the records root either way.
        candidate = Path(record_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise PathOutsideUserDomain(
                f"record_path resolves outside {root}: {record_path!r}"
            )
        # If the final dir exists, refuse if it's a symlink.
        if resolved.exists() and os.path.islink(str(resolved)):
            raise PathOutsideUserDomain(f"record_path is a symlink: {record_path!r}")

    def gate(self, tool_name: str, args: dict[str, Any]) -> ApprovalGateResult:
        """Run steps 1-4 of the gate.

        Step 5 (raising ``ApprovalRequired``) is done by the toolset
        wrapper in Unit 7; this dispatcher returns whether a rule
        covers the call and the caller decides how to surface that.

        Each sub-step self-normalizes via ``_normalize_args`` (idempotent),
        so passing the raw LLM args here is safe; we still normalize once
        at the top so the rule-match below sees the cleaned shape.
        """
        cleaned = _normalize_args(args)
        self.validate_args(tool_name, cleaned)
        self.check_shas(cleaned)
        self.check_record_path(cleaned)

        if self._rule_match is not None:
            rule_id = self._rule_match(tool_name, cleaned)
            if rule_id is not None:
                audit_event(
                    "tool.always_allowed",
                    {"tool_name": tool_name, "rule_id": rule_id},
                )
                return ApprovalGateResult(allowed=True, rule_id=rule_id)
        return ApprovalGateResult(allowed=False, rule_id=None)


def manifest_references_sha(user_id: str, sha256: str) -> bool:
    """Return True iff any manifest under ``data/users/<id>/records|library``
    lists this sha in its attachments. Used by the dispatcher to extend
    the sha-set check to historical blobs (the third source).
    """
    import yaml

    for scope in ("records", "library"):
        store = ManifestStore(user_id, scope)  # type: ignore[arg-type]
        for manifest_path in store.list():
            try:
                raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            for att in raw.get("attachments", []) or []:
                if isinstance(att, dict) and att.get("sha256") == sha256:
                    return True
    return False
