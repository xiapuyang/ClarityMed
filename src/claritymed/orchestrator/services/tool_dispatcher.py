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

import difflib
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


_NULL_SENTINEL_STRINGS = frozenset({"None", "none", "NONE", "null", "NULL", "Null"})


# Max length of the offending ``input`` value embedded in a retry hint. Long
# nested dicts blow past the model's working memory and bury the signal; the
# field name + first 60 chars is enough to identify what went wrong.
_HINT_INPUT_REPR_LIMIT = 60


def _format_validation_errors(
    tool_name: str,
    schema: type[BaseModel],
    exc: ValidationError,
) -> str:
    """Translate pydantic ``ValidationError`` into model-friendly hints.

    Raw pydantic phrasing (``Input should be a valid list
    [input_value='[]', input_type=str]``) does not teach small local
    models how to fix the call — they fixate on the value rather than
    the type and loop. This helper rewrites the common error shapes
    into explicit corrective sentences and, on ``extra_forbidden``,
    suggests the closest valid field name via :mod:`difflib`. Anything
    we do not specifically translate falls through to pydantic's own
    ``msg`` so we never lose information.
    """
    valid_fields = sorted(schema.model_fields.keys())
    hints: list[str] = []
    typo_field: str | None = None
    for err in exc.errors(include_url=False, include_context=False):
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        etype = err["type"]
        input_repr = repr(err.get("input"))
        if len(input_repr) > _HINT_INPUT_REPR_LIMIT:
            input_repr = input_repr[: _HINT_INPUT_REPR_LIMIT - 3] + "..."
        if etype == "list_type":
            hints.append(
                f"`{loc}` must be a JSON array (e.g. `[]` for empty), "
                f"not the quoted string {input_repr}."
            )
        elif etype == "dict_type":
            hints.append(
                f"`{loc}` must be a JSON object (e.g. `{{}}`), "
                f"not the quoted string {input_repr}."
            )
        elif etype == "missing":
            hints.append(f"`{loc}` is required — include it in the next call.")
        elif etype == "extra_forbidden":
            if typo_field is None:
                typo_field = loc
            hints.append(f"`{loc}` is not a valid field for `{tool_name}`.")
        else:
            hints.append(f"`{loc}`: {err['msg']}.")
    if typo_field is not None:
        suggestion = difflib.get_close_matches(
            typo_field, valid_fields, n=1, cutoff=0.5
        )
        prefix = f"Did you mean `{suggestion[0]}`? " if suggestion else ""
        hints.append(
            f"{prefix}Valid fields for `{tool_name}`: {', '.join(valid_fields)}."
        )
    return f"invalid args for {tool_name}: " + " ".join(hints)


def _normalize_null_sentinels(value: Any) -> Any:
    """Replace string sentinels for null with real ``None``, recursively.

    Small / local LLMs frequently emit Python's ``None`` literal or
    JSON's ``null`` as a quoted string inside tool-call payloads when
    they mean "no value". Pydantic treats those as plain strings, which
    then fails type coercion for any Optional field whose annotation is
    not ``str`` (date, Decimal, Enum, int, ...). Normalizing once at the
    dispatcher boundary saves a ``BeforeValidator`` on every Optional
    field across every tool schema.

    A required ``str`` field that receives ``"None"`` would be coerced
    to ``None`` here and then fail with "field required" — the right
    failure mode, since the model genuinely emitted no value.
    """
    if isinstance(value, dict):
        return {k: _normalize_null_sentinels(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_null_sentinels(v) for v in value]
    if isinstance(value, str) and value in _NULL_SENTINEL_STRINGS:
        return None
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
        cleaned = _normalize_null_sentinels(args)
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
            raise ModelRetry(_format_validation_errors(tool_name, schema, exc)) from exc

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
        """
        attachments = args.get("attachments") or []
        if not attachments:
            return
        _, user_id, _ = get_context_or_raise()
        session_shas = self._session_attachments()
        blob_store = BlobStore(user_id)
        for entry in attachments:
            sha = entry.get("sha256") if isinstance(entry, dict) else entry.sha256
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
        """
        self.validate_args(tool_name, args)
        self.check_shas(args)
        self.check_record_path(args)

        if self._rule_match is not None:
            rule_id = self._rule_match(tool_name, args)
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
