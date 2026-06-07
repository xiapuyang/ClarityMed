"""Structured audit events: one JSON line per business-meaningful action.

The format is deliberately OTel-friendly (kind + payload + ts + the three
request ids), so later OTel / Phoenix / Langfuse ingest does not need a
separate parser. Every kind is a member of ``AuditKind`` — adding a new one
is a typed change, not a string typo.

Construction enforces that the request ContextVars are set, refusing to
write a "user_id = unknown" audit line. The one exception is ``request_start``
events fired from the FastAPI middleware before user_id can be resolved; those
go through ``audit_event_no_user`` instead.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from claritymed.context import (
    MissingContextError,
    language_ctx,
    request_id_ctx,
    user_id_ctx,
)
from claritymed.core.observability.logging import get_audit_logger

AuditKind = Literal[
    # request lifecycle
    "request_start",
    "request_end",
    # safety / phi
    "redflag_trigger",
    "phi_guard_block",
    "phi_guard_allow",
    # tools
    "tool_invoke",
    "tool_result",
    "retrieval",
    "vision_predict",
    "knowledge_wipe",
    # decisions
    "uncertainty_decision",
    "human_handoff",
    # account / auth
    "account_created",
    "require_admin_pass",
    "require_admin_blocked",
]


class AuditEvent(BaseModel):
    """One audit line. Serialized as JSON, one event per log line."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: AuditKind
    payload: dict[str, Any] = Field(default_factory=dict)
    request_id: str
    user_id: str
    language: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def audit_event(kind: AuditKind, payload: dict[str, Any] | None = None) -> AuditEvent:
    """Build, emit, and return an ``AuditEvent`` from current ContextVars.

    Raises ``MissingContextError`` if any of the three ContextVars is unset —
    the audit trail must never carry an unknown user.
    """
    rid = request_id_ctx.get()
    uid = user_id_ctx.get()
    lang = language_ctx.get()
    if not rid or not uid or not lang:
        raise MissingContextError(
            "audit_event requires request_id / user_id / language to be set"
        )
    event = AuditEvent(
        kind=kind,
        payload=payload or {},
        request_id=rid,
        user_id=uid,
        language=lang,
    )
    get_audit_logger().info(event.model_dump_json())
    return event
