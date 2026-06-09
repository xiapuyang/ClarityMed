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
    # mode lifecycle (orchestrator services)
    "mode.ingest",
    "mode.ask",
    "mode.ask.scrub",
    "mode.rag",
    "mode.routed",
    "mode.cancelled",
    "mode.ask.history_trimmed",
    # Tool-mode compliance: LLM wrote "I will retrieve..." but never
    # actually invoked retrieve_medical_literature this turn. Grep to
    # quantify per-provider compliance with the prompt's tool protocol
    # and decide whether to switch a misbehaving provider to
    # deterministic mode.
    "mode.ask.tool_announced_but_skipped",
    # RAG retrieval lifecycle
    "rag.retrieval",
    "rag.retrieval.failed",
    "rag.rerank.fallback",
    # LLM call lifecycle — paired with mode.ask. ``llm.call.start`` marks
    # the boundary between RAG retrieval done and the LLM call begun, so
    # post-mortem latency analysis can separate retrieval cost from
    # model TTFT without inferring from timestamps.
    "llm.call.start",
    # OCR extraction — one event per extract_text call.
    # payload: provider, file, size_bytes, status, duration_ms,
    #          chars (on success), error (on failure),
    #          fallback (bool, true when image default failed).
    "ocr.extract",
    # Privacy-filter model inference — one event per _layer_model call.
    # payload: backend ("onnx"|"torch"), status ("ok"|"error"), duration_ms,
    #          hits (entity count), chars_in, chars_out (on success),
    #          error (on failure).
    "scrub.privacy_filter",
]


class AuditEvent(BaseModel):
    """One audit line. Serialized as JSON, one event per log line.

    ``trace_id`` / ``span_id`` are populated when an OpenTelemetry span is
    active at audit time. They let an operator pivot from a grep hit in
    ``audit.log`` straight to the matching trace in Phoenix / Tempo without
    timestamp gymnastics. Absent when tracing is off — the field is null,
    not missing, so downstream parsers don't need to care.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: AuditKind
    payload: dict[str, Any] = Field(default_factory=dict)
    request_id: str
    user_id: str
    language: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: str | None = None
    span_id: str | None = None


def _current_trace_context() -> tuple[str | None, str | None]:
    """Return (trace_id_hex, span_id_hex) for the currently active span,
    or ``(None, None)`` when no provider / no active span. Never raises —
    a broken tracer must not break the audit path.
    """
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if not ctx.is_valid:
            return None, None
        return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
    except Exception:  # noqa: BLE001
        return None, None


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
    trace_id, span_id = _current_trace_context()
    event = AuditEvent(
        kind=kind,
        payload=payload or {},
        request_id=rid,
        user_id=uid,
        language=lang,
        trace_id=trace_id,
        span_id=span_id,
    )
    get_audit_logger().info(event.model_dump_json())
    return event
