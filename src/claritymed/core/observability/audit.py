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
    # Emergency triage gate (pre-step before agent loop).
    #   redflag.gate_disabled — effective sensitivity resolved to "off".
    #     payload allowlist: requested, effective, reason
    #     reasons: user_preference, cli_override
    #   redflag.reply_missing_action — final reply text did not contain
    #     the suggested-action phrase even after the output_validator
    #     passed. Audit-only tripwire; does not mutate the reply.
    #     payload allowlist: rule_id, level, action_i18n_key
    #   redflag.validator_unrecoverable — pydantic-ai output_validator
    #     exhausted its retry budget; reply passed through with original
    #     text. Operators grep this to spot prompt drift.
    #     payload allowlist: rule_id, level, retries
    "redflag.gate_disabled",
    "redflag.reply_missing_action",
    "redflag.validator_unrecoverable",
    "phi_guard_block",
    "phi_guard_allow",
    # Layer-3 PHI defense (PhiAssertionModel). Fires when a cloud-bound
    # message stream contained PHI that the guard scrubber detected,
    # *before* the inner Model is invoked. Payload: layer_triggered,
    # part_type.
    "phi.leak_detected",
    # Tool dispatcher (Unit 6+7) — single PHI write gate. Every tool
    # call funnels through and produces exactly one of:
    #   tool.always_allowed — a SettingsStore rule covers the call.
    #   tool.approval.granted / .denied / .modified — user-side decision.
    #   tool.auto_approved — headless CLI --auto-approve (severity=high).
    #   tool.<name> — actual write succeeded; payload carries non-PHI fields.
    #   tool.rule_evicted — settings.rules cap reached, oldest evicted.
    #   tool.rule_expired — TTL'd allow-rule lapsed at match time.
    #   tool.cancelled_by_shutdown — TUI dismissed mid-modal.
    "tool.always_allowed",
    "tool.approval.granted",
    "tool.approval.denied",
    "tool.approval.modified",
    "tool.auto_approved",
    "tool.save_record",
    "tool.save_medication",
    "tool.save_allergy",
    "tool.save_condition",
    "tool.update_profile_field",
    "tool.save_to_library",
    "tool.delete_record",
    "tool.delete_record.qdrant_failed",
    "tool.rule_evicted",
    "tool.rule_expired",
    "tool.cancelled_by_shutdown",
    "settings.rules.load_failed",
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
    "mode.ask",
    "mode.ask.scrub",
    "mode.ask.scrub_assembled",
    "mode.rag",
    "mode.cancelled",
    "mode.ask.history_trimmed",
    # Tool-mode compliance: LLM wrote "I will retrieve..." but never
    # actually invoked retrieve_medical_literature this turn. Grep to
    # quantify per-provider compliance with the prompt's tool protocol
    # and decide whether to switch a misbehaving provider to
    # deterministic mode.
    "mode.ask.tool_announced_but_skipped",
    # Citation index out of range — clamped before delivery to user.
    # payload: user_id, valid_max, offending (list of citation indices)
    "ask.citation.out_of_range",
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
    # payload: provider, blob_filename, original_filename (when known),
    #          size_bytes, status, duration_ms, chars (on success),
    #          error (on failure), fallback (bool, true when image
    #          default failed).
    "ocr.extract",
    # Paste-time filetype gate — one event per paste that did NOT match
    # the active chain's accept set on first try.
    # payload: declared_ext, detected_ext (None if Magika failed),
    #          label (Magika class), score, outcome ("recovered" |
    #          "rejected").
    "filetype.detect",
    # Regex scrub — one event per scrub() call that had at least one hit.
    # payload: rule_hits (dict[rule_name, count]).
    "scrub.regex",
    # Privacy-filter model inference — one event per _layer_model call.
    # payload: backend ("onnx"|"torch"), status ("ok"|"error"), duration_ms,
    #          hits (entity count), chars_in, chars_out (on success),
    #          error (on failure).
    "scrub.privacy_filter",
    # Evaluation runs (lm-evaluation-harness). One pair (started/completed)
    # per `claritymed eval` invocation; ``eval.run.failed`` replaces the
    # ``completed`` event when the underlying simple_evaluate raises.
    # payload (started): provider_id, model_name, task_id, limit
    # payload (completed): provider_id, task_id, n_questions, accuracy,
    #                      output_path, duration_s
    # payload (failed): provider_id, task_id, error_type, message
    "eval.run.started",
    "eval.run.completed",
    "eval.run.failed",
    # Per-question wall-clock timeout in ClaritymedRagLM. Fires when a
    # single AskService turn exceeds the adapter's question_timeout_s
    # cap. The question is scored wrong (empty completion → filter
    # misses) and the run continues — one row beats losing the batch.
    # payload: provider_id, model_name, timeout_s, doc_id
    "eval.question.timeout",
    # Delta report comparing a baseline and a with-rag run of the same
    # (provider, task). One row per `claritymed eval delta` invocation.
    # payload: provider_id, task_id, n_questions, baseline_accuracy,
    #          rag_accuracy, delta, regression_count, gain_count,
    #          baseline_path, rag_path, report_path, regressions_sidecar
    "eval.delta.completed",
    # --- symptoms feature (predict_disease_from_symptoms) ---------------
    #
    # KTD-5: the full audit vocabulary lands in one structural commit so
    # downstream units can call ``audit_event(...)`` without per-call enum
    # extensions. Field allowlists below document the keys allowed in
    # each kind's *non-PHI* payload — the Q&A transcript, complaint text,
    # and disease *names* live in audit_payloads/, never here.
    #
    # tool.predict_disease_from_symptoms — tool invocation.
    #   payload allowlist: tool_name, dataset_hint, has_symptom_summary
    "tool.predict_disease_from_symptoms",
    # symptoms.session.started — server accepted the start request.
    #   payload allowlist: dataset_id, model_id, session_id
    "symptoms.session.started",
    # symptoms.session.turn — one question/answer round-trip.
    #   payload allowlist: turn_index, question_id, answer_type
    "symptoms.session.turn",
    # symptoms.session.completed — model reached stop gate.
    #   payload allowlist: dataset_id, model_id, session_id, turns_used,
    #                      severity_tier, top_condition_id  (id, not name)
    "symptoms.session.completed",
    # symptoms.session.cancelled — user cancelled mid-loop OR upstream
    # error (server unreachable, session expired, channel unavailable).
    #   payload allowlist: phase, turn_index, partial_confidence,
    #                      meets_confidence_threshold, severity_override_fired,
    #                      reason
    "symptoms.session.cancelled",
    # symptoms.session.cap_hit — maxstep reached without stop gate firing.
    #   payload allowlist: turns_used, partial_confidence, severity_tier
    "symptoms.session.cap_hit",
    # symptoms.session.ineligible — eligibility filter rejected the
    # complaint (lifecycle outcome of the eligibility step).
    #   payload allowlist: dataset_id, strategy_id, reason
    #   reasons: out_of_scope, demographic_mismatch, unknown_hint,
    #            no_interactive_channel, server_error, strategy_unavailable
    "symptoms.session.ineligible",
    # symptoms.eligibility.checked — eligibility decision telemetry,
    # regardless of outcome. One per pre-invoke run.
    #   payload allowlist: dataset_id, strategy_id, confidence, eligible
    "symptoms.eligibility.checked",
    # symptoms.eligibility.strategy_unavailable — runtime dependency
    # missing (NoOpTermService, provider down, sidecar JSON absent).
    #   payload allowlist: dataset_id, strategy_id, reason
    "symptoms.eligibility.strategy_unavailable",
    # symptoms.safety_keywords.missing — KTD-2 audit-only signal. The
    # post_process hook scanned a tier ≤2 reply and found no keyword
    # from configs/i18n/<lang>/symptoms.yaml symptoms.safety_keywords.<tier>.
    #   payload allowlist: max_severity, tier, observed_leading_chars
    #   (observed_leading_chars is the count of characters scanned, not
    #    the reply text — that would be PHI-adjacent)
    "symptoms.safety_keywords.missing",
    # --- vision feature (detect_disease_from_image) ---------------------
    #
    # KTD-V1: the full audit vocabulary lands in one structural commit so
    # downstream units can call ``audit_event(...)`` without per-call enum
    # extensions. PHI-bearing fields (image bytes, OCR text, segmentation
    # masks) live in audit_payloads/ only; this enum carries non-PHI keys.
    #
    # tool.detect_disease_from_image — tool invocation entry.
    #   payload allowlist: tool_name, disease_id, model_id_hint,
    #                      image_sha_prefix (8-char only, never full)
    "tool.detect_disease_from_image",
    # vision_detection_event — lifecycle event for one tool call. Fires
    # multiple times within a single tool body run (phase: confirm,
    # fallback, post_process) so an operator can grep the request_id
    # and see the whole flow.
    #   payload allowlist: disease_id, model_id, server_id, elapsed_ms,
    #                      top1, top1_prob, confidence_tier, cancer_status,
    #                      clinical_action, quality_gate_passed,
    #                      fallback_count, phase, outcome, reason,
    #                      ocr_override_fired, user_declined,
    #                      specialist_keyword_match
    "vision_detection_event",
    # vision_skipped_ocr_override — KTD-V6 short-circuit fired.
    #   payload allowlist: disease_id, image_sha_prefix
    "vision_skipped_ocr_override",
    # vision_shadow_inference — KTD-V9 opt-in inference recorded for
    # offline eval. Lifecycle marker — actual PHI lands in audit_payloads/.
    #   payload allowlist: disease_id, reason, image_sha_prefix
    "vision_shadow_inference",
    # vision_disabled_short_circuit — _detect was dispatched after the
    # feature was disabled (config flip mid-session, cached tool def).
    # Should be rare since as_tool() returns None when disabled — the
    # event tells the operator a stale tool def made it to the LLM.
    #   payload allowlist: reason, disease_id, image_sha_prefix
    "vision_disabled_short_circuit",
    # vision.specialist_keywords.missing — KTD-V1 audit-only signal. The
    # post_process hook scanned an urgent_specialist / soon_specialist
    # reply and found no phrase from configs/i18n/<lang>/vision.yaml
    # vision.specialist_keywords.<action>.
    #   payload allowlist: disease_id, clinical_action,
    #                      observed_leading_chars (count, not text)
    "vision.specialist_keywords.missing",
    # --- web layer (FastAPI app + JWT cookie auth + CSRF + chat SSE) -----
    #
    # All web.* kinds are emitted only when the web extra is loaded.
    # Web-side context (request_id / user_id / language) is populated by
    # WebContextMiddleware before any handler runs, so audit_event never
    # raises MissingContextError inside the web path.
    #
    # JWT lifecycle:
    #   web.jwt.invalid — silent downgrade (signature mismatch, expired,
    #     malformed). Normal lifecycle event.
    #     payload allowlist: reason
    #   web.jwt.tamper_suspected — fail-loud (valid HMAC + disallowed
    #     `lang` claim or other schema violation). Implies secret
    #     compromise or buggy issuer.
    #     payload allowlist: reason
    "web.jwt.invalid",
    "web.jwt.tamper_suspected",
    # Auth router:
    #   web.auth.login_success — successful credential verification +
    #     JWT issuance.
    #     payload allowlist: user_id
    #   web.auth.login_failed — unknown user OR wrong password (same
    #     response shape, but the audit distinguishes via ip_hmac). The
    #     ip_hmac field lets post-hoc clustering detect login-spray
    #     without storing raw IPs.
    #     payload allowlist: user_id_attempted, ip_hmac
    #   web.auth.logout — explicit logout (cookie clear).
    #     payload allowlist: user_id
    "web.auth.login_success",
    "web.auth.login_failed",
    "web.auth.logout",
    # CSRF middleware:
    #   web.csrf.blocked — mutation request without a matching
    #     X-CSRF-Token header (or missing csrf_token cookie).
    #     payload allowlist: path, method
    "web.csrf.blocked",
    # OpenAPI gating:
    #   web.openapi.access_blocked — /openapi.json or /docs requested in
    #     production by a non-admin (or unauthenticated). Returns 404 to
    #     avoid revealing existence.
    #     payload allowlist: path, reason
    "web.openapi.access_blocked",
    # /me router:
    #   web.me.language_changed — PATCH /api/v1/me {language} succeeded;
    #     JWT cookie reissued with new lang claim.
    #     payload allowlist: from, to
    #   web.me.provider_changed — PATCH /api/v1/me {provider_id} succeeded.
    #     The next stream call will resolve against the new id.
    #     payload allowlist: from, to
    #   web.me.unknown_provider — PATCH /api/v1/me {provider_id} requested
    #     an id absent from the catalog; rejected with 422 before the
    #     change touches disk.
    #     payload allowlist: requested
    "web.me.language_changed",
    "web.me.provider_changed",
    "web.me.unknown_provider",
    # Chat SSE router:
    #   web.chat.unknown_provider — user's Account.provider_id (or the
    #     default) is not in app.state.ask_services. Config error.
    #     payload allowlist: provider_id
    #   web.chat.q_invalid — body validation rejected `q` (empty or
    #     length cap exceeded). Surfaces malformed clients.
    #     payload allowlist: reason
    #   web.chat.session_busy — second concurrent stream against the
    #     same session_id; returns 409 immediately (interleaved tokens
    #     would be user-visible garbage).
    #     payload allowlist: session_id
    "web.chat.unknown_provider",
    "web.chat.q_invalid",
    "web.chat.session_busy",
    #   web.library.ingest — POST /api/v1/library/ingest finished a run;
    #     aggregate counts go here so the audit log carries a single line
    #     per upload bundle (per-part outcomes stay in the response only).
    #     payload allowlist: user_id, added_parts, skipped_parts,
    #     failed_parts, added_chunks
    "web.library.ingest",
    # CLI out-of-band password management:
    #   cli.user.password_set — `claritymed user set-password <user_id>`
    #     succeeded. Payload carries the user_id only; the plaintext is
    #     never logged or echoed.
    "cli.user.password_set",
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
