"""Audit lines pick up the active OTel trace id; spans pick up our baggage.

Two-way correlation: a trace_id in audit.log lets an operator pivot to
Phoenix; a request_id attribute on every span lets a Phoenix trace pivot
back to grep. Tests use an in-memory OTel SDK so they don't depend on a
Phoenix collector being reachable.
"""

from __future__ import annotations

import json

import pytest

from claritymed.context import (
    BAGGAGE_REQUEST_ID,
    BAGGAGE_SESSION_ID,
    BAGGAGE_USER_ID,
    apply_context,
    attach_session_baggage,
    detach_session_baggage,
    new_request_id,
    reset_context,
)
from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.tracing import (
    BaggageSpanProcessor,
    reset_for_testing,
)


@pytest.fixture
def in_memory_tracer():
    """Build a local TracerProvider with an in-memory exporter and return
    ``(provider, exporter)``. We do **not** call ``set_tracer_provider``
    (OTel forbids overriding it more than once per process, which would
    couple every test in the session). Tests grab a tracer directly off
    the returned provider; OTel's context propagation still works
    because span activation lives in ContextVars, not in the provider.
    """
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "test"}))
    provider.add_span_processor(BaggageSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reset_for_testing()
    try:
        yield provider, exporter
    finally:
        provider.shutdown()
        reset_for_testing()


def test_audit_event_has_null_trace_id_when_no_span_is_active(caplog):
    """Tracing-off path: trace_id present in the JSON but ``null`` — schema
    stays the same so downstream parsers do not have to special-case it."""
    rid = new_request_id()
    tokens = apply_context(rid, "test", "en")
    try:
        with caplog.at_level("INFO", logger="claritymed.audit"):
            event = audit_event("mode.ask", payload={"answer_len": 12})
    finally:
        reset_context(tokens)

    assert event.trace_id is None
    assert event.span_id is None
    assert '"trace_id":null' in event.model_dump_json()


def test_audit_event_picks_up_active_span_trace_id(in_memory_tracer, caplog):
    provider, _ = in_memory_tracer
    rid = new_request_id()
    tokens = apply_context(rid, "test", "en")
    try:
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("mode.ask") as span:
            ctx = span.get_span_context()
            expected_trace = format(ctx.trace_id, "032x")
            expected_span = format(ctx.span_id, "016x")
            event = audit_event("mode.ask", payload={"answer_len": 4})
    finally:
        reset_context(tokens)

    assert event.trace_id == expected_trace
    assert event.span_id == expected_span
    decoded = json.loads(event.model_dump_json())
    assert decoded["request_id"] == rid
    assert decoded["trace_id"] == expected_trace


def test_baggage_processor_copies_request_id_and_user_id_onto_span(in_memory_tracer):
    provider, exporter = in_memory_tracer
    rid = new_request_id()
    tokens = apply_context(rid, "test", "en")
    try:
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("custom-work"):
            pass
    finally:
        reset_context(tokens)

    spans = exporter.get_finished_spans()
    assert spans, "span did not finish"
    last = spans[-1]
    assert last.attributes.get(BAGGAGE_REQUEST_ID) == rid
    assert last.attributes.get(BAGGAGE_USER_ID) == "test"


def test_session_id_baggage_appears_on_span_when_attached(in_memory_tracer):
    """``AskService`` attaches ``claritymed.session_id`` for the duration
    of an LLM call; spans inside that scope must surface the id so
    Phoenix can group a trace by conversation."""
    provider, exporter = in_memory_tracer
    rid = new_request_id()
    outer = apply_context(rid, "test", "en")
    sid = "session-abc-123"
    inner = attach_session_baggage(sid)
    try:
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("llm.call"):
            pass
    finally:
        detach_session_baggage(inner)
        reset_context(outer)

    spans = exporter.get_finished_spans()
    last = spans[-1]
    assert last.attributes.get(BAGGAGE_SESSION_ID) == sid
    # The request + user baggage still rides along on the same span.
    assert last.attributes.get(BAGGAGE_REQUEST_ID) == rid
    assert last.attributes.get(BAGGAGE_USER_ID) == "test"


def test_session_baggage_detach_does_not_leak_to_next_run(in_memory_tracer):
    provider, exporter = in_memory_tracer
    rid = new_request_id()
    outer = apply_context(rid, "test", "en")
    inner = attach_session_baggage("session-A")
    detach_session_baggage(inner)
    try:
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("after-detach"):
            pass
    finally:
        reset_context(outer)

    last = exporter.get_finished_spans()[-1]
    assert BAGGAGE_SESSION_ID not in last.attributes
    # request_id / user_id baggage from the outer apply_context still wins.
    assert last.attributes.get(BAGGAGE_REQUEST_ID) == rid


def test_detach_session_baggage_accepts_none_silently():
    """When OTel is unavailable ``attach_session_baggage`` returns None;
    callers must be able to blindly hand that back without a guard."""
    detach_session_baggage(None)


def test_baggage_is_detached_after_reset(in_memory_tracer):
    """Outside the with-block, new spans must NOT carry the prior context's
    baggage — otherwise multi-request servers leak user identity."""
    provider, exporter = in_memory_tracer
    rid_a = new_request_id()
    tokens = apply_context(rid_a, "test", "en")
    reset_context(tokens)

    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("after-reset"):
        pass

    spans = exporter.get_finished_spans()
    last = spans[-1]
    assert BAGGAGE_REQUEST_ID not in last.attributes
    assert BAGGAGE_USER_ID not in last.attributes


def test_apply_context_returns_four_tuple_even_without_otel():
    """Tuple shape is stable so callers can rely on `len(tokens) == 4`."""
    tokens = apply_context(new_request_id(), "test", "en")
    try:
        assert len(tokens) == 4
    finally:
        reset_context(tokens)


def test_reset_context_accepts_legacy_three_tuple():
    """Older callers still pass 3-tuples (pre-tracing era). They must not
    blow up when handed back to reset_context."""
    rid_token = None
    uid_token = None
    lang_token = None
    from claritymed.context import language_ctx, request_id_ctx, user_id_ctx

    rid_token = request_id_ctx.set("20260606222522DEADBEEF")
    uid_token = user_id_ctx.set("test")
    lang_token = language_ctx.set("en")
    reset_context((rid_token, uid_token, lang_token))  # no exception, no OTel touch
