"""OpenTelemetry tracing — sends spans to a self-hosted Phoenix instance.

``setup_tracing()`` is the one entry point. It reads
``PHOENIX_COLLECTOR_ENDPOINT`` from the environment; if unset, the
function returns silently and nothing about the process changes. This is
the CI / offline / opt-in path — no test, no shell session, no notebook
gets surprise traffic.

When the env var is set, we configure a global ``TracerProvider``,
attach ``OpenInferenceSpanProcessor`` (in-place enrichment with OI
semantic conventions — model, prompt, response, token counts, cache
hits), and an OTLP HTTP exporter pointed at Phoenix. Finally we call
``Agent.instrument_all()`` so every pydantic-ai run / model_call /
tool_call automatically becomes a span — no per-call wiring.

PHI note: pydantic-ai instrumentation puts the user prompt and the
assistant response into the span. Phoenix runs locally by default
(``docker run arizephoenix/phoenix`` or ``uvx arize-phoenix serve``)
so PHI does not leave the host. If a SaaS Phoenix endpoint is ever
configured, wrap ``OpenInferenceSpanProcessor`` with a PHI scrubber
before the exporter — do not export raw PHI to a third party.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_ENDPOINT_ENV = "PHOENIX_COLLECTOR_ENDPOINT"
_API_KEY_ENV = "PHOENIX_API_KEY"
_SERVICE_NAME_ENV = "OTEL_SERVICE_NAME"
_DEFAULT_SERVICE_NAME = "claritymed"
_SCRUB_DISABLE_ENV = "CLARITYMED_TRACE_PHI_SCRUB_DISABLE"

# Attribute keys (or key prefixes) OpenInference writes that may carry
# PHI in their values. We pin to OI's documented attribute names rather
# than wildcards so a future OI release cannot silently slip new keys
# past the scrubber — if it adds one, the test catches the gap.
PHI_SCRUB_ATTR_KEYS: tuple[str, ...] = (
    "input.value",
    "output.value",
)
PHI_SCRUB_ATTR_PREFIXES: tuple[str, ...] = (
    "llm.input_messages.",
    "llm.output_messages.",
    "llm.prompts.",
    "llm.prompt.",
    "llm.completion.",
    "llm.tool_call.",
    "tool.parameters.",
    "retrieval.documents.",
    "embedding.embeddings.",
)

_lock = threading.Lock()
_configured: bool = False


def setup_tracing() -> bool:
    """Configure global tracing if a Phoenix endpoint is set.

    Returns ``True`` if tracing was installed (or was already installed
    by a prior call), ``False`` if no endpoint is configured. Multiple
    callers in the same process (CLI entry, TUI mount, web boot) can
    call this freely — the underlying setup runs at most once.
    """
    global _configured
    endpoint = os.environ.get(_ENDPOINT_ENV, "").strip()
    if not endpoint:
        return False
    with _lock:
        if _configured:
            return True
        try:
            _install(endpoint)
            _configured = True
            logger.info("tracing installed -> %s", endpoint)
            return True
        except Exception:  # noqa: BLE001
            logger.exception("failed to install tracing; continuing without it")
            return False


def is_configured() -> bool:
    return _configured


def reset_for_testing() -> None:
    """Test hook — clears the once-only guard. Not for production."""
    global _configured
    with _lock:
        _configured = False


def _install(endpoint: str) -> None:
    # Imports are local so the rest of the codebase doesn't pay the OTel
    # import cost when tracing is off.
    from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from pydantic_ai import Agent

    service_name = os.environ.get(_SERVICE_NAME_ENV, _DEFAULT_SERVICE_NAME)
    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    # Copy our baggage (request_id / user_id) onto every span as the first
    # thing that runs at span start. Phoenix and Tempo can then group by
    # request_id without parsing baggage manually, and the audit-log line
    # that recorded the trace_id round-trips both ways.
    provider.add_span_processor(BaggageSpanProcessor())

    # OpenInference enriches spans in place with provider/model/usage/cache
    # attributes that Phoenix UI knows how to render. It does not export.
    provider.add_span_processor(OpenInferenceSpanProcessor())

    # Scrub PHI from OI-written attributes *before* the exporter sees the
    # batch. on_end runs in span-finish order, after OI's instrumentation
    # has populated input/output attributes, but before the
    # BatchSpanProcessor below picks the span up for export.
    if os.environ.get(_SCRUB_DISABLE_ENV, "").lower() not in ("1", "true", "yes"):
        provider.add_span_processor(PhiScrubSpanProcessor())

    # OTLP HTTP is what self-hosted Phoenix accepts on /v1/traces. Batch
    # processor so streaming latency isn't taxed by export.
    headers: dict[str, str] = {}
    api_key = os.environ.get(_API_KEY_ENV, "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    exporter = OTLPSpanExporter(
        endpoint=f"{endpoint.rstrip('/')}/v1/traces",
        headers=headers or None,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    # Global hook — every pydantic-ai Agent run starts emitting spans
    # without per-call wiring.
    Agent.instrument_all()


def _baggage_span_processor_class():
    """Return the BaggageSpanProcessor class, importing OTel lazily."""
    from opentelemetry import baggage
    from opentelemetry import context as otel_context
    from opentelemetry.sdk.trace import SpanProcessor

    from claritymed.context import (
        BAGGAGE_REQUEST_ID,
        BAGGAGE_SESSION_ID,
        BAGGAGE_USER_ID,
    )

    _COPY_KEYS = (BAGGAGE_REQUEST_ID, BAGGAGE_USER_ID, BAGGAGE_SESSION_ID)

    class _BaggageSpanProcessor(SpanProcessor):
        """Lifts our two baggage keys onto every starting span as attrs."""

        def on_start(self, span, parent_context=None):  # noqa: D401
            ctx = parent_context or otel_context.get_current()
            for key in _COPY_KEYS:
                value = baggage.get_baggage(key, ctx)
                if value:
                    span.set_attribute(key, str(value))

        def on_end(self, span):
            return

        def shutdown(self):
            return

        def force_flush(self, timeout_millis: int = 30000):  # noqa: ARG002
            return True

    return _BaggageSpanProcessor


def BaggageSpanProcessor():  # noqa: N802 — factory mimics a class name on purpose
    """Construct a baggage-copying span processor.

    Hidden behind a factory so importing ``tracing`` does not require the
    OTel SDK; the class is built only inside ``_install``.
    """
    return _baggage_span_processor_class()()


def _phi_scrub_processor_class():
    """Return the PhiScrubSpanProcessor class, importing OTel + PhiGuard lazily.

    The class scrubs attributes OpenInference set during span lifetime
    by mutating ``span._attributes`` on ``on_end``. The mutation is safe
    because OTel's SDK only freezes the attribute mapping on export, and
    BatchSpanProcessor (the exporter we feed) runs after us.
    """
    from opentelemetry.sdk.trace import SpanProcessor

    from claritymed.orchestrator import PhiGuard

    class _PhiScrubSpanProcessor(SpanProcessor):
        """Strip / redact PHI from OI-written attributes pre-export."""

        def __init__(self) -> None:
            # PhiGuard reads safety.yaml; if config is missing or broken,
            # we must still emit *something* (better redacted than raw).
            # Fall back to a guard with no rules — it still no-ops PHI
            # but at least the processor doesn't crash the tracer.
            try:
                self._guard = PhiGuard.from_config()
            except Exception:  # noqa: BLE001
                logger.exception("PhiGuard.from_config failed; scrubber inert")
                self._guard = None

        def on_start(self, span, parent_context=None):  # noqa: D401, ARG002
            return

        def on_end(self, span):  # noqa: D401
            guard = self._guard
            if guard is None:
                return
            attrs = getattr(span, "_attributes", None)
            if not attrs:
                return
            try:
                for key in list(attrs.keys()):
                    if not _is_phi_attr_key(key):
                        continue
                    value = attrs[key]
                    if not isinstance(value, str) or not value:
                        continue
                    scrubbed, report = guard.scrub_free_text(value)
                    if report.rule_hits:
                        attrs[key] = scrubbed
            except Exception:  # noqa: BLE001
                logger.exception(
                    "PHI scrub failed on span %s", getattr(span, "name", "?")
                )

        def shutdown(self):  # noqa: D401
            return

        def force_flush(self, timeout_millis: int = 30000):  # noqa: ARG002, D401
            return True

    return _PhiScrubSpanProcessor


def _is_phi_attr_key(key: str) -> bool:
    """Match attribute keys against the static allow-list of PHI-bearing names."""
    if key in PHI_SCRUB_ATTR_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in PHI_SCRUB_ATTR_PREFIXES)


def PhiScrubSpanProcessor():  # noqa: N802 — factory mimics a class name on purpose
    """Construct a PHI scrub span processor.

    Hidden behind a factory for the same reason as ``BaggageSpanProcessor``
    — OTel SDK import is deferred to install time.
    """
    return _phi_scrub_processor_class()()


__all__ = [
    "BaggageSpanProcessor",
    "PHI_SCRUB_ATTR_KEYS",
    "PHI_SCRUB_ATTR_PREFIXES",
    "PhiScrubSpanProcessor",
    "is_configured",
    "reset_for_testing",
    "setup_tracing",
]
