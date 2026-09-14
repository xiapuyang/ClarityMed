"""Tracing module — opt-in via ``tracing.enabled`` in configs/app.yaml."""

from __future__ import annotations

import pytest

from claritymed.core.observability.tracing import (
    TracingConfig,
    is_configured,
    reset_for_testing,
    setup_tracing,
)

_DISABLED = TracingConfig(enabled=False)
_LOCAL = TracingConfig(enabled=True, endpoint="http://localhost:6006")


@pytest.fixture(autouse=True)
def _isolate_tracing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "claritymed.core.observability.tracing._load_config",
        lambda: _DISABLED,
    )
    reset_for_testing()
    yield
    reset_for_testing()


def test_disabled_returns_false_without_side_effects():
    assert setup_tracing() is False
    assert is_configured() is False


def test_enabled_installs_provider_once(monkeypatch: pytest.MonkeyPatch):
    install_calls: list[TracingConfig] = []

    def _fake_install(cfg: TracingConfig) -> None:
        install_calls.append(cfg)

    monkeypatch.setattr(
        "claritymed.core.observability.tracing._load_config",
        lambda: _LOCAL,
    )
    monkeypatch.setattr(
        "claritymed.core.observability.tracing._install",
        _fake_install,
    )

    assert setup_tracing() is True
    assert is_configured() is True
    # Subsequent calls do not re-install — multiple CLI / TUI boots are safe.
    assert setup_tracing() is True
    assert len(install_calls) == 1
    assert install_calls[0].endpoint == "http://localhost:6006"


def test_install_failure_is_swallowed_so_main_flow_keeps_running(
    monkeypatch: pytest.MonkeyPatch,
):
    """If OTel setup blows up (e.g. broken collector), the app must still
    boot. Tracing is best-effort, not a startup gate."""

    def _boom(cfg: TracingConfig) -> None:
        raise RuntimeError("collector unreachable")

    monkeypatch.setattr(
        "claritymed.core.observability.tracing._load_config",
        lambda: _LOCAL,
    )
    monkeypatch.setattr("claritymed.core.observability.tracing._install", _boom)

    assert setup_tracing() is False
    assert is_configured() is False


# ---------------------------------------------------------------------------
# PhiScrub / Baggage span processors
# ---------------------------------------------------------------------------


def test_is_phi_attr_key_matches_allowlist():
    from claritymed.core.observability.tracing import (
        PHI_SCRUB_ATTR_KEYS,
        _is_phi_attr_key,
    )

    # Pick any item from the allowlist.
    sample_key = next(iter(PHI_SCRUB_ATTR_KEYS))
    assert _is_phi_attr_key(sample_key) is True


def test_is_phi_attr_key_matches_prefix():
    from claritymed.core.observability.tracing import (
        PHI_SCRUB_ATTR_PREFIXES,
        _is_phi_attr_key,
    )

    sample_prefix = next(iter(PHI_SCRUB_ATTR_PREFIXES))
    assert _is_phi_attr_key(f"{sample_prefix}.tail") is True


def test_is_phi_attr_key_rejects_unrelated_key():
    from claritymed.core.observability.tracing import _is_phi_attr_key

    assert _is_phi_attr_key("net.peer.port") is False


def test_phi_scrub_processor_on_end_redacts_phi_attrs():
    from claritymed.core.observability.tracing import (
        PHI_SCRUB_ATTR_KEYS,
        _phi_scrub_processor_class,
    )

    cls = _phi_scrub_processor_class()

    class _StubGate:
        def scrub(self, text):
            return "[SCRUBBED]"

    proc = cls(gate=_StubGate())
    sample_key = next(iter(PHI_SCRUB_ATTR_KEYS))

    class _Span:
        name = "test-span"

        def __init__(self):
            self._attributes = {sample_key: "secret-value", "safe.key": "fine"}

    span = _Span()
    proc.on_end(span)
    assert span._attributes[sample_key] == "[SCRUBBED]"
    # Non-PHI key is left alone.
    assert span._attributes["safe.key"] == "fine"


def test_phi_scrub_processor_with_no_gate_is_noop():
    from claritymed.core.observability.tracing import _phi_scrub_processor_class

    cls = _phi_scrub_processor_class()
    proc = cls(gate=None)

    class _Span:
        name = "x"

        _attributes = {"k": "v"}

    span = _Span()
    proc.on_end(span)
    assert span._attributes == {"k": "v"}


def test_phi_scrub_processor_handles_missing_attributes():
    from claritymed.core.observability.tracing import _phi_scrub_processor_class

    cls = _phi_scrub_processor_class()

    class _StubGate:
        def scrub(self, text):
            return "[X]"

    proc = cls(gate=_StubGate())

    class _Span:
        name = "x"
        # No _attributes attribute at all.

    proc.on_end(_Span())  # must not raise


def test_phi_scrub_processor_force_flush_and_shutdown():
    from claritymed.core.observability.tracing import _phi_scrub_processor_class

    cls = _phi_scrub_processor_class()
    proc = cls(gate=None)
    assert proc.force_flush(1000) is True
    assert proc.shutdown() is None


def test_phi_scrub_processor_swallows_gate_exception(caplog):
    """If the gate raises during scrub, the processor must log + continue."""
    from claritymed.core.observability.tracing import (
        PHI_SCRUB_ATTR_KEYS,
        _phi_scrub_processor_class,
    )

    cls = _phi_scrub_processor_class()

    class _BrokenGate:
        def scrub(self, text):
            raise RuntimeError("gate down")

    proc = cls(gate=_BrokenGate())
    sample_key = next(iter(PHI_SCRUB_ATTR_KEYS))

    class _Span:
        name = "x"
        _attributes = {sample_key: "secret"}

    with caplog.at_level("ERROR"):
        proc.on_end(_Span())  # must NOT raise
    assert any("PHI scrub failed" in r.message for r in caplog.records)


def test_baggage_span_processor_copies_baggage_keys_onto_span():
    from opentelemetry import baggage as _baggage
    from opentelemetry import context as otel_context

    from claritymed.context import BAGGAGE_REQUEST_ID, BAGGAGE_USER_ID
    from claritymed.core.observability.tracing import _baggage_span_processor_class

    cls = _baggage_span_processor_class()
    proc = cls()
    captured: dict[str, str] = {}

    class _Span:
        def set_attribute(self, key, value):
            captured[key] = value

    ctx = otel_context.get_current()
    ctx = _baggage.set_baggage(BAGGAGE_REQUEST_ID, "req-1", context=ctx)
    ctx = _baggage.set_baggage(BAGGAGE_USER_ID, "alice", context=ctx)
    proc.on_start(_Span(), parent_context=ctx)
    assert captured.get(BAGGAGE_REQUEST_ID) == "req-1"
    assert captured.get(BAGGAGE_USER_ID) == "alice"


def test_baggage_span_processor_no_baggage_skips():
    from claritymed.core.observability.tracing import _baggage_span_processor_class

    cls = _baggage_span_processor_class()
    proc = cls()
    captured: dict[str, str] = {}

    class _Span:
        def set_attribute(self, key, value):
            captured[key] = value

    proc.on_start(_Span(), parent_context=None)
    # No baggage in default context → nothing copied.
    assert captured == {}


def test_baggage_span_processor_lifecycle_noops():
    from claritymed.core.observability.tracing import _baggage_span_processor_class

    cls = _baggage_span_processor_class()
    proc = cls()
    assert proc.on_end(object()) is None
    assert proc.shutdown() is None
    assert proc.force_flush(1000) is True


def test_phi_scrub_processor_on_start_is_noop():
    from claritymed.core.observability.tracing import _phi_scrub_processor_class

    cls = _phi_scrub_processor_class()
    proc = cls(gate=None)
    assert proc.on_start(object()) is None


def test_baggage_span_processor_factory_returns_instance():
    from claritymed.core.observability.tracing import BaggageSpanProcessor

    proc = BaggageSpanProcessor()
    # Has the protocol methods we exercise.
    assert hasattr(proc, "on_start")
    assert hasattr(proc, "on_end")
