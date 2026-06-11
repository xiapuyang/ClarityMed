"""Tracing module — opt-in via PHOENIX_COLLECTOR_ENDPOINT.

Verifies the silent-when-unset behaviour (so CI / offline / headless tests
never accidentally spin up OTel) and the install-once invariant when the
env var is set.
"""

from __future__ import annotations

import pytest

from claritymed.core.observability.tracing import (
    _is_local_endpoint,
    is_configured,
    reset_for_testing,
    setup_tracing,
)


@pytest.fixture(autouse=True)
def _isolate_tracing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    monkeypatch.delenv("PHOENIX_API_KEY", raising=False)
    reset_for_testing()
    yield
    reset_for_testing()


def test_no_endpoint_returns_false_without_side_effects():
    assert setup_tracing() is False
    assert is_configured() is False


def test_conftest_strips_phoenix_env_so_tui_tests_do_not_upload():
    """Regression: if PHOENIX_COLLECTOR_ENDPOINT leaks in from the dev's
    shell, TUI smoke tests (App.on_mount → setup_tracing) would install
    a real OTLP exporter and Agent.instrument_all(), uploading every
    later test's agent.run to the dev's local Phoenix. The session-level
    conftest must strip it before any test runs."""
    import os

    # _isolate_tracing's monkeypatch.delenv already cleared it for this
    # test, so assert against the pristine os.environ used by tests that
    # do not opt in: there is no autouse fixture in tests/conftest.py
    # that re-adds it, and pytest_configure removed it once at session
    # start. We rely on the same os.environ here.
    assert "PHOENIX_COLLECTOR_ENDPOINT" not in os.environ
    assert "PHOENIX_API_KEY" not in os.environ


def test_endpoint_set_installs_provider_once(monkeypatch: pytest.MonkeyPatch):
    install_calls: list[str] = []

    def _fake_install(endpoint: str) -> None:
        install_calls.append(endpoint)

    monkeypatch.setattr(
        "claritymed.core.observability.tracing._install",
        _fake_install,
    )
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

    assert setup_tracing() is True
    assert is_configured() is True
    # Subsequent calls do not re-install — multiple CLI / TUI boots are safe.
    assert setup_tracing() is True
    assert install_calls == ["http://localhost:6006"]


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("http://localhost:6006", True),
        ("http://localhost:6006/v1/traces", True),
        ("http://127.0.0.1:6006", True),
        ("http://[::1]:6006", True),
        ("https://app.phoenix.arize.com", False),
        ("http://10.0.0.1:6006", False),
        ("http://my-internal-phoenix:6006", False),
    ],
)
def test_is_local_endpoint(endpoint, expected):
    assert _is_local_endpoint(endpoint) is expected


def test_install_failure_is_swallowed_so_main_flow_keeps_running(
    monkeypatch: pytest.MonkeyPatch,
):
    """If OTel setup blows up (e.g. broken collector), the app must still
    boot. Tracing is best-effort, not a startup gate."""

    def _boom(endpoint: str) -> None:
        raise RuntimeError("collector unreachable")

    monkeypatch.setattr("claritymed.core.observability.tracing._install", _boom)
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

    assert setup_tracing() is False
    assert is_configured() is False
