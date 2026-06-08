"""Tracing module — opt-in via PHOENIX_COLLECTOR_ENDPOINT.

Verifies the silent-when-unset behaviour (so CI / offline / headless tests
never accidentally spin up OTel) and the install-once invariant when the
env var is set.
"""

from __future__ import annotations

import pytest

from claritymed.core.observability.tracing import (
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
