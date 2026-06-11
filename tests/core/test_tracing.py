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
