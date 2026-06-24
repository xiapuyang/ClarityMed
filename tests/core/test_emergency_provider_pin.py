"""Tests for ``EmergencyConfig.provider_id`` — the gate's explicit pin.

The pin exists because "first kind=local in models.yaml" is brittle:
on a host running both ollama and omlx, YAML order alone decides which
the gate binds to. If the chosen one happens to be down (server
crashed, port changed, service deferred), the gate falls open every
turn — silently, because ``EmergencyTriage.assess`` swallows the
connection error to keep the agent loop alive. The pin lets the
operator say "bind to omlx no matter what order things end up in".

These tests pin the contract of ``build_local_gate_model(prefer_id=…)``
and the helper factories that read it from ``EmergencyConfig``.
"""

from __future__ import annotations

import claritymed.core.emergency.config as _emcfg_mod
import claritymed.stores.models as _models_mod
from claritymed.core.emergency import (
    build_default_composer,
    build_default_critical_reply,
    build_default_extractor,
)
from claritymed.core.emergency._provider import build_local_gate_model
from claritymed.core.emergency.config import EmergencyConfig
from claritymed.core.schemas import ModelsConfig, ProviderConfig


def _two_local_catalog() -> ModelsConfig:
    """Catalog with two local providers — exercises pin precedence over order."""
    return ModelsConfig(
        providers=[
            # ollama first in YAML order — without a pin, the helper
            # would bind here even though our pin (below) targets omlx.
            ProviderConfig(
                id="ollama",
                kind="local",
                model="qwen3:14b",
                base_url="http://127.0.0.1:11434/v1",
            ),
            ProviderConfig(
                id="omlx",
                kind="local",
                model="test-model",
                base_url="http://127.0.0.1:8000/v1",
            ),
        ],
        default_provider="ollama",
    )


def _pin(monkeypatch, *, provider_id: str | None, catalog: ModelsConfig) -> None:
    """Helper: stub both ``load_emergency_config`` and ``load_models``."""
    monkeypatch.setattr(_models_mod, "load_models", lambda: catalog)
    monkeypatch.setattr(
        _emcfg_mod,
        "load_emergency_config",
        lambda: EmergencyConfig(provider_id=provider_id),
    )


# --- happy path: pin is honored over YAML order -----------------------


def test_pin_honored_when_target_exists_and_local(monkeypatch):
    """provider_id pinned to a real kind=local entry → that entry binds.

    The catalog lists ollama first; without the pin the helper would
    return an ollama-backed model. With the pin pointing at omlx, the
    helper should bind to omlx. We assert via the model object's
    ``base_url`` rather than its id (pydantic-ai's Model surface).
    """
    _pin(monkeypatch, provider_id="omlx", catalog=_two_local_catalog())
    model = build_local_gate_model("extractor", prefer_id="omlx")
    assert model is not None
    # OpenAIModel exposes the bound endpoint via ``.client.base_url``.
    # If the pin were ignored we'd see :11434 (ollama), not :8000 (omlx).
    assert "8000" in str(model.client.base_url)


def test_factory_helpers_propagate_pin_to_provider(monkeypatch):
    """build_default_extractor / composer / critical_reply honor cfg pin.

    Same catalog (ollama first, omlx second) + pin to omlx → all three
    helpers must end up bound to omlx, not the first-listed ollama.
    """
    _pin(monkeypatch, provider_id="omlx", catalog=_two_local_catalog())
    extractor = build_default_extractor()
    composer = build_default_composer()
    critical = build_default_critical_reply()
    assert extractor is not None
    assert composer is not None
    assert critical is not None
    for component in (extractor._model, composer._model, critical._model):
        assert "8000" in str(component.client.base_url), (
            f"helper bound to {component.client.base_url!r} but pin was omlx"
        )


# --- pin rejection: typo / unknown id ---------------------------------


def test_pin_returns_none_when_id_missing_from_catalog(monkeypatch):
    """provider_id pointing at a non-existent catalog entry → None.

    Critical: even though a usable local provider (ollama) IS in the
    catalog, the pin must NOT fall back to it — a silent swap would
    defeat the operator's explicit choice and route requests to a
    provider they did not authorise. None lets the gate stay in
    routine_noop mode (visible failure) instead.
    """
    _pin(monkeypatch, provider_id="ghost_provider", catalog=_two_local_catalog())
    assert build_local_gate_model("extractor", prefer_id="ghost_provider") is None
    # And via the public factory:
    assert build_default_extractor() is None


# --- pin rejection: cloud provider (KTD-E1 PHI floor) -----------------


def test_pin_rejects_cloud_provider(monkeypatch):
    """provider_id pointing at a kind=cloud entry → None (KTD-E1).

    The gate's extractor sees raw patient text; KTD-E1 mandates it
    runs only on a local provider. Even if an operator typo-pins to
    a cloud entry, the helper must refuse rather than expose PHI.
    """
    catalog = ModelsConfig(
        providers=[
            ProviderConfig(
                id="claude",
                kind="cloud",
                model="anthropic:claude-sonnet-4-6",
            ),
            ProviderConfig(
                id="ollama",
                kind="local",
                model="qwen3:14b",
                base_url="http://127.0.0.1:11434/v1",
            ),
        ],
        default_provider="claude",
    )
    _pin(monkeypatch, provider_id="claude", catalog=catalog)
    assert build_local_gate_model("extractor", prefer_id="claude") is None
    assert build_default_extractor() is None


# --- no pin → first-local fallback path preserved ---------------------


def test_no_pin_falls_back_to_first_local(monkeypatch):
    """provider_id unset → today's "first kind=local" behavior preserved.

    Ensures the new pin field is purely additive: deployments that
    haven't set it continue to work exactly as before.
    """
    _pin(monkeypatch, provider_id=None, catalog=_two_local_catalog())
    model = build_local_gate_model("extractor")
    assert model is not None
    # First local in YAML order is ollama (:11434).
    assert "11434" in str(model.client.base_url)
