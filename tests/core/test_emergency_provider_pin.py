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

import pytest

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


# --- Fix #8: startup fail-loud cross-catalog check -------------------


def _pin_for_validated(
    monkeypatch, *, provider_id: str | None, catalog: ModelsConfig
) -> None:
    """Stub load_emergency_config, load_rules, and load_models for
    load_validated_emergency_config tests.

    load_emergency_config is imported locally inside the function body,
    so we patch it at the module where it originates (claritymed.core.emergency.config)
    AND at the stores.models module for the load_models call inside _check_provider_in_catalog.
    """
    import claritymed.core.emergency.config as cfg_mod
    import claritymed.core.emergency.rules as rules_mod
    import claritymed.stores.models as models_mod

    ec = EmergencyConfig(provider_id=provider_id)

    monkeypatch.setattr(
        cfg_mod,
        "load_emergency_config",
        lambda *_a, **_kw: ec,
    )
    monkeypatch.setattr(models_mod, "load_models", lambda: catalog)
    # load_rules falls back to empty list when the file path doesn't exist —
    # stub it to return a minimal valid rule so enforce_floors has something to work with.
    from claritymed.core.emergency.rules import Rule, RuleTriggers

    minimal_rule = Rule(
        id="test_rule",
        triggers=RuleTriggers(any_of=["throat_tightness"], min_qualifier_matches=1),
        level="critical",
        action_key="emergency.action.epi_then_ems",
        citations=["WAO 2020"],
    )
    monkeypatch.setattr(rules_mod, "load_rules", lambda *_a, **_kw: [minimal_rule])


def test_load_validated_raises_when_provider_id_not_in_catalog(monkeypatch):
    """emergency.yaml::provider_id pointing at a nonexistent id raises at startup."""
    from claritymed.core.emergency.rules import load_validated_emergency_config

    _pin_for_validated(
        monkeypatch, provider_id="nonexistent_id", catalog=_two_local_catalog()
    )
    with pytest.raises(ValueError, match="nonexistent_id"):
        load_validated_emergency_config()


def test_load_validated_raises_when_provider_id_is_cloud(monkeypatch):
    """emergency.yaml::provider_id resolving to kind=cloud raises at startup (KTD-E1)."""
    import pytest

    from claritymed.core.emergency.rules import load_validated_emergency_config
    from claritymed.core.schemas import ProviderConfig

    cloud_catalog = ModelsConfig(
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
    _pin_for_validated(monkeypatch, provider_id="claude", catalog=cloud_catalog)
    with pytest.raises(ValueError, match="kind='cloud'"):
        load_validated_emergency_config()


def test_load_validated_passes_with_local_provider_id(monkeypatch):
    """emergency.yaml::provider_id pointing at a valid kind=local entry passes."""
    from claritymed.core.emergency.rules import load_validated_emergency_config

    _pin_for_validated(monkeypatch, provider_id="omlx", catalog=_two_local_catalog())
    cfg, rules = load_validated_emergency_config()
    assert cfg.provider_id == "omlx"
    assert len(rules) >= 1


def test_load_validated_passes_with_no_provider_id(monkeypatch):
    """Unset provider_id (None) skips the catalog check entirely."""
    from claritymed.core.emergency.rules import load_validated_emergency_config

    _pin_for_validated(monkeypatch, provider_id=None, catalog=_two_local_catalog())
    cfg, rules = load_validated_emergency_config()
    assert cfg.provider_id is None
