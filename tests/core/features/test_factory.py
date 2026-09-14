"""``build_features`` registry tests."""

from __future__ import annotations

import pytest

from claritymed.core.features import build_features
from claritymed.core.rag.feature import RagFeature


def test_build_features_returns_rag_plugin_by_default():
    plugins = build_features(rag_mode="tool", rag_strategy=None)
    assert any(isinstance(p, RagFeature) and p.mode == "tool" for p in plugins)


def test_build_features_deterministic_mode_carries_through():
    plugins = build_features(rag_mode="deterministic", rag_strategy=None)
    rag = next(p for p in plugins if p.name == "rag")
    assert rag.mode == "deterministic"


def test_build_features_agentic_raises_not_implemented():
    """Agentic is reserved for v2 (state-graph workflow). Any active
    feature picking it should fail loud — no silent downgrade."""
    with pytest.raises(NotImplementedError) as exc:
        build_features(rag_mode="agentic", rag_strategy=None)
    assert "agentic" in str(exc.value).lower()


def test_build_features_symptoms_factory_included_when_provided():
    """When a ``symptoms_factory`` is passed, the returned plugin lands
    in the active list. Absence (the default) keeps the registry untouched
    — the symptoms feature is opt-in per-deployment via
    ``configs/symptoms.yaml`` + AskService wiring."""

    class _StubSymptoms:
        name = "symptoms"
        mode = "tool"

    plugins = build_features(
        rag_mode="tool", rag_strategy=None, symptoms_factory=_StubSymptoms
    )
    assert any(p.name == "symptoms" for p in plugins)


def test_build_features_symptoms_absent_by_default():
    """No ``symptoms_factory`` → no symptoms plugin in the active list.
    Clean degradation when ``configs/symptoms.yaml`` is missing."""
    plugins = build_features(rag_mode="tool", rag_strategy=None)
    assert all(p.name != "symptoms" for p in plugins)
