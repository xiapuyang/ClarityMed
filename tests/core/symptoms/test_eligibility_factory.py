"""Factory dispatch for eligibility strategies — fail-loud on missing branch."""

from __future__ import annotations

import pytest

from claritymed.core.rag.terms.base import (
    Alias,
    ConceptHit,
    ConceptLanguage,
    TermService,
)
from claritymed.core.rag.terms.umls_cmekg import NoOpTermService
from claritymed.core.symptoms.eligibility import (
    DirectEligibility,
    TermServiceEligibility,
    build_eligibility_strategy,
)
from claritymed.core.symptoms.schemas import (
    DirectEligibilityEntry,
    EligibilityCatalogConfig,
    TermServiceEligibilityEntry,
    TranslationEligibilityEntry,
)
from claritymed.errors import (
    EligibilityStrategyUnavailableError,
    UnknownEligibilityStrategyError,
)


class _StubTermService(TermService):
    def lookup(self, surface: str, language: ConceptLanguage) -> list[ConceptHit]:
        return []

    def cross_lingual_aliases(self, concept_id: str) -> list[Alias]:
        return []


def _direct_cfg() -> EligibilityCatalogConfig:
    return EligibilityCatalogConfig(
        active="direct",
        catalog=[DirectEligibilityEntry(id="direct", kind="direct")],
    )


def _term_service_cfg() -> EligibilityCatalogConfig:
    return EligibilityCatalogConfig(
        active="term_service",
        catalog=[TermServiceEligibilityEntry(id="term_service", kind="term_service")],
    )


def _translation_cfg() -> EligibilityCatalogConfig:
    return EligibilityCatalogConfig(
        active="translation",
        catalog=[
            TranslationEligibilityEntry(
                id="translation",
                kind="translation",
                provider_id="omlx",
                prompt_name="translate_complaint",
            )
        ],
    )


def test_direct_kind_returns_direct_eligibility() -> None:
    strategy = build_eligibility_strategy(_direct_cfg(), vocabs={})
    assert isinstance(strategy, DirectEligibility)


def test_direct_without_vocabs_still_constructs() -> None:
    """Vocabs are required at check time, but absence at construct must
    not raise — the plugin may wire them in lazily (per-dataset) later
    or run with an empty map in degraded mode (the strategy returns
    ``strategy_unavailable``)."""
    strategy = build_eligibility_strategy(_direct_cfg())
    assert isinstance(strategy, DirectEligibility)


def test_translation_kind_dispatches_with_overrides() -> None:
    """End-to-end translation wiring lives in
    ``test_eligibility_translation.py``; this case just confirms the
    factory does not raise when translation is the active kind."""
    from claritymed.core.schemas import ProviderConfig
    from claritymed.core.symptoms.eligibility import TranslationEligibility

    def _local(pid: str) -> ProviderConfig:
        return ProviderConfig(
            id=pid, kind="local", model="qwen3:14b", base_url="http://127.0.0.1:8000"
        )

    class _Registry:
        def get(self, name: str, version: str = "latest", language=None) -> str:
            return "system"

    class _Agent:
        async def run(self, prompt: str):
            class _R:
                output = "chest pain"

            return _R()

    strategy = build_eligibility_strategy(
        _translation_cfg(),
        vocabs={"ddxplus": {"E_1": frozenset({"chest pain"})}},
        translation_overrides={
            "agent_factory": lambda s, p: _Agent(),
            "provider_resolver": _local,
            "availability_check": lambda prov: True,
            "registry": _Registry(),
        },
    )
    assert isinstance(strategy, TranslationEligibility)


def test_term_service_kind_dispatches_with_factory() -> None:
    strategy = build_eligibility_strategy(
        _term_service_cfg(),
        sidecars={"ddxplus": {"E_1": "C_1"}},
        term_service_factory=_StubTermService,
    )
    assert isinstance(strategy, TermServiceEligibility)


def test_term_service_without_factory_raises_unknown_strategy() -> None:
    """Operator wiring bug — the active strategy needs a dependency the
    caller forgot to pass. Surfacing as UnknownEligibilityStrategyError
    keeps the failure mode aligned with the typo case."""
    with pytest.raises(UnknownEligibilityStrategyError) as info:
        build_eligibility_strategy(_term_service_cfg())
    assert "factory" in str(info.value).lower()


def test_term_service_with_noop_propagates_unavailable() -> None:
    """A misconfigured term_service (``term_service.active = "none"``)
    should not silently produce empty eligibility results — the
    construct-time check must fire."""
    with pytest.raises(EligibilityStrategyUnavailableError):
        build_eligibility_strategy(
            _term_service_cfg(),
            sidecars={"ddxplus": {"E_1": "C_1"}},
            term_service_factory=NoOpTermService,
        )
