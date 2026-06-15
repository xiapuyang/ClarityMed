"""Resolve the active :class:`EligibilityStrategy` from config.

Mirrors :func:`claritymed.core.rag.terms.factory.build_term_service`: the
catalog + active-id pattern keeps the dispatch loud — a typo'd ``active``
already failed at config-load (handled by
:class:`~claritymed.core.symptoms.schemas.EligibilityCatalogConfig._resolve_active`),
and a catalog kind with no factory branch raises
:class:`~claritymed.errors.UnknownEligibilityStrategyError` here.

Strategy-specific dependencies (vocab maps for ``direct``, term-service
factories for ``term_service``, provider-resolvers for ``translation``)
are passed as keyword arguments rather than read from globals so tests
can inject stubs without touching the file system or model registry.

Optional keyword arguments unused by the resolved strategy are silently
ignored — the plugin wires every dependency up at startup, and a typo'd
dependency only matters if the active strategy needs it.
"""

from __future__ import annotations

from typing import Any, Callable

from claritymed.core.rag.terms.base import TermService
from claritymed.core.symptoms.eligibility.base import EligibilityStrategy
from claritymed.core.symptoms.eligibility.direct import (
    DirectEligibility,
    EvidenceVocabMap,
)
from claritymed.core.symptoms.eligibility.term_service import (
    SidecarMap,
    TermServiceEligibility,
)
from claritymed.core.symptoms.eligibility.translation import TranslationEligibility
from claritymed.core.symptoms.schemas import (
    EligibilityCatalogConfig,
    TranslationEligibilityEntry,
)
from claritymed.errors import UnknownEligibilityStrategyError


def build_eligibility_strategy(
    cfg: EligibilityCatalogConfig,
    *,
    vocabs: EvidenceVocabMap | None = None,
    sidecars: SidecarMap | None = None,
    term_service_factory: Callable[[], TermService] | None = None,
    translation_overrides: dict[str, Any] | None = None,
) -> EligibilityStrategy:
    """Instantiate the active eligibility strategy.

    Args:
        cfg: The eligibility catalog block from ``configs/symptoms.yaml``.
            ``cfg.active`` selects the strategy; the catalog entry's
            ``kind`` selects the implementation.
        vocabs: Per-dataset evidence vocabularies — required for the
            ``direct`` strategy. The plugin loads these at startup from
            the dataset's evidence-name vocab.
        sidecars: Per-dataset concept sidecars (``evidence_id → concept_id``)
            consumed by the ``term_service`` strategy. Built once by
            ``prepare.py`` and loaded at plugin startup.
        term_service_factory: Zero-arg callable returning a configured
            :class:`TermService`. The factory is invoked lazily so a
            broken term_service config (e.g. missing concepts.jsonl)
            only fails the load when ``term_service`` is the active
            strategy.
        translation_overrides: Optional injection point for the
            ``translation`` strategy's collaborators (``agent_factory``,
            ``provider_resolver``, ``availability_check``, ``registry``).
            Production callers leave this ``None``; tests pass stubs to
            bypass the real LLM and provider env-var lookups.

    Raises:
        UnknownEligibilityStrategyError: The resolved catalog entry's
            ``kind`` has no factory branch.
        EligibilityStrategyConfigError: ``term_service`` strategy
            selected but no factory was provided (operator wiring bug).
        EligibilityStrategyUnavailableError: ``term_service`` strategy
            selected but the resolved service is a
            :class:`~claritymed.core.rag.terms.umls_cmekg.NoOpTermService`.
    """
    entry = cfg.resolved()
    if entry.kind == "direct":
        return DirectEligibility(vocabs=vocabs or {})
    if entry.kind == "term_service":
        if term_service_factory is None:
            raise UnknownEligibilityStrategyError(
                "term_service eligibility selected but no "
                "term_service_factory was provided. Wire one through "
                "the plugin construction path."
            )
        term_service = term_service_factory()
        return TermServiceEligibility(
            term_service=term_service,
            sidecars=sidecars or {},
        )
    if entry.kind == "translation":
        if not isinstance(entry, TranslationEligibilityEntry):
            raise UnknownEligibilityStrategyError(
                f"Expected TranslationEligibilityEntry, got {type(entry).__name__}"
            )
        # Translation always needs a direct strategy underneath — it's
        # the matcher for the translated EN string. The same vocabs map
        # feeds both layers so EN short-circuit and ZH-via-translation
        # produce comparable scores.
        direct = DirectEligibility(vocabs=vocabs or {})
        return TranslationEligibility(
            provider_id=entry.provider_id,
            prompt_name=entry.prompt_name,
            max_tokens=entry.max_tokens,
            direct_strategy=direct,
            **(translation_overrides or {}),
        )
    raise UnknownEligibilityStrategyError(
        f"build_eligibility_strategy has no factory branch for kind="
        f"{entry.kind!r}; available kinds: ['direct', 'term_service', "
        f"'translation']"
    )
