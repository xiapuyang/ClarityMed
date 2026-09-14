"""Concept-grounded eligibility strategy.

Cross-lingual eligibility via a :class:`~claritymed.core.rag.terms.base.TermService`:
tokenize the complaint (same regex + n-gram rule as the RAG-query
expander), look each surface up in the term service, collect the
returned concept ids, and intersect against a per-dataset sidecar
(``evidence_id → concept_id``) built once by ``prepare.py``.

The strategy adds no Phoenix / LLM hops — ``TermService.lookup`` is a
local dict lookup. ZH and EN parity comes from the term service's own
cross-lingual aliases: a Chinese complaint resolves to the same
``concept_id`` as the English label, so the sidecar intersection works
identically in both languages.

Fail-loud at construct time when the injected service is a
:class:`~claritymed.core.rag.terms.umls_cmekg.NoOpTermService`. The
caller can catch :class:`~claritymed.errors.EligibilityStrategyUnavailableError`
and fall through to another strategy, but the strategy itself refuses
to silently produce empty results.
"""

from __future__ import annotations

from claritymed.core.rag.terms.base import (
    ConceptHit,
    ConceptLanguage,
    ConceptType,
    TermService,
)
from claritymed.core.rag.terms.expansion import MAX_NGRAM, _TOKEN_RE
from claritymed.core.rag.terms.umls_cmekg import NoOpTermService
from claritymed.core.schemas.patient import Profile
from claritymed.core.symptoms.eligibility.base import (
    EligibilityResult,
    EligibilityStrategy,
    _CONFIDENCE_SATURATION,
    _MIN_DISTINCT_HITS,
)
from claritymed.core.symptoms.schemas import DatasetSpec
from claritymed.errors import EligibilityStrategyUnavailableError

# Concept types worth treating as eligibility signal. Drug / procedure
# matches still contribute to retrieval but are noise for the "is this
# in scope for a diagnostic model" question.
_ELIGIBLE_CONCEPT_TYPES: set[ConceptType] = {"symptom", "disease"}


# Per-dataset sidecar: dataset_id → evidence_id → concept_id (one
# canonical concept per evidence; collisions resolved upstream by
# ``prepare.py`` picking the top hit).
SidecarMap = dict[str, dict[str, str]]


def _surfaces(complaint: str) -> list[str]:
    """Token + sliding n-gram surfaces, matching ``expand_query``.

    Multi-word concepts ("abdominal pain") are caught by the n-gram
    sweep; single-token concepts ("nausea") are caught by the unigram
    loop. We dedupe at the call site since the lookup itself is cheap.
    """
    tokens = _TOKEN_RE.findall(complaint)
    if not tokens:
        return []
    out: list[str] = list(tokens)
    for n in range(2, MAX_NGRAM + 1):
        for i in range(0, len(tokens) - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    return out


class TermServiceEligibility(EligibilityStrategy):
    """Cross-lingual concept-grounded matcher.

    Construct with a configured :class:`TermService` and the per-dataset
    sidecar map. The strategy does not load anything from disk itself —
    sidecar loading is the plugin's responsibility (Unit 15), keeping
    this module pure / mockable.
    """

    def __init__(
        self,
        *,
        term_service: TermService,
        sidecars: SidecarMap,
    ) -> None:
        if isinstance(term_service, NoOpTermService):
            raise EligibilityStrategyUnavailableError(
                "TermServiceEligibility requires a non-NoOp TermService. "
                "Configure `term_service.active != 'none'` in "
                "configs/retrieval.yaml or pick a different eligibility "
                "strategy in configs/symptoms.yaml.eligibility.active."
            )
        self._term_service = term_service
        # Outer map preserved as-is; inner sidecar inverted to
        # concept_id → evidence_id for O(1) hit lookup at check time.
        self._evidence_by_concept: dict[str, dict[str, str]] = {}
        for ds_id, per_evidence in sidecars.items():
            inverted: dict[str, str] = {}
            for evidence_id, concept_id in per_evidence.items():
                inverted[concept_id] = evidence_id
            self._evidence_by_concept[ds_id] = inverted

    async def check(
        self,
        complaint: str,
        language: ConceptLanguage,
        profile: Profile,
        dataset: DatasetSpec,
    ) -> EligibilityResult:
        evidence_by_concept = self._evidence_by_concept.get(dataset.id)
        if evidence_by_concept is None:
            return EligibilityResult(eligible=False, reason="strategy_unavailable")

        surfaces = _surfaces(complaint)
        if not surfaces:
            return EligibilityResult(eligible=False, reason="out_of_scope")

        # Distinct evidence ids hit by any concept found in the complaint.
        hit_evidence: set[str] = set()
        seen_surfaces: set[str] = set()
        for surface in surfaces:
            key = surface.lower()
            if key in seen_surfaces:
                continue
            seen_surfaces.add(key)
            hits: list[ConceptHit] = self._term_service.lookup(surface, language)
            for hit in hits:
                if hit.type not in _ELIGIBLE_CONCEPT_TYPES:
                    continue
                evidence_id = evidence_by_concept.get(hit.concept_id)
                if evidence_id is not None:
                    hit_evidence.add(evidence_id)

        high_spec_ids = set(dataset.severity_high_specificity_evidence_ids)
        high_spec_hit = bool(hit_evidence & high_spec_ids)
        sorted_hits = sorted(hit_evidence)
        confidence = min(1.0, len(sorted_hits) / _CONFIDENCE_SATURATION)

        if high_spec_hit or len(sorted_hits) >= _MIN_DISTINCT_HITS:
            return EligibilityResult(
                eligible=True,
                reason="in_scope",
                evidence_hits=sorted_hits,
                confidence=confidence,
            )
        return EligibilityResult(
            eligible=False,
            reason="out_of_scope",
            evidence_hits=sorted_hits,
            confidence=confidence,
        )
