"""EN-only token-match eligibility strategy.

Tokenizes the complaint with the shared :data:`_TOKEN_RE` from query
expansion (same boundaries as the RAG path — CJK + ASCII words) and
intersects against per-evidence EN vocabularies. The strategy holds a
flat per-evidence token set built once at construct time; check-time
work is O(evidences) set intersection.

Threshold (origin D7):

* eligible if at least :data:`_MIN_DISTINCT_HITS` distinct evidence
  vocabularies match, OR
* eligible if any evidence listed in
  :attr:`DatasetSpec.severity_high_specificity_evidence_ids` matches
  (the "rule out the bad thing" short-circuit — a single high-spec hit
  is enough to justify running the model).
"""

from __future__ import annotations

from claritymed.core.rag.terms.base import ConceptLanguage
from claritymed.core.rag.terms.expansion import _TOKEN_RE
from claritymed.core.schemas.patient import Profile
from claritymed.core.symptoms.eligibility.base import (
    EligibilityResult,
    EligibilityStrategy,
)
from claritymed.core.symptoms.schemas import DatasetSpec

# Origin D7 threshold: ≥2 distinct vocabulary hits = in_scope.
_MIN_DISTINCT_HITS = 2

# Cap for the normalized confidence score: matching this many distinct
# evidences saturates at 1.0. Picked as a "this is clearly in-scope"
# anchor — five separate evidence vocabularies hitting at once is more
# than enough signal.
_CONFIDENCE_SATURATION = 5.0

# Per-dataset vocab map. Outer key is ``DatasetSpec.id``, inner key is
# the evidence id (e.g. DDXPlus ``E_91``), value is the set of EN
# phrases the loader extracted for that evidence (typically the
# question prompt plus value labels).
EvidenceVocabMap = dict[str, dict[str, frozenset[str]]]


def _tokens_lower(text: str) -> set[str]:
    """Lowercased token set for ``text`` using the shared tokenizer.

    Empty input collapses to an empty set without raising.
    """
    return {tok.lower() for tok in _TOKEN_RE.findall(text)}


def _flatten_evidence_vocab(
    per_evidence: dict[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    """Pre-tokenize each evidence's phrases into a flat lowercase set."""
    flat: dict[str, frozenset[str]] = {}
    for evidence_id, phrases in per_evidence.items():
        tokens: set[str] = set()
        for phrase in phrases:
            tokens.update(t.lower() for t in _TOKEN_RE.findall(phrase))
        flat[evidence_id] = frozenset(tokens)
    return flat


class DirectEligibility(EligibilityStrategy):
    """Single-language evidence-vocab matcher.

    The strategy is dataset-aware via the per-construct ``vocabs`` map:
    each registered dataset id resolves to its evidence-id → token-set
    dictionary. Datasets not in the map silently report
    ``strategy_unavailable`` so a partially-initialised plugin (e.g.
    one dataset's prepare.py hasn't run) does not block the others.

    The complaint's language must match the dataset's native vocab
    language (``DatasetSpec.native_language``). Mismatched complaints
    should route through the ``translation`` strategy — which will
    translate to the dataset's native language and re-enter direct
    matching — or through the ``term_service`` strategy (Unit 8),
    which is cross-lingual by design.
    """

    def __init__(self, *, vocabs: EvidenceVocabMap):
        self._vocabs: dict[str, dict[str, frozenset[str]]] = {
            ds_id: _flatten_evidence_vocab(per_evidence)
            for ds_id, per_evidence in vocabs.items()
        }

    async def check(
        self,
        complaint: str,
        language: ConceptLanguage,
        profile: Profile,
        dataset: DatasetSpec,
    ) -> EligibilityResult:
        if language != dataset.native_language:
            return EligibilityResult(eligible=False, reason="strategy_unavailable")
        per_evidence = self._vocabs.get(dataset.id)
        if per_evidence is None:
            return EligibilityResult(eligible=False, reason="strategy_unavailable")

        complaint_tokens = _tokens_lower(complaint)
        if not complaint_tokens:
            return EligibilityResult(eligible=False, reason="out_of_scope")

        high_spec_ids = set(dataset.severity_high_specificity_evidence_ids)
        hits: list[str] = []
        high_spec_hit = False
        for evidence_id, vocab in per_evidence.items():
            if complaint_tokens & vocab:
                hits.append(evidence_id)
                if evidence_id in high_spec_ids:
                    high_spec_hit = True

        hits.sort()
        confidence = min(1.0, len(hits) / _CONFIDENCE_SATURATION)
        if high_spec_hit or len(hits) >= _MIN_DISTINCT_HITS:
            return EligibilityResult(
                eligible=True,
                reason="in_scope",
                evidence_hits=hits,
                confidence=confidence,
            )
        return EligibilityResult(
            eligible=False,
            reason="out_of_scope",
            evidence_hits=hits,
            confidence=confidence,
        )
