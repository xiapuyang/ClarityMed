"""TermServiceEligibility — concept-grounded cross-lingual matching.

The TermService is stubbed in-process; integration against
``UmlsCmekgLocalService`` lives in the eligibility A/B harness (Unit 10).
The strategy itself is pure dispatch + intersection, so the stub is
enough to cover branching.
"""

from __future__ import annotations

import pytest

from claritymed.core.rag.terms.base import (
    Alias,
    ConceptHit,
    ConceptLanguage,
    ConceptType,
    TermService,
)
from claritymed.core.rag.terms.umls_cmekg import NoOpTermService
from claritymed.core.schemas.patient import Profile
from claritymed.core.symptoms.eligibility import (
    EligibilityResult,
    TermServiceEligibility,
)
from claritymed.core.symptoms.schemas import DatasetSpec
from claritymed.errors import EligibilityStrategyUnavailableError


class _StubTermService(TermService):
    """Lookup table: ``{(surface_lower, language): [(concept_id, type), ...]}``.

    Implements the bare minimum the eligibility strategy uses — the rest
    of the protocol is intentionally absent so a mistake reaching for a
    non-stubbed method surfaces immediately.
    """

    def __init__(
        self,
        table: dict[tuple[str, ConceptLanguage], list[tuple[str, ConceptType]]],
    ) -> None:
        self._table = table

    def lookup(self, surface: str, language: ConceptLanguage) -> list[ConceptHit]:
        key = (surface.lower(), language)
        return [
            ConceptHit(
                concept_id=cid,
                surface=surface,
                language=language,
                score=1.0,
                type=ctype,
                aliases=(),
            )
            for cid, ctype in self._table.get(key, [])
        ]

    def cross_lingual_aliases(self, concept_id: str) -> list[Alias]:
        return []


def _dataset(
    *,
    id_: str = "ddxplus",
    high_spec: list[str] | None = None,
) -> DatasetSpec:
    return DatasetSpec(
        id=id_,
        enabled=True,
        model_ids=["typed_basd_v1"],
        maxstep=8,
        severity_high_specificity_evidence_ids=high_spec or [],
    )


def _profile() -> Profile:
    return Profile()


_DDXPLUS_SIDECAR = {
    "ddxplus": {
        "E_1": "C_CHEST_PAIN",
        "E_2": "C_NAUSEA",
        "E_3": "C_DYSPNEA",
        "E_HS": "C_RADIATING",
    }
}


async def test_en_two_concept_hits_in_scope() -> None:
    """Two distinct evidence vocabularies match → eligible by D7 threshold."""
    term_service = _StubTermService(
        {
            ("chest pain", "en"): [("C_CHEST_PAIN", "symptom")],
            ("nausea", "en"): [("C_NAUSEA", "symptom")],
        }
    )
    strategy = TermServiceEligibility(
        term_service=term_service, sidecars=_DDXPLUS_SIDECAR
    )
    result = await strategy.check(
        complaint="I have chest pain and nausea",
        language="en",
        profile=_profile(),
        dataset=_dataset(),
    )
    assert isinstance(result, EligibilityResult)
    assert result.eligible is True
    assert result.reason == "in_scope"
    assert set(result.evidence_hits) == {"E_1", "E_2"}


async def test_zh_cross_lingual_match_preserves_parity() -> None:
    """A Chinese complaint resolves to the same concept ids as the English
    label via the term service's own cross-lingual aliases. The strategy
    treats both as equivalent — the sidecar is concept-id keyed, not
    language-keyed."""
    term_service = _StubTermService(
        {
            ("胸口疼", "zh"): [("C_CHEST_PAIN", "symptom")],
            ("恶心", "zh"): [("C_NAUSEA", "symptom")],
        }
    )
    strategy = TermServiceEligibility(
        term_service=term_service, sidecars=_DDXPLUS_SIDECAR
    )
    result = await strategy.check(
        complaint="胸口疼 恶心",
        language="zh",
        profile=_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is True
    assert set(result.evidence_hits) == {"E_1", "E_2"}


async def test_high_specificity_single_hit_short_circuits() -> None:
    term_service = _StubTermService({("radiating", "en"): [("C_RADIATING", "symptom")]})
    strategy = TermServiceEligibility(
        term_service=term_service, sidecars=_DDXPLUS_SIDECAR
    )
    result = await strategy.check(
        complaint="pain radiating from somewhere",
        language="en",
        profile=_profile(),
        dataset=_dataset(high_spec=["E_HS"]),
    )
    assert result.eligible is True
    assert "E_HS" in result.evidence_hits


async def test_drug_concept_type_does_not_contribute() -> None:
    """Eligibility only triggers on symptom / disease concepts. Drug /
    procedure hits are noise for the "is this in scope for the diagnostic
    model" question."""
    term_service = _StubTermService(
        {
            ("ibuprofen", "en"): [("C_DRUG", "drug")],
            ("nausea", "en"): [("C_NAUSEA", "symptom")],
        }
    )
    strategy = TermServiceEligibility(
        term_service=term_service,
        sidecars={
            "ddxplus": {
                "E_DRUG": "C_DRUG",
                "E_NAUSEA": "C_NAUSEA",
            }
        },
    )
    result = await strategy.check(
        complaint="taking ibuprofen and feeling nausea",
        language="en",
        profile=_profile(),
        dataset=_dataset(),
    )
    # Only the symptom contributes; one distinct hit doesn't meet the
    # ≥2 threshold (and there's no high-spec marker).
    assert result.eligible is False
    assert "E_NAUSEA" in result.evidence_hits
    assert "E_DRUG" not in result.evidence_hits


async def test_concept_not_in_sidecar_falls_through_out_of_scope() -> None:
    """The complaint resolves to concept ids, but none are in the sidecar
    — the dataset doesn't cover those conditions."""
    term_service = _StubTermService({("toothache", "en"): [("C_TOOTHACHE", "symptom")]})
    strategy = TermServiceEligibility(
        term_service=term_service, sidecars=_DDXPLUS_SIDECAR
    )
    result = await strategy.check(
        complaint="I have a toothache",
        language="en",
        profile=_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is False
    assert result.reason == "out_of_scope"


async def test_empty_complaint_out_of_scope() -> None:
    term_service = _StubTermService({})
    strategy = TermServiceEligibility(
        term_service=term_service, sidecars=_DDXPLUS_SIDECAR
    )
    result = await strategy.check(
        complaint="",
        language="en",
        profile=_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is False
    assert result.reason == "out_of_scope"


async def test_unknown_dataset_falls_through_strategy_unavailable() -> None:
    """An asked-about dataset id missing from the sidecar map surfaces as
    strategy_unavailable rather than crashing."""
    term_service = _StubTermService({})
    strategy = TermServiceEligibility(
        term_service=term_service, sidecars=_DDXPLUS_SIDECAR
    )
    result = await strategy.check(
        complaint="chest pain",
        language="en",
        profile=_profile(),
        dataset=_dataset(id_="future_dataset"),
    )
    assert result.eligible is False
    assert result.reason == "strategy_unavailable"


async def test_noop_term_service_raises_at_construct() -> None:
    """A NoOp service injected means the operator hasn't configured a
    real terminology export; the strategy refuses to silently produce
    empty results."""
    with pytest.raises(EligibilityStrategyUnavailableError) as info:
        TermServiceEligibility(
            term_service=NoOpTermService(), sidecars=_DDXPLUS_SIDECAR
        )
    assert "NoOp" in str(info.value)


async def test_ngram_surface_matches_multi_word_concept() -> None:
    """The n-gram sweep catches multi-word concepts the single-token loop
    would miss. The strategy must mirror :func:`expand_query`'s behavior
    so the eligibility A/B has parity with the retrieval rewriter."""
    term_service = _StubTermService(
        {
            ("abdominal pain", "en"): [("C_ABDO_PAIN", "symptom")],
            ("nausea", "en"): [("C_NAUSEA", "symptom")],
        }
    )
    strategy = TermServiceEligibility(
        term_service=term_service,
        sidecars={
            "ddxplus": {
                "E_ABDO": "C_ABDO_PAIN",
                "E_NAUSEA": "C_NAUSEA",
            }
        },
    )
    result = await strategy.check(
        complaint="abdominal pain and nausea",
        language="en",
        profile=_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is True
    assert set(result.evidence_hits) == {"E_ABDO", "E_NAUSEA"}
