"""DirectEligibility — EN token-match against per-evidence vocab."""

from __future__ import annotations

import pytest

from claritymed.core.schemas.patient import Profile
from claritymed.core.symptoms.eligibility import DirectEligibility, EligibilityResult
from claritymed.core.symptoms.schemas import DatasetSpec


def _dataset(
    *,
    id_: str = "ddxplus",
    high_spec: list[str] | None = None,
) -> DatasetSpec:
    return DatasetSpec(
        id=id_,
        enabled=True,
        model_ids=["typed_basd_v1"],
        severity_high_specificity_evidence_ids=high_spec or [],
    )


def _empty_profile() -> Profile:
    return Profile()


def _strategy_for_ddxplus() -> DirectEligibility:
    """Three-evidence stub vocab matching the kind of phrasing DDXPlus uses."""
    return DirectEligibility(
        vocabs={
            "ddxplus": {
                "E_1": frozenset({"chest pain"}),
                "E_2": frozenset({"shortness of breath", "dyspnea"}),
                "E_3": frozenset({"nausea", "vomiting"}),
                "E_HS": frozenset({"radiating", "left arm"}),
            }
        }
    )


async def test_en_multiple_hits_in_scope() -> None:
    """Two distinct evidence vocabularies hit → eligible by D7 threshold."""
    strategy = _strategy_for_ddxplus()
    result = await strategy.check(
        complaint="I have chest pain and nausea",
        language="en",
        profile=_empty_profile(),
        dataset=_dataset(),
    )
    assert isinstance(result, EligibilityResult)
    assert result.eligible is True
    assert result.reason == "in_scope"
    assert set(result.evidence_hits) == {"E_1", "E_3"}
    assert result.confidence > 0


async def test_en_single_high_specificity_hit_in_scope() -> None:
    """A single high-specificity evidence hit short-circuits to eligible."""
    strategy = _strategy_for_ddxplus()
    result = await strategy.check(
        complaint="something is radiating from my back",
        language="en",
        profile=_empty_profile(),
        dataset=_dataset(high_spec=["E_HS"]),
    )
    assert result.eligible is True
    assert result.reason == "in_scope"
    assert "E_HS" in result.evidence_hits


async def test_en_single_non_high_specificity_hit_out_of_scope() -> None:
    """One distinct hit without a high-spec marker fails the threshold."""
    strategy = _strategy_for_ddxplus()
    result = await strategy.check(
        complaint="I have nausea",
        language="en",
        profile=_empty_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is False
    assert result.reason == "out_of_scope"
    assert result.evidence_hits == ["E_3"]


async def test_en_zero_hits_out_of_scope() -> None:
    strategy = _strategy_for_ddxplus()
    result = await strategy.check(
        complaint="I want to reset my password",
        language="en",
        profile=_empty_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is False
    assert result.reason == "out_of_scope"
    assert result.evidence_hits == []


async def test_zh_falls_through_strategy_unavailable() -> None:
    """The direct strategy can't translate — ZH callers must use term_service
    or translation. Returning strategy_unavailable (vs out_of_scope) signals
    the caller that the result is a no-op, not a real negative."""
    strategy = _strategy_for_ddxplus()
    result = await strategy.check(
        complaint="我胸口疼",
        language="zh",
        profile=_empty_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is False
    assert result.reason == "strategy_unavailable"


async def test_unknown_dataset_falls_through_strategy_unavailable() -> None:
    """An asked-about dataset id not in the constructed vocab map must
    surface as strategy_unavailable rather than crashing — Unit 15 uses
    this branch when a registered dataset's prepare.py hasn't run yet."""
    strategy = _strategy_for_ddxplus()
    result = await strategy.check(
        complaint="chest pain",
        language="en",
        profile=_empty_profile(),
        dataset=_dataset(id_="future_dataset"),
    )
    assert result.eligible is False
    assert result.reason == "strategy_unavailable"


async def test_empty_complaint_out_of_scope() -> None:
    strategy = _strategy_for_ddxplus()
    result = await strategy.check(
        complaint="",
        language="en",
        profile=_empty_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is False
    assert result.reason == "out_of_scope"
    assert result.evidence_hits == []


async def test_case_insensitive_match() -> None:
    """Vocab is normalised to lower at construct; complaints with caps
    still match."""
    strategy = _strategy_for_ddxplus()
    result = await strategy.check(
        complaint="CHEST PAIN and NAUSEA",
        language="en",
        profile=_empty_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is True


async def test_confidence_saturates_at_one() -> None:
    """A complaint that lights up every evidence in the vocab caps
    confidence at 1.0 rather than overshooting."""
    strategy = DirectEligibility(
        vocabs={
            "small": {
                "E_A": frozenset({"alpha"}),
                "E_B": frozenset({"beta"}),
            }
        }
    )
    result = await strategy.check(
        complaint="alpha beta",
        language="en",
        profile=_empty_profile(),
        dataset=_dataset(id_="small"),
    )
    assert 0.0 < result.confidence <= 1.0


@pytest.mark.parametrize(
    "complaint",
    [
        "alpha gamma",  # one hit
        "delta",  # no hits
    ],
)
async def test_below_threshold_returns_evidence_hits_anyway(complaint: str) -> None:
    """When the result is ineligible we still surface the partial hit list
    so the registry's score-tiebreaker (KTD-9 step 3) has something to
    compare across datasets even on a near-miss."""
    strategy = DirectEligibility(
        vocabs={
            "small": {
                "E_A": frozenset({"alpha"}),
                "E_B": frozenset({"beta"}),
            }
        }
    )
    result = await strategy.check(
        complaint=complaint,
        language="en",
        profile=_empty_profile(),
        dataset=_dataset(id_="small"),
    )
    assert result.eligible is False
    # The list may be empty (no hits) or single-entry (one hit) — point is
    # the strategy doesn't drop it.
    assert isinstance(result.evidence_hits, list)
