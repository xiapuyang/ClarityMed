"""Tests for the multi-card renderer's hydration helper."""

from __future__ import annotations

import asyncio

import pytest

from claritymed.core.events import DifferentialReady
from claritymed.core.symptoms.conditions_catalog import SymptomsConditionsCatalog
from claritymed.core.symptoms.wire import DifferentialRow
from claritymed.orchestrator.features.symptoms_card_builder import (
    build_differential_ready,
    build_session_meta,
    confidence_bucket_for,
    confidence_label_for,
    directional_headline,
)


@pytest.fixture()
def catalog() -> SymptomsConditionsCatalog:
    return SymptomsConditionsCatalog()


def _row(condition_id: str, probability: float, severity: int) -> DifferentialRow:
    return DifferentialRow(
        condition_id=condition_id,
        condition_idx=0,
        condition_name=condition_id.replace("_", " ").title(),
        probability=probability,
        severity=severity,
    )


# ---- confidence bucketing ------------------------------------------------


def test_bucket_very_high():
    assert confidence_bucket_for(0.90, "en") == "very_high"
    assert confidence_bucket_for(0.75, "en") == "very_high"


def test_bucket_high():
    assert confidence_bucket_for(0.60, "en") == "high"
    assert confidence_bucket_for(0.55, "en") == "high"


def test_bucket_moderate():
    assert confidence_bucket_for(0.40, "en") == "moderate"
    assert confidence_bucket_for(0.30, "en") == "moderate"


def test_bucket_low():
    assert confidence_bucket_for(0.05, "en") == "low"


def test_confidence_labels_localized():
    assert confidence_label_for("very_high", "en") == "Highly Confident"
    assert confidence_label_for("very_high", "zh") == "非常确定"


# ---- directional headline ------------------------------------------------


def test_headline_rank1_likely():
    h = directional_headline(1, 0.72, "Urgent", "Pneumonia", "en")
    assert h == "Likely Pneumonia"


def test_headline_rank1_probably():
    h = directional_headline(1, 0.45, "Moderate", "Bronchitis", "en")
    assert h == "Probably Bronchitis"


def test_headline_rank1_consider():
    h = directional_headline(1, 0.20, "Moderate", "URTI", "en")
    assert h == "Consider URTI"


def test_headline_rank2_standard_is_none():
    """Standard rank 2+ headline was removed — card title already carries name."""
    h = directional_headline(2, 0.35, "Moderate", "Bronchitis", "en")
    assert h is None


def test_headline_rank2_unlikely_but_rule_out_critical():
    """Low-probability rank-2 with Critical severity → rule-out variant."""
    h = directional_headline(2, 0.10, "Critical", "PE", "en")
    assert h is not None and "Unlikely" in h and "rule out" in h


def test_headline_rank2_unlikely_but_rule_out_urgent():
    h = directional_headline(2, 0.10, "Urgent", "NSTEMI", "en")
    assert h is not None and "Unlikely" in h and "rule out" in h


def test_headline_rank2_moderate_severity_stays_none():
    """Low probability at Moderate severity does NOT get the rule-out variant.

    Moderate is below the ``_UNLIKELY_RULE_OUT_TIERS`` bar, so the card
    falls into the standard rank 2+ path — which now returns ``None``
    since the previous ``Also consider`` template was removed.
    """
    h = directional_headline(2, 0.10, "Moderate", "URTI", "en")
    assert h is None


def test_headline_rank2_high_probability_stays_none_even_at_critical():
    """Rule-out variant only fires below the low-probability threshold.

    Above threshold, even a Critical rank-2 falls into the standard
    path — which now returns ``None``.
    """
    h = directional_headline(2, 0.55, "Critical", "PE", "en")
    assert h is None


def test_headline_zh_interpolation():
    h = directional_headline(1, 0.72, "Urgent", "肺炎", "zh")
    assert h == "很可能是肺炎"


# ---- build_session_meta / banner_key -------------------------------------


def test_banner_key_normal_completion_none():
    meta = build_session_meta({"eligible": True}, cards_empty=False)
    assert meta.banner_key is None
    assert not meta.cancelled and not meta.hit_cap


def test_banner_key_hit_cap():
    meta = build_session_meta({"eligible": True, "hit_cap": True}, cards_empty=False)
    assert meta.banner_key == "symptoms.card.banner.hit_cap"


def test_banner_key_severity_override_takes_precedence_over_cancelled():
    meta = build_session_meta(
        {
            "eligible": True,
            "cancelled": True,
            "severity_override": True,
            "meets_confidence_threshold": False,
        },
        cards_empty=False,
    )
    assert meta.banner_key == "symptoms.card.banner.severity_override"


def test_banner_key_cancelled_early_meaningful():
    meta = build_session_meta(
        {
            "eligible": True,
            "cancelled": True,
            "meets_confidence_threshold": True,
        },
        cards_empty=False,
    )
    assert meta.banner_key == "symptoms.card.banner.cancelled_early_meaningful"


def test_banner_key_cancelled_no_result():
    meta = build_session_meta(
        {
            "eligible": True,
            "cancelled": True,
            "meets_confidence_threshold": False,
        },
        cards_empty=True,
    )
    assert meta.banner_key == "symptoms.card.banner.cancelled_no_result"


def test_banner_key_session_expired():
    meta = build_session_meta(
        {"eligible": True, "session_expired": True}, cards_empty=True
    )
    assert meta.banner_key == "symptoms.card.banner.session_expired"


# ---- build_differential_ready end-to-end ---------------------------------


def test_build_event_normal_completion(catalog: SymptomsConditionsCatalog):
    rows = [
        _row("pneumonia", 0.72, 2),
        _row("bronchitis", 0.18, 3),
        _row("influenza", 0.05, 3),
        _row("urti", 0.03, 4),
    ]
    result = {"eligible": True, "_raw_differential": rows}
    ev = build_differential_ready(result, catalog=catalog, language="en", top_n_cap=3)
    assert isinstance(ev, DifferentialReady)
    assert len(ev.cards) == 3, "clipped to top_n_cap"
    assert ev.cards[0].condition_id == "pneumonia"
    assert ev.cards[0].probability == pytest.approx(0.72)
    assert ev.cards[0].confidence_bucket == "high"
    assert ev.cards[0].headline == "Likely Pneumonia"
    assert ev.cards[0].report
    assert ev.session.banner_key is None


def test_build_event_returns_none_when_user_declined(catalog):
    ev = build_differential_ready(
        {"eligible": True, "user_declined": True, "_raw_differential": []},
        catalog=catalog,
        language="en",
        top_n_cap=3,
    )
    assert ev is None


def test_build_event_returns_none_when_ineligible(catalog):
    ev = build_differential_ready(
        {"eligible": False},
        catalog=catalog,
        language="en",
        top_n_cap=3,
    )
    assert ev is None


def test_build_event_returns_none_on_server_error(catalog):
    ev = build_differential_ready(
        {"eligible": True, "server_error": True},
        catalog=catalog,
        language="en",
        top_n_cap=3,
    )
    assert ev is None


def test_build_event_hit_cap_uses_partial_and_sets_banner(catalog):
    rows = [_row("pneumonia", 0.65, 2), _row("bronchitis", 0.20, 3)]
    ev = build_differential_ready(
        {
            "eligible": True,
            "hit_cap": True,
            "_raw_partial_differential": rows,
        },
        catalog=catalog,
        language="en",
        top_n_cap=3,
    )
    assert ev is not None
    assert ev.session.banner_key == "symptoms.card.banner.hit_cap"
    assert len(ev.cards) == 2


def test_build_event_drops_zero_probability_rows(catalog):
    rows = [
        _row("pneumonia", 0.60, 2),
        _row("bronchitis", 0.0, 3),  # dropped
    ]
    ev = build_differential_ready(
        {"eligible": True, "_raw_differential": rows},
        catalog=catalog,
        language="en",
        top_n_cap=5,
    )
    assert len(ev.cards) == 1
    assert ev.cards[0].condition_id == "pneumonia"


def test_build_event_skips_uncatalogued_condition_with_warning(catalog, caplog):
    rows = [
        _row("pneumonia", 0.60, 2),
        _row("fictional_unknown_condition", 0.30, 3),
    ]
    with caplog.at_level("WARNING"):
        ev = build_differential_ready(
            {"eligible": True, "_raw_differential": rows},
            catalog=catalog,
            language="en",
            top_n_cap=5,
        )
    assert len(ev.cards) == 1
    assert ev.cards[0].condition_id == "pneumonia"
    assert any("fictional_unknown_condition" in rec.message for rec in caplog.records)


def test_build_event_severity_override_case_e(catalog):
    """Case E: cancelled + severity_override — critical low-prob card gets 'rule out' headline."""
    rows = [
        _row("panic_attack", 0.55, 5),
        _row("possible_nstemi_stemi", 0.08, 1),
    ]
    ev = build_differential_ready(
        {
            "eligible": True,
            "cancelled": True,
            "severity_override": True,
            "meets_confidence_threshold": False,
            "max_low_severity_seen": 1,
            "_raw_partial_differential": rows,
        },
        catalog=catalog,
        language="en",
        top_n_cap=3,
    )
    assert ev.session.banner_key == "symptoms.card.banner.severity_override"
    # Panic ranks 1 (higher probability), NSTEMI ranks 2 with rule-out
    assert ev.cards[0].condition_id == "panic_attack"
    assert ev.cards[1].condition_id == "possible_nstemi_stemi"
    assert "Unlikely" in ev.cards[1].headline and "rule out" in ev.cards[1].headline


def test_build_event_zh_uses_zh_catalog_and_labels(catalog):
    rows = [_row("pneumonia", 0.72, 2)]
    ev = build_differential_ready(
        {"eligible": True, "_raw_differential": rows},
        catalog=catalog,
        language="zh",
        top_n_cap=3,
    )
    assert ev.cards[0].condition_name == "肺炎"
    assert ev.cards[0].confidence_label == "较为确定"
    assert ev.cards[0].headline == "很可能是肺炎"


def test_build_event_cancelled_no_result_empty_cards(catalog):
    """Case D: cancelled with no partial → empty cards + banner."""
    ev = build_differential_ready(
        {
            "eligible": True,
            "cancelled": True,
            "meets_confidence_threshold": False,
            "severity_override": False,
            "_raw_partial_differential": [],
        },
        catalog=catalog,
        language="en",
        top_n_cap=3,
    )
    assert ev is not None
    assert ev.cards == []
    assert ev.session.banner_key == "symptoms.card.banner.cancelled_no_result"


# ---- integration with plugin's _maybe_emit_differential_ready ----------


class _StubDeps:
    def __init__(self) -> None:
        self.event_queue: asyncio.Queue = asyncio.Queue()


@pytest.mark.asyncio
async def test_plugin_emits_differential_ready(catalog):
    """SymptomsFeature._maybe_emit_differential_ready puts DifferentialReady on the queue."""
    from claritymed.core.symptoms.registry import DatasetRegistry
    from claritymed.orchestrator.features.symptoms_plugin import SymptomsFeature
    from claritymed.core.symptoms.eligibility.base import (
        EligibilityStrategy,
        EligibilityResult,
    )
    from claritymed.config import load_symptoms_config

    class _NoOpEligibility(EligibilityStrategy):
        async def check(self, complaint, language, profile, dataset):
            return EligibilityResult(eligible=True, reason=None, confidence=1.0)

        @property
        def strategy_id(self) -> str:
            return "test"

    cfg = load_symptoms_config()
    registry = DatasetRegistry(cfg.datasets)

    class _FakeClient:
        pass

    feature = SymptomsFeature(
        config=cfg,
        registry=registry,
        client=_FakeClient(),
        eligibility=_NoOpEligibility(),
    )

    deps = _StubDeps()
    result = {
        "eligible": True,
        "_raw_differential": [_row("pneumonia", 0.72, 2)],
    }
    await feature._maybe_emit_differential_ready(deps, result, "en")

    assert deps.event_queue.qsize() == 1
    ev = deps.event_queue.get_nowait()
    assert isinstance(ev, DifferentialReady)
    assert ev.cards[0].condition_id == "pneumonia"
