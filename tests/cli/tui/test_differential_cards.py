"""Rendering tests for the multi-card renderer widget tree.

These tests avoid Textual's ``App`` runtime — the widget classes are
tested in isolation via their public API (i18n resolution, Rich text
assembly, tier→color mapping). App-level integration lives in the
smoke suite.
"""

from __future__ import annotations

from claritymed.cli.tui.widgets.differential_cards import (
    DifferentialBanner,
    DifferentialCardBubble,
    _card_text,
    render_differential_ready,
)
from claritymed.core.events import (
    DifferentialCard,
    DifferentialReady,
    DifferentialSessionMeta,
)


def _card(**overrides) -> DifferentialCard:
    payload = {
        "condition_id": "pneumonia",
        "condition_name": "Pneumonia",
        "probability": 0.72,
        "severity_tier": "Urgent",
        "confidence_bucket": "high",
        "confidence_label": "Confident",
        "headline": "Likely Pneumonia",
        "report": "Infection of the lung tissue.",
    }
    payload.update(overrides)
    return DifferentialCard.model_validate(payload)


def test_card_text_contains_headline_and_probability():
    card = _card()
    plain = _card_text(card).plain
    assert "Likely Pneumonia" in plain
    assert "72%" in plain
    assert "Confident" in plain
    assert "Severity tier: Urgent" in plain


def test_card_text_contains_report():
    card = _card()
    plain = _card_text(card).plain
    assert "Infection of the lung tissue." in plain


def test_card_text_omits_next_steps_and_citations():
    """Per-card next-steps + citations were removed — the LLM's summary
    paragraph carries the actionable guidance and authority anchoring."""
    card = _card()
    plain = _card_text(card).plain
    assert "Next steps" not in plain
    assert "Authoritative source" not in plain
    assert "Peer-reviewed reference" not in plain


def test_card_text_skips_none_headline_but_keeps_confidence_chip():
    """Standard rank 2+ cards have ``headline=None``; renderer must not
    emit a leading bold line but the confidence chip still shows.

    Before this cut the widget rendered ``Also consider Pneumonia`` bold
    above the description — a duplicate of the card title. Now the None
    case falls straight through to the confidence chip.
    """
    card = _card(headline=None)
    plain = _card_text(card).plain
    assert "Likely Pneumonia" not in plain
    assert "Also consider" not in plain
    # Chip + severity + report body still render.
    assert "Confident" in plain
    assert "72%" in plain
    assert "Severity tier: Urgent" in plain
    assert "Infection of the lung tissue." in plain


def test_card_bubble_constructs_without_language():
    card = _card()
    bubble = DifferentialCardBubble(card)
    assert bubble._card is card


def test_banner_constructs_with_key():
    banner = DifferentialBanner("symptoms.card.banner.hit_cap", "en")
    assert banner._banner_key == "symptoms.card.banner.hit_cap"


def test_render_differential_ready_builds_container_with_cards():
    ev = DifferentialReady(
        cards=[_card(), _card(condition_id="influenza", condition_name="Influenza")],
        session=DifferentialSessionMeta(),
        language="en",
    )
    container = render_differential_ready(ev)
    assert len(container._cards) == 2
    assert container._banner_key is None


def test_render_differential_ready_carries_banner_key():
    ev = DifferentialReady(
        cards=[],
        session=DifferentialSessionMeta(
            cancelled=True,
            meets_confidence_threshold=False,
            banner_key="symptoms.card.banner.cancelled_no_result",
        ),
        language="en",
    )
    container = render_differential_ready(ev)
    assert container._banner_key == "symptoms.card.banner.cancelled_no_result"
    assert container._cards == []
