"""Multi-card renderer for the ``differential_ready`` SSE event.

Renders one Textual ``Static`` per :class:`DifferentialCard` in the
event payload, plus (optionally) an above-cards banner ``Static``
whose text is resolved via :func:`t` from the server-supplied
``session.banner_key``. Card block is meant to be mounted **before**
the assistant's summary bubble so the visual order (cards → summary)
matches the streaming order (event → token_chunk).

Each card renders headline + confidence chip + severity + report only.
Per-card next-steps and citations were removed by design — those are
aggregated into the LLM's summary paragraph so the card list stays a
scannable list rather than a stack of long report blocks.
"""

from __future__ import annotations

from typing import Iterable

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Static

from claritymed.core.events import DifferentialCard, DifferentialReady
from claritymed.core.i18n.loader import t

# Confidence bucket → chip color. Deliberately not i18n — colors are a
# design constant.
_BUCKET_COLOR: dict[str, str] = {
    "very_high": "green",
    "high": "cyan",
    "moderate": "yellow",
    "low": "grey58",
}


def _card_text(card: DifferentialCard) -> Text:
    """Assemble one card's Rich Text block.

    Layout::

        ┌ Headline (bold)                          [confidence chip]
        │ Severity tier: <tier>
        │
        │ report paragraph …
    """
    txt = Text()

    # Headline (when present) + confidence chip on one row. Chip is
    # right-aligned in spirit but Textual's Static doesn't do inline
    # right-align on a single Rich Text, so we render
    # ``headline    [bucket]`` with a visible separator and let the
    # terminal wrap on narrow widths. ``headline`` is ``None`` for
    # standard rank 2+ cards (see :func:`directional_headline` — the
    # card title already carries the condition name, so repeating it
    # here would be a bold duplicate). In that case the confidence chip
    # leads the row alone.
    bucket_color = _BUCKET_COLOR.get(card.confidence_bucket, "white")
    if card.headline:
        txt.append(card.headline, style="bold")
        txt.append("   ")
    txt.append(
        f"[{card.confidence_label} · {int(card.probability * 100)}%]",
        style=f"bold {bucket_color}",
    )
    txt.append("\n")
    txt.append(f"Severity tier: {card.severity_tier}", style="italic dim")
    txt.append("\n\n")

    # Report body — plain, no markdown parsing (Textual Markdown widget
    # doesn't inline into Static; the trade-off is acceptable for a
    # 3-5 sentence description).
    txt.append(card.report.strip())
    return txt


class DifferentialCardBubble(Static):
    """One card in the differential list."""

    DEFAULT_CSS = """
    DifferentialCardBubble {
        height: auto;
        margin: 0 1 1 1;
        padding: 1 2;
        border-left: thick $warning;
        background: $surface;
    }
    """

    def __init__(self, card: DifferentialCard) -> None:
        super().__init__()
        self._card = card

    def on_mount(self) -> None:
        self.update(_card_text(self._card))


class DifferentialBanner(Static):
    """Above-cards banner resolved from ``session.banner_key`` via i18n."""

    DEFAULT_CSS = """
    DifferentialBanner {
        height: auto;
        margin: 0 1 0 1;
        padding: 0 2;
        border-left: thick $accent;
        color: $text-muted;
    }
    """

    def __init__(self, banner_key: str, language: str) -> None:
        super().__init__()
        self._banner_key = banner_key
        self._language = language

    def on_mount(self) -> None:
        text = t(self._banner_key, lang=self._language)
        self.update(text)


class DifferentialCards(Vertical):
    """Ordered list of :class:`DifferentialCardBubble`, plus optional banner.

    Mount into the conversation stream before the assistant's summary
    bubble so the visual order matches the semantic order: cards render
    immediately from the sidecar ``differential_ready`` event, and the
    LLM's short summary paragraph fills in above them via subsequent
    ``token_chunk`` events.
    """

    DEFAULT_CSS = """
    DifferentialCards {
        height: auto;
    }
    """

    def __init__(
        self,
        cards: Iterable[DifferentialCard],
        banner_key: str | None,
        language: str,
    ) -> None:
        super().__init__()
        self._cards = list(cards)
        self._banner_key = banner_key
        self._language = language

    def on_mount(self) -> None:
        if self._banner_key:
            self.mount(DifferentialBanner(self._banner_key, self._language))
        for card in self._cards:
            self.mount(DifferentialCardBubble(card))


def render_differential_ready(event: DifferentialReady) -> DifferentialCards:
    """Build the mountable widget tree from the SSE event.

    Empty ``cards`` list is still mounted (banner-only view) so cases
    D / E of the session-flag table render their explanatory banner
    even without cards. Callers get an empty ``DifferentialCards`` if
    there is nothing to show.
    """
    return DifferentialCards(
        cards=event.cards,
        banner_key=event.session.banner_key,
        language=event.language,
    )
