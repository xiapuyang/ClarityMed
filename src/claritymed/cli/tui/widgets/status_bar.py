"""Top status bar — shows user / mode / language / provider / request id.

Each reactive corresponds to a single piece of session state the user wants
to see at a glance. The bar itself does no business logic — the app owns the
state and the widget re-renders when a reactive changes.
"""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static

CONFIDENCE_HIGH = 0.9
CONFIDENCE_MEDIUM = 0.7

ModeName = str  # Literal['ingest','ask','rag'] — kept loose for reactive cleanup


def _confidence_band(value: float | None) -> str:
    """Map a raw confidence to ``high`` / ``medium`` / ``low`` / ``-``."""
    if value is None:
        return "-"
    if value >= CONFIDENCE_HIGH:
        return "high"
    if value >= CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def _fmt_ctx(chars: int) -> str:
    """Compact display: 0 / 980 / 12.4k / 1.2M."""
    if chars < 1000:
        return str(chars)
    if chars < 1_000_000:
        return f"{chars / 1000:.1f}k"
    return f"{chars / 1_000_000:.1f}M"


class StatusBar(Static):
    """Reactive status line. Updates whenever any of its fields change."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }
    StatusBar.flash {
        background: $warning;
    }
    """

    user_id: reactive[str] = reactive("default")
    mode: reactive[ModeName] = reactive("ask")
    language: reactive[str] = reactive("en")
    provider_id: reactive[str] = reactive("?")
    provider_kind: reactive[str] = reactive("?")
    request_id: reactive[str] = reactive("-")
    confidence: reactive[float | None] = reactive(None)
    routing_flash: reactive[str] = reactive("")
    context_chars: reactive[int] = reactive(0)

    def render(self) -> str:
        band = _confidence_band(self.confidence)
        flash = f" → {self.routing_flash}" if self.routing_flash else ""
        return (
            f"user={self.user_id}  "
            f"mode={self.mode}{flash}  "
            f"lang={self.language}  "
            f"model={self.provider_id}({self.provider_kind})  "
            f"ctx={_fmt_ctx(self.context_chars)}  "
            f"req={self.request_id[-8:]}  "
            f"conf={band}"
        )

    def watch_routing_flash(self, value: str) -> None:
        """Toggle the flash class when ``routing_flash`` is non-empty."""
        if value:
            self.add_class("flash")
        else:
            self.remove_class("flash")
