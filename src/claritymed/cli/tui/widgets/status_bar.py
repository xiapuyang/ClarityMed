"""Top status bar — shows user / mode / language / provider / request id.

Each reactive corresponds to a single piece of session state the user wants
to see at a glance. The bar itself does no business logic — the app owns the
state and the widget re-renders when a reactive changes.
"""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static

ModeName = str  # Literal['ingest','ask','rag'] — kept loose for reactive cleanup


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
    routing_flash: reactive[str] = reactive("")
    context_chars: reactive[int] = reactive(0)

    def render(self) -> str:
        return (
            f"user={self.user_id}  "
            f"lang={self.language}  "
            f"model={self.provider_id}({self.provider_kind})  "
            f"ctx={_fmt_ctx(self.context_chars)}  "
            f"req={self.request_id[-8:]}"
        )

    def watch_routing_flash(self, value: str) -> None:
        """Toggle the flash class when ``routing_flash`` is non-empty."""
        if value:
            self.add_class("flash")
        else:
            self.remove_class("flash")
