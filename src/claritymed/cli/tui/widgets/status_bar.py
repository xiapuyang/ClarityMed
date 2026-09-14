"""Top status bar — shows user / mode / language / provider / request id.

Each reactive corresponds to a single piece of session state the user wants
to see at a glance. The bar itself does no business logic — the app owns the
state and the widget re-renders when a reactive changes.
"""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static

ModeName = str  # Literal['ask'] — kept loose for reactive cleanup


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
    """

    user_id: reactive[str] = reactive("default")
    mode: reactive[ModeName] = reactive("ask")
    language: reactive[str] = reactive("en")
    provider_id: reactive[str] = reactive("?")
    provider_kind: reactive[str] = reactive("?")
    request_id: reactive[str] = reactive("-")
    context_chars: reactive[int] = reactive(0)
    # Empty string when the vision feature is healthy (or absent). Any
    # non-empty string is rendered verbatim as a trailing chip
    # (``vision: ⚠ <reason>``) so a pasted CT can't fail silently —
    # the user sees the warning the moment they look at the bar.
    vision_status: reactive[str] = reactive("")

    def render(self) -> str:
        base = (
            f"user={self.user_id}  "
            f"lang={self.language}  "
            f"model={self.provider_id}({self.provider_kind})  "
            f"ctx={_fmt_ctx(self.context_chars)}  "
            f"req={self.request_id[-8:]}"
        )
        if self.vision_status:
            return f"{base}  vision:⚠ {self.vision_status}"
        return base
