"""Right side panel — shows the tool start / complete log for the current turn.

Auto-hides when empty (the common case for ask in Phase 1 since retrieval is
still stubbed). Becomes visible on the first event from the service and
collapses again on ``reset()``.
"""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static


class ToolSteps(VerticalScroll):
    """Append-only log of tool start / completed events."""

    DEFAULT_CSS = """
    ToolSteps {
        width: 1fr;
        max-width: 40;
        background: $surface;
        padding: 1;
        border-left: solid $primary;
        display: none;
    }
    ToolSteps.has_events {
        display: block;
    }
    ToolSteps > Static {
        height: auto;
        color: $text-muted;
    }
    ToolSteps > Static.complete {
        color: $success;
    }
    ToolSteps > Static.start {
        color: $accent;
    }
    ToolSteps > Static.filtered {
        color: $warning;
    }
    """

    def reset(self) -> None:
        for child in list(self.children):
            child.remove()
        self.remove_class("has_events")

    def _ensure_visible(self) -> None:
        self.add_class("has_events")

    def push_start(self, tool_name: str, args_preview: str = "") -> None:
        line = f"→ {tool_name}"
        if args_preview:
            line += f" ({args_preview})"
        self._ensure_visible()
        item = Static(line, classes="start")
        self.mount(item)
        self.scroll_end(animate=False)

    def push_complete(
        self, tool_name: str, duration_ms: int = 0, summary: str = ""
    ) -> None:
        bits = [f"✓ {tool_name}"]
        if duration_ms:
            bits.append(f"{duration_ms}ms")
        if summary:
            bits.append(summary)
        self._ensure_visible()
        item = Static(" — ".join(bits), classes="complete")
        self.mount(item)
        self.scroll_end(animate=False)

    def push_filtered(self, total: int, kept: int, filtered_phi: int) -> None:
        line = f"ⓘ {kept}/{total} sources kept ({filtered_phi} filtered for PHI policy)"
        self._ensure_visible()
        item = Static(line, classes="filtered")
        self.mount(item)
        self.scroll_end(animate=False)
