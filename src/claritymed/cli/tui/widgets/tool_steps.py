"""Right side panel — shows the tool start / complete log for the current turn.

Auto-hides when empty (the common case for ask in Phase 1 since retrieval is
still stubbed). Becomes visible on the first event from the service and
collapses again on ``reset()``.

Press F2 to toggle the panel open/closed without resetting its content.
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
    ToolSteps.user_collapsed {
        display: none;
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

    def toggle_collapse(self) -> None:
        """Show/hide the panel without clearing content (F2)."""
        self.toggle_class("user_collapsed")

    def _ensure_visible(self) -> None:
        self.add_class("has_events")

    def push_start(self, tool_name: str, args_preview: str = "") -> Static:
        line = f"→ {tool_name}"
        if args_preview:
            line += f" ({args_preview})"
        self._ensure_visible()
        item = Static(line, classes="start")
        self.mount(item)
        self.scroll_end(animate=False)
        return item

    def push_complete(
        self, tool_name: str, duration_ms: int = 0, summary: str = ""
    ) -> Static:
        bits = [f"✓ {tool_name}"]
        if duration_ms:
            bits.append(f"{duration_ms}ms")
        if summary:
            bits.append(summary)
        self._ensure_visible()
        item = Static(" — ".join(bits), classes="complete")
        self.mount(item)
        self.scroll_end(animate=False)
        return item

    def clear_streaming(self, item: Static | None) -> None:
        """Replace 'streaming…' with 'done' on an llm-first-token step.

        Called from the stream worker's finally block so the label always
        resolves regardless of whether the turn ended via Done, Cancelled,
        Error, or an unhandled exception.
        """
        if item is None:
            return
        text = str(item.renderable)
        if "streaming…" in text:
            item.update(text.replace("streaming…", "done"))

    def push_filtered(self, total: int, kept: int, filtered_phi: int) -> None:
        line = f"ⓘ {kept}/{total} sources kept ({filtered_phi} filtered for PHI policy)"
        self._ensure_visible()
        item = Static(line, classes="filtered")
        self.mount(item)
        self.scroll_end(animate=False)
