"""Right side panel — shows tool call lifecycle (start → in-progress → done).

Each tool name maps to a single row that updates in place:
  ⟳ tool_name  args…    ← start / in-progress (accent)
  ✓ tool_name — Nms — summary   ← done (success)

Auto-hides when empty. Visible on first event, collapses on reset().
Press F2 to toggle open/closed without clearing content.
"""

from __future__ import annotations

from textual.containers import VerticalScroll
from textual.widgets import Static


class ToolSteps(VerticalScroll):
    """Per-turn tool call log with in-place start → complete updates."""

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

    # Keyed by tool_name; holds the Static that was mounted by push_start so
    # push_complete can update it in place rather than appending a new row.
    _active: dict[str, Static]

    def on_mount(self) -> None:
        self._active = {}

    def reset(self, *, preserve_active: bool = False) -> None:
        """Clear completed rows.

        ``preserve_active=False`` (default) wipes everything — used by
        /clear and /user where the whole session restarts.
        ``preserve_active=True`` keeps in-flight rows (``⟳`` widgets
        still in ``_active``) so per-turn cleanup at the start of a new
        message doesn't kill an OCR job the user enqueued before
        submitting. Without this the row reappears as a stray ``✓``
        when the job completes, because ``push_complete`` can no
        longer find the original row in ``_active``."""
        keep = set(self._active.values()) if preserve_active else set()
        for child in list(self.children):
            if child not in keep:
                child.remove()
        if not preserve_active:
            self._active = {}
        if not self._active:
            self.remove_class("has_events")

    def toggle_collapse(self) -> None:
        """Show/hide the panel without clearing content (F2)."""
        self.toggle_class("user_collapsed")

    def _ensure_visible(self) -> None:
        self.add_class("has_events")

    def push_start(self, tool_name: str, args_preview: str = "") -> Static:
        """Add a ⟳ in-progress row for tool_name; returns the widget."""
        line = f"⟳ {tool_name}"
        if args_preview:
            line += f"  {args_preview}"
        self._ensure_visible()
        item = Static(line, classes="start")
        self.mount(item)
        self._active[tool_name] = item
        self.scroll_end(animate=False)
        return item

    def push_complete(
        self, tool_name: str, duration_ms: int = 0, summary: str = ""
    ) -> Static:
        """Mark tool_name done.

        If push_start was called for the same tool_name this turn, updates
        that row in place (⟳ → ✓) instead of appending a new row.
        """
        bits = [f"✓ {tool_name}"]
        if duration_ms:
            bits.append(f"{duration_ms}ms")
        if summary:
            bits.append(summary)
        text = " — ".join(bits)

        existing = self._active.pop(tool_name, None)
        if existing is not None:
            existing.update(text)
            existing.remove_class("start")
            existing.add_class("complete")
            return existing

        self._ensure_visible()
        item = Static(text, classes="complete")
        self.mount(item)
        self.scroll_end(animate=False)
        return item

    def clear_streaming(self, item: Static | None) -> None:
        """Replace 'streaming…' with 'done' on the llm step when the turn ends.

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
