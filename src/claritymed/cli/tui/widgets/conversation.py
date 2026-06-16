"""Conversation pane — scrolling list of turn bubbles.

Each turn is one ``TurnBubble``. The active assistant turn is mutable: token
chunks from the streaming service get appended to ``streaming_text``; when
the service yields ``Done`` the bubble is locked. Cancelled turns keep their
partial text plus a ``⊘ cancelled`` marker.
"""

from __future__ import annotations

from typing import Literal

from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Markdown, Static

Role = Literal["user", "assistant", "system"]


class TurnBubble(Static):
    """One conversation turn. Static-derived so tests can read ``renderable``."""

    DEFAULT_CSS = """
    TurnBubble {
        height: auto;
        margin: 0 1 1 1;
        padding: 0 1;
    }
    TurnBubble.user {
        color: $text;
        border-left: thick $accent;
    }
    TurnBubble.assistant {
        color: $text;
        border-left: thick $success;
    }
    TurnBubble.system {
        color: $text-muted;
        border-left: thick $primary;
    }
    TurnBubble.cancelled {
        color: $text-disabled;
    }
    TurnBubble.error {
        color: $error;
        border-left: thick $error;
    }
    """

    streaming_text: reactive[str] = reactive("")
    cancelled: reactive[bool] = reactive(False)
    is_error: reactive[bool] = reactive(False)

    def __init__(
        self,
        role: Role,
        text: str = "",
        *,
        error: bool = False,
        prefix_override: str | None = None,
    ) -> None:
        super().__init__()
        # Assigned before ``streaming_text`` so the reactive watcher (which
        # fires during the assignment below) can already read it.
        self._prefix_override = prefix_override
        self._role: Role = role
        self.streaming_text = text
        self.is_error = error

    def on_mount(self) -> None:
        self.add_class(self._role)
        if self.is_error:
            self.add_class("error")
        self._refresh_content()

    @property
    def role(self) -> Role:
        return self._role

    def append(self, chunk: str) -> None:
        """Append a streaming token (assistant turns only)."""
        self.streaming_text = self.streaming_text + chunk

    def mark_cancelled(self) -> None:
        self.cancelled = True

    def watch_streaming_text(self, _value: str) -> None:
        self._refresh_content()

    def watch_cancelled(self, value: bool) -> None:
        if value:
            self.add_class("cancelled")
        else:
            self.remove_class("cancelled")
        self._refresh_content()

    def _refresh_content(self) -> None:
        if self._prefix_override is not None:
            prefix = self._prefix_override
        else:
            prefix = {"user": "› ", "assistant": "‹ ", "system": "ⓘ "}[self._role]
        suffix = "\n⊘ cancelled" if self.cancelled else ""
        self.update(f"{prefix}{self.streaming_text}{suffix}")


class AssistantMarkdown(Markdown):
    """Final rendered Markdown bubble for an assistant turn."""

    DEFAULT_CSS = """
    AssistantMarkdown {
        margin: 0 1 1 1;
        padding: 0 1;
        border-left: thick $success;
        background: $surface;
    }
    """


class Conversation(VerticalScroll):
    """Scrolling list of turns plus the empty-state placeholder."""

    DEFAULT_CSS = """
    Conversation {
        width: 2fr;
        background: $surface;
    }
    Conversation > .empty {
        color: $text-muted;
        padding: 2;
    }
    """

    _active_assistant: TurnBubble | None = None

    def show_empty_state(self, hint: str) -> None:
        empty = Static(hint, classes="empty")
        self.mount(empty)

    def clear_empty_state(self) -> None:
        for child in list(self.children):
            if "empty" in child.classes:
                child.remove()

    def add_user_turn(self, text: str) -> TurnBubble:
        self.clear_empty_state()
        bubble = TurnBubble("user", text)
        self.mount(bubble)
        self.scroll_end(animate=False)
        return bubble

    def add_system_turn(self, text: str) -> TurnBubble:
        bubble = TurnBubble("system", text)
        self.mount(bubble)
        self.scroll_end(animate=False)
        return bubble

    def add_error_turn(self, text: str) -> TurnBubble:
        bubble = TurnBubble("assistant", text, error=True)
        self.mount(bubble)
        self._active_assistant = None
        self.scroll_end(animate=False)
        return bubble

    def add_command_error_turn(self, text: str) -> TurnBubble:
        """Render a slash-command error inline (e.g. ``⏺ Unknown command: /foo``).

        Used when input validation rejects a typed command — we keep the
        line in transcript history rather than a transient toast so the
        user can see *what* they mistyped after the fact.
        """
        bubble = TurnBubble("system", text, error=True, prefix_override="⏺ ")
        self.mount(bubble)
        self.scroll_end(animate=False)
        return bubble

    def clear_active_streaming_text(self) -> None:
        """Erase any text streamed into the active assistant bubble."""
        if self._active_assistant is not None:
            self._active_assistant.streaming_text = ""

    def drop_active_streaming_bubble(self) -> None:
        """Remove the active streaming bubble from the DOM and deregister it.

        Called by TextualPromptChannel before showing any modal so that the
        modal result (系统 turn) mounts after it, and the LLM's post-tool
        answer starts a fresh bubble after the system turn. This gives the
        ordering: [modal result] → [answer], regardless of whether the
        prompt channel was invoked directly by the LLM or from inside a tool
        body. Pre-tool streaming text is discarded — the LLM generates a
        proper post-tool answer once the modal resolves.
        """
        if self._active_assistant is not None:
            self._active_assistant.remove()
            self._active_assistant = None

    def start_assistant_turn(self) -> TurnBubble:
        bubble = TurnBubble("assistant", "")
        self.mount(bubble)
        self._active_assistant = bubble
        self.scroll_end(animate=False)
        return bubble

    def append_to_active(self, chunk: str) -> None:
        if self._active_assistant is None:
            self.start_assistant_turn()
        assert self._active_assistant is not None
        self._active_assistant.append(chunk)
        self.scroll_end(animate=False)

    def finalize_active(self, markdown_text: str | None = None) -> TurnBubble | None:
        """Lock the active assistant turn.

        When ``markdown_text`` is provided, the streaming Static bubble is
        replaced with a fully rendered ``AssistantMarkdown`` widget so the
        LLM's markdown (lists, headings, bold) shows properly. When it is
        ``None`` (e.g. system / ingest finalize), the streaming bubble is
        left as-is.
        """
        finalized = self._active_assistant
        self._active_assistant = None
        if finalized is not None and markdown_text:
            try:
                md = AssistantMarkdown(markdown_text)
                self.mount(md, before=finalized)
                finalized.remove()
                self.scroll_end(animate=False)
            except Exception:  # noqa: BLE001 — fall through to raw bubble
                pass
        elif finalized is not None and markdown_text is not None:
            # Empty text — just remove the streaming bubble, mount nothing.
            try:
                finalized.remove()
            except Exception:  # noqa: BLE001
                pass
        return finalized

    def cancel_active(self) -> TurnBubble | None:
        if self._active_assistant is not None:
            self._active_assistant.mark_cancelled()
        finalized = self._active_assistant
        self._active_assistant = None
        return finalized
