"""Help modal — keyboard shortcuts and slash command reference.

Opens via the ``/help`` slash command or by pressing ``?`` on an empty input
bar. The earlier inline ``HELP_TEXT`` rendered into a system bubble suffered
from Rich-markup collisions (``[query]``, ``[code]``, ``[path]`` were parsed
as markup tags — ``[code]`` in particular flipped on a code-style background).
Rendering inside a modal with markup disabled gives us aligned columns and a
predictable background.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, Static

SHORTCUT_ROWS: tuple[tuple[str, str], ...] = (
    ("?", "Show this help"),
    ("/", "Slash command menu"),
    ("F2", "Toggle steps panel"),
    ("F3", "Switch LLM provider"),
    ("Ctrl+V", "Paste clipboard image / file"),
    ("Esc", "Cancel streaming response"),
    ("Ctrl+C", "Quit"),
)

COMMAND_ROWS: tuple[tuple[str, str], ...] = (
    ("/upload <path>", "Upload a file into your personal RAG library"),
    ("/library <query>", "List collections (no arg) or run full RAG retrieval"),
    ("/user <id>", "Switch active user (admin only; no arg opens picker)"),
    ("/provider <id>", "Switch LLM provider (no arg opens picker, F3)"),
    ("/lang <code>", "Switch reply language en/zh (no arg opens picker)"),
    ("/clear", "Start a new chat session (history stays on disk)"),
    ("/help", "Show this help"),
    ("/quit", "Exit the TUI"),
)


def _format_rows(rows: tuple[tuple[str, str], ...]) -> str:
    """Render ``(key, description)`` pairs as a 2-column aligned block."""
    width = max(len(key) for key, _ in rows)
    return "\n".join(f"  {key.ljust(width)}   {desc}" for key, desc in rows)


class HelpModal(ModalScreen):
    """Two-section keyboard + slash-command reference.

    ``markup=False`` on every ``Static`` so command argument hints like
    ``<path>`` / ``<id>`` render literally instead of being eaten by Rich.
    """

    DEFAULT_CSS = """
    HelpModal {
        align: center middle;
    }
    HelpModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 72;
        height: auto;
        max-height: 90%;
    }
    HelpModal #title {
        color: $text-muted;
        margin-bottom: 1;
    }
    HelpModal .section {
        color: $accent;
        margin-top: 1;
    }
    HelpModal .rows {
        color: $text;
    }
    HelpModal .hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("question_mark", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("ClarityMed — Help", id="title")
            yield Static("Keyboard", classes="section", markup=False)
            yield Static(_format_rows(SHORTCUT_ROWS), classes="rows", markup=False)
            yield Static("Slash commands", classes="section", markup=False)
            yield Static(_format_rows(COMMAND_ROWS), classes="rows", markup=False)
            yield Label("esc to close", classes="hint")

    def action_close(self) -> None:
        self.dismiss(None)
