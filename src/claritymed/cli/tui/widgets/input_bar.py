"""Bottom input bar — single ``Input`` plus a mode-aware placeholder.

The bar emits ``InputBar.Submitted(value)`` when the user presses Enter on a
non-empty value. The app handles routing and clearing.
"""

from __future__ import annotations

from textual.containers import Container
from textual.message import Message
from textual.widgets import Input


class InputBar(Container):
    """Container around a single Input. Exposes a typed submit Message."""

    DEFAULT_CSS = """
    InputBar {
        height: 4;
        padding: 0 1 1 1;
        background: $surface;
        border-top: solid $primary;
    }
    InputBar Input {
        background: $boost;
    }
    """

    class Submitted(Message):
        """Emitted when the user submits a non-empty input."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def compose(self):
        yield Input(placeholder="What would you like to know?", id="input")

    def set_placeholder(self, text: str) -> None:
        inp = self.query_one("#input", Input)
        inp.placeholder = text

    def focus_input(self) -> None:
        self.query_one("#input", Input).focus()

    def clear(self) -> None:
        self.query_one("#input", Input).value = ""

    def value(self) -> str:
        return self.query_one("#input", Input).value

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        self.post_message(self.Submitted(text))
