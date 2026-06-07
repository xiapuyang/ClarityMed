"""Mode confirmation modal — surfaced when the router returns ambiguous.

The modal dismisses with the chosen mode string (``"ingest"`` / ``"ask"`` /
``"rag"``), or ``None`` on cancel — the app treats cancel as ``Cancelled``.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ModeModal(ModalScreen):
    """Ask the user which mode to use when auto routing was low-confidence."""

    DEFAULT_CSS = """
    ModeModal {
        align: center middle;
    }
    ModeModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 60;
        height: auto;
    }
    ModeModal Horizontal#buttons {
        align-horizontal: center;
        height: auto;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        detected_mode: str | None = None,
        confidence: float | None = None,
    ) -> None:
        super().__init__()
        self._detected_mode = detected_mode
        self._confidence = confidence

    def compose(self) -> ComposeResult:
        if self._detected_mode and self._confidence is not None:
            head = (
                f"Detected: {self._detected_mode} "
                f"(confidence {self._confidence:.2f}). Use which mode?"
            )
        else:
            head = "Which mode should this run in?"
        with Vertical():
            yield Label(head)
            with Horizontal(id="buttons"):
                yield Button("Ingest", id="ingest")
                yield Button("Ask", id="ask", variant="primary")
                yield Button("Rag", id="rag")
                yield Button("Cancel", id="cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id in ("ingest", "ask", "rag"):
            self.dismiss(event.button.id)
        else:
            self.dismiss(None)
