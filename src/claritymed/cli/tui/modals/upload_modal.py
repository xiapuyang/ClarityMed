"""Upload modal — pick a file and tag it record-vs-reference.

The modal dismisses with a 2-tuple ``(text_to_ingest, public: bool)`` when the
user confirms. ``public=False`` means PHI scrubbing applies and the chunk
stays cloud-blocked; ``public=True`` is for published / reference material.
Dismisses with ``None`` on cancel — the caller treats that as a no-op.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static


class UploadModal(ModalScreen):
    """Path input + record/reference radio + Confirm/Cancel."""

    DEFAULT_CSS = """
    UploadModal {
        align: center middle;
    }
    UploadModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 70;
        height: auto;
    }
    UploadModal Input.error {
        border: tall $error;
    }
    UploadModal #error {
        color: $error;
        height: auto;
    }
    UploadModal Horizontal#buttons {
        align-horizontal: right;
        height: auto;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, initial_path: str = "") -> None:
        super().__init__()
        self._initial_path = initial_path

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Upload a file")
            yield Input(
                value=self._initial_path,
                placeholder="/path/to/file.pdf",
                id="path",
            )
            yield Static("", id="error")
            yield Label("Document kind:")
            with RadioSet(id="kind"):
                yield RadioButton(
                    "My medical record (PHI — scrubbed, local only)",
                    value=True,
                    id="record",
                )
                yield RadioButton(
                    "Public reference (paper, guideline — no scrub)",
                    id="reference",
                )
            with Horizontal(id="buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Upload", id="confirm", variant="primary")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id == "confirm":
            self._try_confirm()

    def _try_confirm(self) -> None:
        path_input = self.query_one("#path", Input)
        error = self.query_one("#error", Static)
        raw = path_input.value.strip()
        if not raw:
            self._mark_error(path_input, error, "Path is required.")
            return
        path = Path(raw)
        if not path.exists() or not path.is_file():
            self._mark_error(path_input, error, "File not found.")
            return
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            self._mark_error(path_input, error, f"Read failed: {exc}")
            return
        public = self._is_reference()
        self.dismiss((text, public))

    def _is_reference(self) -> bool:
        radio_set = self.query_one(RadioSet)
        pressed = radio_set.pressed_button
        return pressed is not None and pressed.id == "reference"

    @staticmethod
    def _mark_error(input_w: Input, error: Static, msg: str) -> None:
        input_w.add_class("error")
        error.update(msg)
