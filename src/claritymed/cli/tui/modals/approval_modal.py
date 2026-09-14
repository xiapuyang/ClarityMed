"""Cloud-provider switch + PHI policy approval.

Dismisses with ``True`` when the user accepts the cloud switch, ``False`` when
they reject. The caller handles the "Run with local instead?" follow-up.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ApprovalModal(ModalScreen):
    """Yes / No approval for sending a query through a cloud provider."""

    DEFAULT_CSS = """
    ApprovalModal {
        align: center middle;
    }
    ApprovalModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 70;
        height: auto;
    }
    ApprovalModal Horizontal#buttons {
        align-horizontal: right;
        height: auto;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "deny", "Deny"),
        ("y", "accept", "Yes"),
        ("n", "deny", "No"),
    ]

    def __init__(
        self,
        provider_id: str,
        model: str,
        phi_policy: str = "filter chunks marked is_phi",
    ) -> None:
        super().__init__()
        self._provider_id = provider_id
        self._model = model
        self._phi_policy = phi_policy

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Cloud provider switch")
            yield Label(f"Provider: {self._provider_id}")
            yield Label(f"Model:    {self._model}")
            yield Label(f"PHI:      {self._phi_policy}")
            yield Label("Proceed?")
            with Horizontal(id="buttons"):
                yield Button("No  (n)", id="deny")
                yield Button("Yes (y)", id="accept", variant="primary")

    def action_accept(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "accept")
