"""Provider picker modal — lists only providers with credentials in the env."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ProviderModal(ModalScreen):
    """Select from providers that have their API keys / env vars configured.

    Dismisses with the chosen provider id string, or None on cancel.
    """

    DEFAULT_CSS = """
    ProviderModal {
        align: center middle;
    }
    ProviderModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 64;
        max-height: 28;
        height: auto;
    }
    ProviderModal Label {
        color: $text-muted;
        margin-bottom: 1;
    }
    ProviderModal VerticalScroll {
        height: auto;
        max-height: 18;
    }
    ProviderModal Button {
        width: 1fr;
        margin-bottom: 0;
        background: $surface-darken-1;
        border: tall $surface-lighten-2;
        color: $text;
    }
    ProviderModal Button:hover,
    ProviderModal Button:focus {
        background: $surface-lighten-1;
        border: tall $primary;
        color: $text;
    }
    ProviderModal Button.selected {
        border: tall $accent;
        color: $accent;
    }
    ProviderModal #cancel {
        margin-top: 1;
        background: $surface;
        border: tall $surface-lighten-1;
        color: $text-muted;
    }
    ProviderModal #cancel:hover,
    ProviderModal #cancel:focus {
        background: $surface-lighten-1;
        border: tall $surface-lighten-2;
        color: $text;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current_provider_id: str | None = None) -> None:
        super().__init__()
        self._current = current_provider_id

    def compose(self) -> ComposeResult:
        from claritymed.stores.models import list_available_providers

        available = list_available_providers()
        with Vertical():
            yield Label("Switch provider  (only configured providers shown)")
            with VerticalScroll():
                if not available:
                    yield Label("No providers available — set API keys in .env")
                for p in available:
                    active = p.id == self._current
                    prefix = "✓ " if active else "  "
                    label = f"{prefix}{p.id}  ·  {p.model}  [{p.kind}]"
                    btn = Button(label, id=f"p_{p.id}")
                    if active:
                        btn.add_class("selected")
                    yield btn
            yield Button("Cancel  (esc)", id="cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id and event.button.id.startswith("p_"):
            self.dismiss(event.button.id[2:])
