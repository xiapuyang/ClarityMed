"""Provider picker modal — lists only providers with credentials in the env."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option


class ProviderModal(ModalScreen):
    """Select from providers that have their API keys / env vars configured.

    Dismisses with the chosen provider id string, or None on cancel.
    Uses OptionList so ↑/↓ and Enter work the same as the single-select
    QuestionModal.
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
        height: auto;
    }
    ProviderModal #title {
        color: $text-muted;
        margin-bottom: 1;
    }
    ProviderModal #picker {
        height: auto;
        max-height: 18;
    }
    ProviderModal .hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        Binding("enter", "submit", "Select", priority=True),
    ]

    def __init__(self, current_provider_id: str | None = None) -> None:
        super().__init__()
        self._current = current_provider_id
        self._providers: list = []

    def compose(self) -> ComposeResult:
        from claritymed.stores.models import list_available_providers

        self._providers = list_available_providers()
        with Vertical():
            yield Label(
                "Switch provider  (only configured providers shown)", id="title"
            )
            if not self._providers:
                yield Label("No providers available — set API keys in .env")
            else:
                options = [
                    Option(
                        f"{'✓ ' if p.id == self._current else '  '}{p.id}  ·  {p.model}  [{p.kind}]",
                        id=f"p_{p.id}",
                    )
                    for p in self._providers
                ]
                picker = OptionList(*options, id="picker")
                yield picker
            yield Label("↑↓ move · enter select · esc cancel", classes="hint")

    def on_mount(self) -> None:
        if not self._providers:
            return
        picker = self.query_one("#picker", OptionList)
        current_ids = [p.id for p in self._providers]
        if self._current in current_ids:
            picker.highlighted = current_ids.index(self._current)
        picker.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_submit(self) -> None:
        if not self._providers:
            self.dismiss(None)
            return
        picker = self.query_one("#picker", OptionList)
        idx = picker.highlighted
        if idx is None or not (0 <= idx < len(self._providers)):
            picker.focus()
            return
        self.dismiss(self._providers[idx].id)
