"""Language picker modal — lists ``i18n.supported_langs`` from ``app.yaml``."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

_LANG_LABELS: dict[str, str] = {"en": "English", "zh": "中文"}


class LanguageModal(ModalScreen):
    """Pick a reply language from ``i18n.supported_langs``.

    Dismisses with the chosen lang code (``"en"`` / ``"zh"``) or ``None`` on
    cancel. Mirrors :class:`ProviderModal` so ↑/↓ + Enter behave identically.
    """

    DEFAULT_CSS = """
    LanguageModal {
        align: center middle;
    }
    LanguageModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 48;
        height: auto;
    }
    LanguageModal #title {
        color: $text-muted;
        margin-bottom: 1;
    }
    LanguageModal #picker {
        height: auto;
        max-height: 10;
    }
    LanguageModal .hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        Binding("enter", "submit", "Select", priority=True),
    ]

    def __init__(self, current_lang: str | None = None) -> None:
        super().__init__()
        self._current = (current_lang or "").lower()
        self._langs: list[str] = []

    def compose(self) -> ComposeResult:
        from claritymed import config as _cfg

        self._langs = list(_cfg.supported_langs())
        with Vertical():
            yield Label("Switch reply language", id="title")
            if not self._langs:
                yield Label("No languages configured in app.yaml")
            else:
                options = [
                    Option(
                        f"{'✓ ' if code == self._current else '  '}"
                        f"{code}  ·  {_LANG_LABELS.get(code, code)}",
                        id=f"l_{code}",
                    )
                    for code in self._langs
                ]
                yield OptionList(*options, id="picker")
            yield Label("↑↓ move · enter select · esc cancel", classes="hint")

    def on_mount(self) -> None:
        if not self._langs:
            return
        picker = self.query_one("#picker", OptionList)
        if self._current in self._langs:
            picker.highlighted = self._langs.index(self._current)
        picker.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_submit(self) -> None:
        if not self._langs:
            self.dismiss(None)
            return
        picker = self.query_one("#picker", OptionList)
        idx = picker.highlighted
        if idx is None or not (0 <= idx < len(self._langs)):
            picker.focus()
            return
        self.dismiss(self._langs[idx])
