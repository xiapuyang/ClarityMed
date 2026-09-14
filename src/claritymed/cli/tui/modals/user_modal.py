"""User picker modal — lists all on-disk users with a ``settings.yaml``."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option


class UserModal(ModalScreen):
    """Pick from users that have ``settings.yaml`` on disk.

    Dismisses with the chosen user_id string, or None on cancel. Loads each
    user's ``Account`` so the row can show ``display_name`` and role badge;
    AccountStore reads are cheap (small YAML, mtime-cached) so listing 50+
    users is fine.
    """

    DEFAULT_CSS = """
    UserModal {
        align: center middle;
    }
    UserModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 64;
        height: auto;
    }
    UserModal #title {
        color: $text-muted;
        margin-bottom: 1;
    }
    UserModal #picker {
        height: auto;
        max-height: 18;
    }
    UserModal .hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        Binding("enter", "submit", "Select", priority=True),
    ]

    def __init__(self, current_user_id: str | None = None) -> None:
        super().__init__()
        self._current = current_user_id
        self._user_ids: list[str] = []

    def compose(self) -> ComposeResult:
        from claritymed.stores.account import AccountStore
        from claritymed.stores.paths import list_user_ids

        self._user_ids = list_user_ids()
        with Vertical():
            yield Label("Switch user", id="title")
            if not self._user_ids:
                yield Label("No users on disk yet — run `claritymed init` first.")
            else:
                options: list[Option] = []
                for uid in self._user_ids:
                    mark = "✓ " if uid == self._current else "  "
                    try:
                        acct = AccountStore(uid).load()
                        label = f"{mark}{uid}  ·  {acct.display_name}  [{acct.role}]"
                    except Exception:  # noqa: BLE001 — broken yaml shouldn't hide the user
                        label = f"{mark}{uid}  ·  (settings unreadable)"
                    options.append(Option(label, id=f"u_{uid}"))
                yield OptionList(*options, id="picker")
            yield Label("↑↓ move · enter select · esc cancel", classes="hint")

    def on_mount(self) -> None:
        if not self._user_ids:
            return
        picker = self.query_one("#picker", OptionList)
        if self._current in self._user_ids:
            picker.highlighted = self._user_ids.index(self._current)
        picker.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_submit(self) -> None:
        if not self._user_ids:
            self.dismiss(None)
            return
        picker = self.query_one("#picker", OptionList)
        idx = picker.highlighted
        if idx is None or not (0 <= idx < len(self._user_ids)):
            picker.focus()
            return
        self.dismiss(self._user_ids[idx])
