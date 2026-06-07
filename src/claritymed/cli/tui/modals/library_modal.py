"""Library modal — show user_rag docs + system RAG collection toggles.

Two columns: left lists per-user uploaded documents (with PHI / cloud flags);
right shows configurable system_rag collections from ``configs/retrieval.yaml``
with a checkbox per collection bound to ``Account.active_system_rag_collections``.

The v1 implementation focuses on layout + wiring the activation toggle. The
left column reads ``user_rag`` lazily — only the doc ids are shown, no chunk
preview (to keep PHI out of accidental screenshots).
"""

from __future__ import annotations

from typing import Iterable

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Label, ListItem, ListView

from claritymed import config as _cfg


def _system_rag_collections() -> list[dict]:
    """Return the configured system RAG collections, or an empty list."""
    retrieval = _cfg.load_yaml("retrieval.yaml")
    return list((retrieval.get("system_rag") or {}).get("collections") or [])


class LibraryModal(ModalScreen):
    """Modal for managing personal RAG library + activating system collections."""

    DEFAULT_CSS = """
    LibraryModal {
        align: center middle;
    }
    LibraryModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 90;
        height: 30;
    }
    LibraryModal #cols {
        height: 1fr;
    }
    LibraryModal .col {
        width: 1fr;
        border-right: solid $primary;
        padding: 0 1;
    }
    LibraryModal .col.last {
        border-right: none;
    }
    LibraryModal Horizontal#buttons {
        align-horizontal: right;
        height: auto;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("d", "delete_selected", "Delete"),
    ]

    def __init__(
        self,
        documents: Iterable[str] | None = None,
        active_collections: Iterable[str] | None = None,
        on_save=None,
    ) -> None:
        super().__init__()
        self._documents = list(documents or [])
        self._active_collections = set(active_collections or [])
        self._on_save = on_save

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Library — personal docs + system collections")
            with Horizontal(id="cols"):
                with Vertical(classes="col"):
                    yield Label("Your uploads")
                    yield ListView(
                        *(ListItem(Label(doc)) for doc in self._documents),
                        id="docs",
                    )
                with Vertical(classes="col last"):
                    yield Label("System RAG collections")
                    for col in _system_rag_collections():
                        name = col.get("name", "?")
                        desc = col.get("description") or ""
                        active = name in self._active_collections
                        yield Checkbox(
                            f"{name}  —  {desc}",
                            value=active,
                            id=f"col_{name}",
                        )
            with Horizontal(id="buttons"):
                yield Button("Close", id="close")
                yield Button("Save", id="save", variant="primary")

    def action_close(self) -> None:
        self.dismiss(None)

    def action_delete_selected(self) -> None:
        docs = self.query_one("#docs", ListView)
        index = docs.index
        if index is None:
            return
        item = docs.children[index]
        item.remove()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.dismiss(None)
            return
        if event.button.id == "save":
            active = self._collect_active()
            if self._on_save is not None:
                self._on_save(active)
            self.dismiss(active)

    def _collect_active(self) -> list[str]:
        active: list[str] = []
        for box in self.query(Checkbox):
            if box.value and box.id and box.id.startswith("col_"):
                active.append(box.id[len("col_") :])
        return active
