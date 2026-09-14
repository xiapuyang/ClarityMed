"""LibraryView — top-level Screen for browsing records + library entries.

v1 surface (per the plan's R-P0-2 scope cut):

* DataTable with columns ``kind | date | title | source | size``.
* ``up``/``down`` navigate, ``enter`` toggles the detail pane (a
  formatted dump of the row's manifest YAML).
* ``d`` triggers a ``delete_record`` proposal via the dispatcher
  (Unit 6) — caller-injected callback so the screen stays UI-only.
* ``e`` opens ``$EDITOR`` on the manifest (deferred to follow-up;
  v1 surfaces a status-bar hint).
* ``q`` / shift+tab returns to the ask screen.

Rules management lives in the headless CLI (Unit 11
``claritymed tool rule-list``/``rule-revoke``) in v1; a future
LibraryView rules subview is the v1.1 design.
"""

from __future__ import annotations

from typing import Callable

import yaml
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from claritymed.stores.manifest_store import ManifestStore


_COLUMNS = ("kind", "date", "title", "source", "size")


class LibraryView(Screen[None]):
    """Browse + delete records and library entries from the TUI."""

    BINDINGS = [
        Binding("q", "back", "Back"),
        Binding("d", "delete", "Delete"),
        Binding("e", "edit", "Edit"),
    ]

    DEFAULT_CSS = """
    LibraryView Vertical { padding: 1; }
    LibraryView DataTable { height: 1fr; }
    LibraryView #detail { background: $boost; padding: 1; }
    """

    def __init__(
        self,
        user_id: str,
        *,
        on_delete: Callable[[str, str], None] | None = None,
    ) -> None:
        super().__init__()
        self._user_id = user_id
        self._on_delete = on_delete
        self._rows: list[tuple[str, str, dict]] = []  # (record_path, scope, manifest)

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield DataTable(id="library-table")
            with Horizontal():
                yield Static("", id="detail")
        yield Footer()

    def on_mount(self) -> None:
        table: DataTable = self.query_one("#library-table", DataTable)
        for col in _COLUMNS:
            table.add_column(col, key=col)
        self._refresh_rows(table)

    def _refresh_rows(self, table: DataTable) -> None:
        table.clear()
        self._rows.clear()
        for scope in ("records", "library"):
            store = ManifestStore(self._user_id, scope)  # type: ignore[arg-type]
            for manifest_path in store.list():
                try:
                    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(raw, dict):
                    continue
                rec_path = f"{raw.get('category', '?')}/{raw.get('slug', '?')}"
                size = sum(int(a.get("size", 0)) for a in raw.get("attachments", []))
                table.add_row(
                    raw.get("kind", "?"),
                    raw.get("date", "") or "",
                    raw.get("title", ""),
                    scope,
                    str(size),
                    key=rec_path,
                )
                self._rows.append((rec_path, scope, raw))

    # --- actions -----------------------------------------------------

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_delete(self) -> None:
        table: DataTable = self.query_one("#library-table", DataTable)
        if not table.cursor_row >= 0 or not self._rows:
            return
        record_path, scope, raw = self._rows[table.cursor_row]
        if scope != "records":
            self.query_one("#detail", Static).update(
                "Only records can be deleted from this view (library uses /rag rm)."
            )
            return
        kind = raw.get("kind", "?")
        if self._on_delete is None:
            self.query_one("#detail", Static).update(
                f"Would delete {record_path!r} (no dispatcher wired)."
            )
            return
        self._on_delete(record_path, kind)
        self._refresh_rows(table)

    def action_edit(self) -> None:
        self.query_one("#detail", Static).update(
            "External editor flow ($EDITOR) coming in v1.1; "
            "for now, edit data/users/<id>/records/<...>/manifest.yaml directly."
        )
