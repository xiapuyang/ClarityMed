"""Library modal — inspect every RAG collection + run the live retrieval pipeline.

Empty query (default) lists every collection that ``HybridRetriever`` would
search: the configured system collections from ``configs/retrieval.yaml`` plus
the per-user ``user_rag`` collection when present. ``user_rag`` rows are
prefixed ``[USER]`` so it is clear which entries are personal.

A non-empty query goes through the **full** RAG pipeline used by ``/ask``:
term expansion → translation → embedding → hybrid search across every active
collection → reranking → parent expansion. The expanded / translated query is
surfaced alongside per-stage timings so the user can see what the retriever
actually did. ``[USER]`` is again used to flag chunks sourced from
``user_rag``.

Requires an injected ``RagStrategy`` — the same one ``AskService`` uses —
plus the active ``user_id`` and ``language``. When the strategy is ``None``
(retrieval disabled), the list view still works but search is blocked with a
banner.
"""

from __future__ import annotations

import logging

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from claritymed import config as _cfg
from claritymed.core.rag.strategies.base import RagStrategy, RetrievalContext

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 200
_USER_PREFIX = "user_rag_"
_USER_TAG = "[USER]"
_SYS_TAG = "[SYS]"

# Column widths for the list-mode table. Sized so the header underline is
# the same length as the widest first-line row in practice; tweak together
# with ``_topic_indent`` if any column changes.
_COL_KIND = 6
_COL_NAME = 18
_COL_CHUNKS = 10
_COL_NOTES = 16
_GAP = "  "
_TOPIC_INDENT = " " * (
    _COL_KIND
    + len(_GAP)
    + _COL_NAME
    + len(_GAP)
    + _COL_CHUNKS
    + len(_GAP)
    + _COL_NOTES
    + len(_GAP)
)


def _system_rag_collections() -> list[dict]:
    """Return the configured system RAG collections from ``retrieval.yaml``."""
    retrieval = _cfg.load_yaml("retrieval.yaml")
    return list((retrieval.get("system_rag") or {}).get("collections") or [])


def _one_line(text: str | None, limit: int = _PREVIEW_CHARS) -> str:
    """Collapse newlines + truncate so each row fits two visual lines."""
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _label_for_collection(collection_name: str) -> str:
    """``user_rag_<uid>`` → ``[USER]``; anything else → ``[SYS] <name>``."""
    if collection_name.startswith(_USER_PREFIX):
        return _USER_TAG
    return f"{_SYS_TAG} {collection_name}"


def _list_header() -> str:
    """Column header for the list mode table."""
    return (
        f"{'KIND':<{_COL_KIND}}{_GAP}"
        f"{'NAME':<{_COL_NAME}}{_GAP}"
        f"{'CHUNKS':>{_COL_CHUNKS}}{_GAP}"
        f"{'NOTES':<{_COL_NOTES}}{_GAP}"
        f"TOPICS"
    )


def _row_first_line(
    tag: str, name: str, chunks_str: str, notes: str, first_topic: str
) -> str:
    """Render the first visual line of a collection row (all 5 columns)."""
    return (
        f"{tag:<{_COL_KIND}}{_GAP}"
        f"{name:<{_COL_NAME}}{_GAP}"
        f"{chunks_str:>{_COL_CHUNKS}}{_GAP}"
        f"{notes:<{_COL_NOTES}}{_GAP}"
        f"{first_topic}"
    )


def _topic_continuation(topic: str) -> str:
    """Continuation line: blank columns 1-4, topic indented to column 5."""
    return f"{_TOPIC_INDENT}{topic}"


class LibraryModal(ModalScreen):
    """Browse every RAG collection + run the live retrieval pipeline."""

    DEFAULT_CSS = """
    LibraryModal {
        align: center middle;
    }
    LibraryModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 110;
        height: 34;
    }
    LibraryModal #title {
        color: $text-muted;
        margin-bottom: 1;
    }
    LibraryModal #query {
        margin-bottom: 1;
    }
    LibraryModal #header {
        color: $text-muted;
        text-style: bold;
        height: 1;
    }
    LibraryModal #results {
        height: 1fr;
    }
    LibraryModal #status {
        color: $text-muted;
        height: auto;
        margin-top: 1;
    }
    LibraryModal Horizontal#buttons {
        align-horizontal: right;
        height: auto;
        margin-top: 1;
    }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
    ]

    def __init__(
        self,
        *,
        user_id: str,
        language: str,
        strategy: RagStrategy | None,
        initial_query: str = "",
    ) -> None:
        super().__init__()
        self._user_id = user_id
        self._language = language
        self._strategy = strategy
        self._initial_query = initial_query

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Library — RAG collections + live retrieval", id="title")
            yield Input(
                value=self._initial_query,
                placeholder="Search…",
                id="query",
            )
            yield Static("", id="header")
            yield ListView(id="results")
            yield Static("", id="status")
            with Horizontal(id="buttons"):
                yield Button("Close", id="close")

    async def on_mount(self) -> None:
        if self._initial_query:
            await self._reload(self._initial_query)
        else:
            await self._reload("")

    def action_close(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "close":
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "query":
            return
        self.run_worker(self._reload(event.value.strip()), exclusive=True)

    async def _reload(self, query: str) -> None:
        listview = self.query_one("#results", ListView)
        status = self.query_one("#status", Static)
        header = self.query_one("#header", Static)
        await listview.clear()
        status.update("Loading…")
        try:
            if query:
                header.update("")
                await self._render_search(listview, status, query)
            else:
                header.update(_list_header())
                await self._render_list(listview, status)
        except Exception as exc:  # noqa: BLE001
            logger.exception("library modal load failed (query=%r)", query)
            status.update(f"Error: {exc}")

    async def _render_list(self, listview: ListView, status: Static) -> None:
        """Render every collection as a 5-column row with one topic per line.

        Columns: KIND, NAME, CHUNKS, NOTES, TOPICS. Continuation lines for
        a multi-topic row blank-pad the first 4 columns and indent the
        topic under the TOPICS header.
        """
        system_cols = _system_rag_collections()
        system_counts = await self._system_collection_counts(
            [c.get("name", "") for c in system_cols if c.get("name")]
        )
        user_chunk_count = await self._count_user_rag_chunks()

        for col in system_cols:
            name = col.get("name", "?")
            language = col.get("language", "?")
            tier = col.get("authority_tier")
            tier_str = f"tier {tier}" if tier is not None else "tier ?"
            notes = f"{language} · {tier_str}"
            count = system_counts.get(name)
            chunks_str = str(count) if count is not None else "?"
            topics = col.get("topics") or []
            first_topic = topics[0] if topics else "(no topics declared)"
            row_lines = [
                _row_first_line(_SYS_TAG, name, chunks_str, notes, first_topic)
            ]
            for topic in topics[1:]:
                row_lines.append(_topic_continuation(topic))
            await listview.append(ListItem(Label("\n".join(row_lines))))

        if user_chunk_count is None:
            user_chunks_str = "?"
            user_topic = "(not initialised — /upload to populate)"
        else:
            user_chunks_str = str(user_chunk_count)
            user_topic = "(your uploads)"
        user_notes = f"user '{self._user_id}'"
        user_row = _row_first_line(
            _USER_TAG, "user_rag", user_chunks_str, user_notes, user_topic
        )
        await listview.append(ListItem(Label(user_row)))

        rag_state = (
            "ready" if self._strategy is not None else "disabled (rag.enabled=false)"
        )
        status.update(f"{len(system_cols)} system + 1 user · retrieval: {rag_state}")

    async def _render_search(
        self, listview: ListView, status: Static, query: str
    ) -> None:
        """Run the same strategy AskService uses and surface chunks + trace."""
        if self._strategy is None:
            status.update(
                "RAG is disabled (configs/retrieval.yaml: rag.enabled=false) — "
                "cannot search."
            )
            return
        ctx = RetrievalContext(
            query=query,
            user_id=self._user_id,
            language=self._language,  # type: ignore[arg-type]
        )
        bundle = await self._strategy.retrieve(ctx)
        chunks = bundle.chunks
        trace = bundle.trace

        for chunk in chunks:
            tag = _label_for_collection(chunk.collection_name or "")
            score = (
                chunk.rerank_score if chunk.rerank_score is not None else chunk.score
            )
            score_str = f"{score:.2f}" if score is not None else "?"
            doc_short = (chunk.doc_id or "?")[:8]
            # Tag and score on the head row, snippet indented underneath —
            # snippets need their own line because they wrap.
            head = f"{tag} {doc_short} · score={score_str}"
            label = f"{head}\n  {_one_line(chunk.text)}"
            await listview.append(ListItem(Label(label)))

        active = ", ".join(trace.active_collections) or "(none)"
        expanded = trace.expanded_query or "(no expansion)"
        timings = (
            f"embed {trace.embed_ms}ms · search {trace.search_ms}ms · "
            f"rerank {trace.rerank_ms}ms · parent {trace.parent_expand_ms}ms"
        )
        status.update(
            f"{len(chunks)} chunk(s) from [{active}]\n"
            f"expanded query: {expanded}\n"
            f"{timings}"
        )

    async def _count_user_rag_chunks(self) -> int | None:
        """Best-effort count of the user's RAG chunks. None = not initialised."""
        try:
            from claritymed.stores.user_rag import make_user_rag_store

            store = make_user_rag_store(self._user_id)
            docs = await store.list_documents(self._user_id)
        except Exception:  # noqa: BLE001
            logger.exception("user_rag count failed")
            return None
        if not docs:
            return 0
        return sum(d.get("chunk_count", 0) for d in docs)

    async def _system_collection_counts(
        self, names: list[str]
    ) -> dict[str, int | None]:
        """Best-effort point count per system collection via the shared
        Qdrant server. ``None`` for any collection the server doesn't
        recognise or that errors out — the row falls back to ``? chunks``.

        Skipped entirely when RAG is disabled (no server to query); all
        collections come back as ``None``.
        """
        result: dict[str, int | None] = {n: None for n in names}
        if not names or self._strategy is None:
            return result
        try:
            from claritymed.core.rag import load_retrieval_config
            from claritymed.core.rag.qdrant_store import build_qdrant_client
        except Exception:  # noqa: BLE001
            logger.exception("system_collection_counts: import failed")
            return result

        cfg = load_retrieval_config()
        try:
            aclient = build_qdrant_client(
                url=cfg.qdrant.url,
                api_key_env=cfg.qdrant.api_key_env,
            )
        except Exception:  # noqa: BLE001
            logger.exception("system_collection_counts: client build failed")
            return result

        try:
            for name in names:
                try:
                    if not await aclient.collection_exists(name):
                        continue
                    info = await aclient.count(name, exact=True)
                    result[name] = int(info.count)
                except Exception:  # noqa: BLE001
                    logger.exception("system_collection_counts: %s count failed", name)
        finally:
            try:
                await aclient.close()
            except Exception:  # noqa: BLE001
                logger.exception("system_collection_counts: client close failed")
        return result
