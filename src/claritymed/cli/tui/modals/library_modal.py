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
import unicodedata

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static

from claritymed import config as _cfg
from claritymed.core.i18n import t
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


def _display_width(text: str) -> int:
    """Terminal display width counting CJK Wide/Fullwidth chars as 2 cells.

    `str.format` pads by character count, but Chinese headers like "类型"
    render 4 cells wide in a monospace font. Without this helper the
    column header drifts right of the body rows.
    """
    return sum(
        2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text
    )


def _pad(text: str, width: int, *, right: bool = False) -> str:
    """Pad ``text`` to ``width`` display cells (not characters).

    Equivalent of ``f"{text:<width}"`` / ``f"{text:>width}"`` but uses
    ``_display_width`` so CJK strings line up with ASCII rows.
    """
    deficit = max(0, width - _display_width(text))
    pad = " " * deficit
    return pad + text if right else text + pad


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


def _list_header(lang: str) -> str:
    """Column header for the list mode table."""
    return (
        f"{_pad(t('library.columns.kind', lang=lang), _COL_KIND)}{_GAP}"
        f"{_pad(t('library.columns.name', lang=lang), _COL_NAME)}{_GAP}"
        f"{_pad(t('library.columns.chunks', lang=lang), _COL_CHUNKS, right=True)}{_GAP}"
        f"{_pad(t('library.columns.notes', lang=lang), _COL_NOTES)}{_GAP}"
        f"{t('library.columns.topics', lang=lang)}"
    )


def _row_first_line(
    tag: str, name: str, chunks_str: str, notes: str, first_topic: str
) -> str:
    """Render the first visual line of a collection row (all 5 columns)."""
    return (
        f"{_pad(tag, _COL_KIND)}{_GAP}"
        f"{_pad(name, _COL_NAME)}{_GAP}"
        f"{_pad(chunks_str, _COL_CHUNKS, right=True)}{_GAP}"
        f"{_pad(notes, _COL_NOTES)}{_GAP}"
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
        lang = self._language
        with Vertical():
            yield Label(t("library.title", lang=lang), id="title")
            yield Input(
                value=self._initial_query,
                placeholder=t("library.search_placeholder", lang=lang),
                id="query",
            )
            yield Static("", id="header")
            yield ListView(id="results")
            yield Static("", id="status")
            with Horizontal(id="buttons"):
                yield Button(t("library.close_button", lang=lang), id="close")

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
        lang = self._language
        listview = self.query_one("#results", ListView)
        status = self.query_one("#status", Static)
        header = self.query_one("#header", Static)
        await listview.clear()
        status.update(t("library.loading", lang=lang))
        try:
            if query:
                header.update("")
                await self._render_search(listview, status, query)
            else:
                header.update(_list_header(lang))
                await self._render_list(listview, status)
        except Exception as exc:  # noqa: BLE001
            logger.exception("library modal load failed (query=%r)", query)
            status.update(t("library.error", lang=lang, error=str(exc)))

    async def _render_list(self, listview: ListView, status: Static) -> None:
        """Render every collection as a 5-column row with one topic per line.

        Columns: KIND, NAME, CHUNKS, NOTES, TOPICS. Continuation lines for
        a multi-topic row blank-pad the first 4 columns and indent the
        topic under the TOPICS header.
        """
        lang = self._language
        system_cols = _system_rag_collections()
        system_counts = await self._system_collection_counts(
            [c.get("name", "") for c in system_cols if c.get("name")]
        )
        user_chunk_count = await self._count_user_rag_chunks()

        topics_none_label = t("library.topics_none", lang=lang)
        for col in system_cols:
            name = col.get("name", "?")
            col_language = col.get("language", "?")
            tier = col.get("authority_tier")
            tier_str = (
                t("library.tier", lang=lang, tier=tier)
                if tier is not None
                else t("library.tier_unknown", lang=lang)
            )
            notes = f"{col_language} · {tier_str}"
            count = system_counts.get(name)
            chunks_str = str(count) if count is not None else "?"
            topics = col.get("topics") or []
            first_topic = topics[0] if topics else topics_none_label
            row_lines = [
                _row_first_line(_SYS_TAG, name, chunks_str, notes, first_topic)
            ]
            for topic in topics[1:]:
                row_lines.append(_topic_continuation(topic))
            await listview.append(ListItem(Label("\n".join(row_lines))))

        if user_chunk_count is None:
            user_chunks_str = "?"
            user_topic = t("library.user.not_initialised", lang=lang)
        else:
            user_chunks_str = str(user_chunk_count)
            user_topic = t("library.user.your_uploads", lang=lang)
        user_notes = t("library.user.notes", lang=lang, user_id=self._user_id)
        user_row = _row_first_line(
            _USER_TAG, "user_rag", user_chunks_str, user_notes, user_topic
        )
        await listview.append(ListItem(Label(user_row)))

        rag_state = (
            t("library.retrieval_ready", lang=lang)
            if self._strategy is not None
            else t("library.retrieval_disabled", lang=lang)
        )
        status.update(
            t(
                "library.list_status",
                lang=lang,
                system_count=len(system_cols),
                state=rag_state,
            )
        )

    async def _render_search(
        self, listview: ListView, status: Static, query: str
    ) -> None:
        """Run the same strategy AskService uses and surface chunks + trace."""
        lang = self._language
        if self._strategy is None:
            status.update(t("library.disabled_banner", lang=lang))
            return
        ctx = RetrievalContext(
            query=query,
            user_id=self._user_id,
            language=lang,  # type: ignore[arg-type]
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
            score_label = t("library.score", lang=lang, score=score_str)
            head = f"{tag} {doc_short} · {score_label}"
            label = f"{head}\n  {_one_line(chunk.text)}"
            await listview.append(ListItem(Label(label)))

        active = ", ".join(trace.active_collections) or t(
            "library.none_collections", lang=lang
        )
        expanded = trace.expanded_query or t("library.no_expansion", lang=lang)
        timings = t(
            "library.timings",
            lang=lang,
            embed_ms=trace.embed_ms,
            search_ms=trace.search_ms,
            rerank_ms=trace.rerank_ms,
            parent_ms=trace.parent_expand_ms,
        )
        status.update(
            t(
                "library.search_status",
                lang=lang,
                count=len(chunks),
                collections=active,
                expanded=expanded,
                timings=timings,
            )
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
