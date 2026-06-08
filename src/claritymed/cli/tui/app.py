"""ClarityMed Textual App.

The app owns session state (user, mode, language, provider) and routes user
input — either explicit slash commands or auto-routed plain text — to the
appropriate ``orchestrator.services`` service. It never touches an agent or
store directly; the service layer is the single boundary.

ESC during a streaming response cancels the active worker and marks the
turn as cancelled in place (text stays, gets a ``⊘ cancelled`` tag). The
status bar reflects user / mode / lang / provider / request id / confidence
in real time.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Literal

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.worker import Worker

from claritymed import config as _cfg
from claritymed.cli.entry import DEFAULT_USER_ID
from claritymed.cli.tui.slash_commands import HELP_TEXT, parse
from claritymed.cli.tui.widgets import (
    Conversation,
    InputBar,
    StatusBar,
    Toast,
    ToolSteps,
)
from claritymed.context import (
    apply_context,
    new_request_id,
    reset_context,
)
from claritymed.core.i18n import t
from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.logging import get_access_logger
from claritymed.core.observability.tracing import setup_tracing
from claritymed.orchestrator.services import (
    Cancelled,
    ChatSession,
    ChatTurn,
    Done,
    Error,
    IngestService,
    LlmCallStarted,
    LlmFirstToken,
    ModeRouted,
    RagService,
    RetrievalCompleted,
    RetrievalFiltered,
    RetrievalPending,
    TokenChunk,
    ToolCompleted,
    ToolStarted,
)

if TYPE_CHECKING:
    from claritymed.cli.tui.slash_commands import ParsedCommand
    from claritymed.orchestrator.services import AskService

logger = logging.getLogger(__name__)

ModeName = Literal["ingest", "ask", "rag"]
_MODE_CYCLE: tuple[ModeName, ...] = ("ask", "ingest", "rag")
ROUTING_FLASH_SECONDS = 1.5

# Substrings (lowercased) that mark an upstream context-window overflow.
# Pulled from real Anthropic / OpenAI / Ollama error strings; matches are
# OR'd so any one hit triggers the friendly hint.
_OVERFLOW_MARKERS: tuple[str, ...] = (
    "prompt is too long",
    "context length",
    "context_length_exceeded",
    "maximum context",
    "context window",
    "too many tokens",
    "input is too long",
    "exceeds the maximum",
    "max_tokens",
)
_OVERFLOW_HINT = (
    "Conversation is too long for the model's context window. "
    "Run /clear to start a fresh session, or switch to a provider with a larger context."
)


def _is_context_overflow(message: str) -> bool:
    if not message:
        return False
    lowered = message.lower()
    return any(marker in lowered for marker in _OVERFLOW_MARKERS)


class ClarityMedApp(App):
    """Top-level Textual application."""

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("shift+tab", "cycle_mode", "Cycle mode", show=True),
        Binding("escape", "cancel_stream", "Cancel", show=False),
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("f2", "toggle_steps", "Steps", show=True),
    ]

    def __init__(
        self,
        *,
        user_id: str | None = None,
        language: str | None = None,
        provider_id: str | None = None,
        ask_service_factory=None,
        ingest_service_factory=None,
        rag_service_factory=None,
        chat_session: ChatSession | None = None,
    ) -> None:
        super().__init__()
        self._initial_user_id = user_id or DEFAULT_USER_ID
        self._initial_language = (language or _cfg.default_lang()).lower()
        self._initial_provider_id = provider_id
        self._ask_service_factory = ask_service_factory
        self._ingest_service_factory = ingest_service_factory
        self._rag_service_factory = rag_service_factory
        self._chat_session: ChatSession | None = chat_session
        # Cached per-session RagStrategy when rag.enabled=true. Owns the
        # two AsyncQdrantClient handles inside HybridRetriever; rebuilding
        # per turn would churn the qdrant file lock. The lock serializes
        # the startup warm worker against a first-turn message — without it
        # both would race to build a second Qdrant client and hit the file
        # lock for the same path.
        self._cached_strategy = None
        self._strategy_lock = threading.Lock()

        self._session_turns: list[ChatTurn] = []
        self._stream_worker: Worker | None = None
        # Track the active user_id at App level (not via StatusBar query) so
        # on_unmount runs after Textual has already torn down child widgets.
        self._current_user_id: str = self._initial_user_id

    # ----- layout ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="main_row"):
            yield Conversation(id="conversation")
            yield ToolSteps(id="tool_steps")
        yield InputBar()
        yield StatusBar()

    def on_mount(self) -> None:
        # Boot tracing first turn — no-op when PHOENIX_COLLECTOR_ENDPOINT
        # is unset, so headless tests and offline runs stay untouched.
        setup_tracing()
        status = self.query_one(StatusBar)
        status.user_id = self._initial_user_id
        status.language = self._initial_language
        status.mode = "ask"
        if self._initial_provider_id:
            status.provider_id = self._initial_provider_id
        else:
            status.provider_id = self._resolve_provider_id()
        status.provider_kind = self._resolve_provider_kind(status.provider_id)
        status.request_id = "-"

        # ContextVars are applied per-turn in the streaming worker — Textual
        # message handlers run in separate Contexts, so tokens applied here
        # would not be resettable from on_unmount.

        # Start with a fresh chat session unless one was injected (tests do
        # this; future /resume command will too). Previous sessions stay
        # discoverable on disk via ChatSession.list_sessions(user_id).
        if self._chat_session is None:
            self._chat_session = ChatSession.new(status.user_id)
        conv = self.query_one(Conversation)
        recent = self._chat_session.load_turns()
        if not recent:
            conv.show_empty_state(self._empty_hint(status.mode, status.language))
        else:
            for turn in recent:
                self._session_turns.append(turn)
                if turn.role == "user":
                    conv.add_user_turn(turn.text)
                elif turn.role == "system":
                    conv.add_system_turn(turn.text)
                else:
                    bubble = conv.start_assistant_turn()
                    bubble.append(turn.text)
                    if turn.cancelled:
                        bubble.mark_cancelled()
                    conv.finalize_active()
            self._refresh_context_chars()

        self.query_one(InputBar).focus_input()
        self._refresh_input_placeholder()

        # Eagerly warm the RAG strategy on a worker thread so (a) Qdrant lock
        # conflicts surface at startup rather than 10s into the first turn,
        # and (b) the first message doesn't pay the embedder/qdrant/parent
        # docstore load latency. Skip when RAG is disabled — no spinner, no
        # work. Cheap probe (load yaml only) stays on the main thread.
        from claritymed.core.rag import load_retrieval_config

        if load_retrieval_config().rag.enabled:
            self._warm_rag_strategy()

    @work(thread=True, exclusive=True)
    def _warm_rag_strategy(self) -> None:
        """Build the RAG strategy off the UI thread and report status.

        Runs as a Textual thread worker so the embedder/reranker/Qdrant
        client init doesn't block the first paint. Uses ``call_from_thread``
        for every UI mutation — Textual widgets aren't thread-safe.
        """
        conv = self.query_one(Conversation)
        loading = self.call_from_thread(
            conv.add_system_turn, "⏳ Initializing RAG (embedder, reranker, qdrant)…"
        )
        try:
            self._strategy_for_session()
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(loading.remove)
            self.call_from_thread(conv.add_error_turn, f"service init failed: {exc}")
            return
        self.call_from_thread(loading.remove)

    def on_unmount(self) -> None:
        # No-op: ask-mode turns are persisted in real time by AskService
        # via ``ChatSession.append_assistant`` after each turn. ingest/rag
        # turns are transient UI feedback, not chat history, so nothing
        # needs flushing here.
        return

    # ----- mode + placeholder ---------------------------------------------

    def action_cycle_mode(self) -> None:
        status = self.query_one(StatusBar)
        idx = _MODE_CYCLE.index(status.mode) if status.mode in _MODE_CYCLE else 0
        status.mode = _MODE_CYCLE[(idx + 1) % len(_MODE_CYCLE)]
        self._refresh_input_placeholder()

    def action_toggle_steps(self) -> None:
        """Toggle the right-side steps panel open/closed (F2)."""
        self.query_one(ToolSteps).toggle_collapse()

    def set_mode(self, mode: ModeName) -> None:
        status = self.query_one(StatusBar)
        status.mode = mode
        self._refresh_input_placeholder()

    def _refresh_input_placeholder(self) -> None:
        status = self.query_one(StatusBar)
        key = f"modes.{status.mode}.empty_prompt"
        placeholder = t(key, lang=status.language)
        self.query_one(InputBar).set_placeholder(placeholder)

    def _empty_hint(self, mode: ModeName, language: str) -> str:
        return t(f"modes.{mode}.empty_prompt", lang=language)

    # ----- input dispatch -------------------------------------------------

    def on_input_bar_submitted(self, message: InputBar.Submitted) -> None:
        value = message.value
        input_bar = self.query_one(InputBar)
        input_bar.clear()
        parsed = parse(value)
        if parsed.is_command:
            self._handle_command(parsed)
            return
        if parsed.name == "unknown":
            self.query_one(Conversation).add_command_error_turn(
                f"Unknown command: /{parsed.arg or '?'}"
            )
            return
        self._dispatch_to_service(value)

    def _handle_command(self, parsed: "ParsedCommand") -> None:
        if parsed.name == "help":
            self.query_one(Conversation).add_system_turn(HELP_TEXT)
            return
        if parsed.name == "quit":
            self.exit()
            return
        if parsed.name == "mode":
            target = parsed.arg.strip().lower()
            if target in ("ingest", "ask", "rag"):
                self.set_mode(target)  # type: ignore[arg-type]
                self.query_one(Conversation).add_system_turn(f"Mode → {target}")
            else:
                self._toast("Usage: /mode <ingest|ask|rag>", kind="error")
            return
        if parsed.name == "user":
            new_uid = parsed.arg.strip()
            if not new_uid:
                self._toast("Usage: /user <id>", kind="error")
                return
            self._switch_user(new_uid)
            return
        if parsed.name == "clear":
            self._clear_session()
            return
        if parsed.name == "upload":
            self._open_upload_modal(parsed.arg.strip())
            return
        if parsed.name == "library":
            self._open_library_modal()
            return
        # Unreachable while is_command guards above
        logger.warning("unhandled command: %s", parsed.name)

    def _switch_user(self, user_id: str) -> None:
        # AskService has already persisted the outgoing user's turns to
        # their chat session file. Flip the tracked id, start that user a
        # fresh session, reset the display.
        self._current_user_id = user_id
        self._chat_session = ChatSession.new(user_id)
        status = self.query_one(StatusBar)
        status.user_id = user_id
        self._session_turns.clear()
        status.context_chars = 0
        conv = self.query_one(Conversation)
        for child in list(conv.children):
            child.remove()
        conv.show_empty_state(self._empty_hint(status.mode, status.language))
        self.query_one(ToolSteps).reset()

    def _clear_session(self) -> None:
        # /clear starts a new session_id and a new on-disk file. The old
        # file is left as-is for resume / audit (Claude Code semantics).
        # In-memory history and the visible conversation reset together so
        # the next agent.run_stream call sees a fresh context.
        status = self.query_one(StatusBar)
        self._chat_session = ChatSession.new(self._current_user_id)
        self._session_turns.clear()
        status.context_chars = 0
        conv = self.query_one(Conversation)
        for child in list(conv.children):
            child.remove()
        conv.show_empty_state(self._empty_hint(status.mode, status.language))
        self.query_one(ToolSteps).reset()
        self._toast("New chat session started", kind="info")

    def _open_upload_modal(self, path: str) -> None:
        from claritymed.cli.tui.modals.upload_modal import UploadModal

        def _handle(result):
            if result is None:
                return
            payload, public = result
            self._dispatch_to_service(payload, force_mode="rag", public=public)

        self.push_screen(UploadModal(initial_path=path), _handle)

    def _open_library_modal(self) -> None:
        from claritymed.cli.tui.modals.library_modal import LibraryModal

        self.push_screen(LibraryModal())

    # ----- service dispatch ----------------------------------------------

    def _dispatch_to_service(
        self,
        text: str,
        force_mode: ModeName | None = None,
        public: bool = False,
    ) -> None:
        status = self.query_one(StatusBar)
        conv = self.query_one(Conversation)
        steps = self.query_one(ToolSteps)
        steps.reset()
        conv.add_user_turn(text)
        self._session_turns.append(ChatTurn(role="user", text=text))
        self._refresh_context_chars()

        mode: ModeName = force_mode or status.mode  # type: ignore[assignment]
        # Stream the chosen service in a Textual worker so the UI stays
        # responsive and ESC can cancel via Worker.cancel().
        self._stream_worker = self._run_stream(text, mode, public)

    @work(exclusive=True)
    async def _run_stream(self, text: str, mode: ModeName, public: bool) -> None:
        status = self.query_one(StatusBar)
        conv = self.query_one(Conversation)
        steps = self.query_one(ToolSteps)
        rid = new_request_id()
        status.request_id = rid

        # Apply per-turn ContextVars (request_id flips, user / lang are stable).
        per_turn = apply_context(rid, status.user_id, status.language)
        access = get_access_logger()
        request_status = "ok"
        llm_step = None  # Static widget for the LlmFirstToken step; cleared in finally
        try:
            audit_event(
                "request_start",
                payload={"entry": "tui", "mode": mode},
            )
            access.info("tui_turn_start mode=%s", mode)
            try:
                events = self._make_service_stream(mode, text, status.user_id, public)
            except Exception as exc:  # noqa: BLE001
                request_status = "init_error"
                conv.add_error_turn(f"service init failed: {exc}")
                return

            if mode == "ask":
                conv.start_assistant_turn()
            final_text_parts: list[str] = []
            try:
                async for event in events:
                    if isinstance(event, ModeRouted):
                        self._flash_routing(event.detected_mode, event.confidence)
                    elif isinstance(event, ToolStarted):
                        steps.push_start(event.tool_name, event.args_preview)
                    elif isinstance(event, ToolCompleted):
                        steps.push_complete(
                            event.tool_name, event.duration_ms, event.summary
                        )
                    elif isinstance(event, RetrievalPending):
                        steps.push_start("retrieval", "embed + qdrant + rerank")
                    elif isinstance(event, RetrievalCompleted):
                        total_ms = (
                            event.embed_ms
                            + event.search_ms
                            + event.rerank_ms
                            + event.parent_expand_ms
                        )
                        summary = (
                            f"{event.num_chunks} chunks "
                            f"(embed {event.embed_ms} / search {event.search_ms} "
                            f"/ rerank {event.rerank_ms}ms)"
                        )
                        if event.rerank_fallback:
                            summary += " — rerank fallback"
                        steps.push_complete("retrieval", total_ms, summary)
                    elif isinstance(event, RetrievalFiltered):
                        steps.push_filtered(event.total, event.kept, event.filtered_phi)
                    elif isinstance(event, LlmCallStarted):
                        label = event.model_name or event.provider_id or "model"
                        steps.push_start("llm", f"{label}, awaiting first token…")
                    elif isinstance(event, LlmFirstToken):
                        llm_step = steps.push_complete(
                            "llm first token", event.ttft_ms, "streaming…"
                        )
                    elif isinstance(event, TokenChunk):
                        conv.append_to_active(event.text)
                        final_text_parts.append(event.text)
                    elif isinstance(event, Cancelled):
                        conv.cancel_active()
                        self._session_turns.append(
                            ChatTurn(
                                role="assistant",
                                text="".join(final_text_parts),
                                cancelled=True,
                            )
                        )
                        request_status = "cancelled"
                        return
                    elif isinstance(event, Error):
                        if _is_context_overflow(event.message):
                            conv.add_error_turn(_OVERFLOW_HINT)
                            request_status = "context_overflow"
                        else:
                            conv.add_error_turn(f"{event.error_type}: {event.message}")
                            request_status = "error"
                        return
                    elif isinstance(event, Done):
                        self._on_done(mode, event.final, "".join(final_text_parts))
                        return
            except Exception as exc:  # noqa: BLE001
                if _is_context_overflow(str(exc)):
                    request_status = "context_overflow"
                    conv.add_error_turn(_OVERFLOW_HINT)
                else:
                    request_status = "exception"
                    conv.add_error_turn(f"stream failed: {exc}")
        finally:
            steps.clear_streaming(llm_step)
            try:
                audit_event(
                    "request_end",
                    payload={"status": request_status, "mode": mode},
                )
                access.info(
                    "tui_turn_end mode=%s status=%s",
                    mode,
                    request_status,
                )
            except Exception:  # noqa: BLE001 — never let observability bring down a turn
                logger.exception("failed to emit request_end audit/access")
            reset_context(per_turn)

    def _on_done(self, mode: ModeName, final, streamed_text: str) -> None:
        conv = self.query_one(Conversation)
        if mode == "ask":
            text = streamed_text or (final if isinstance(final, str) else str(final))
            bubble = conv.finalize_active(markdown_text=text)
            if bubble is None and text:
                conv.add_system_turn(text)
            self._session_turns.append(ChatTurn(role="assistant", text=text))
        else:
            summary = getattr(final, "summary", None) or repr(final)
            conv.add_system_turn(f"{mode} ✓ {summary}")
            self._session_turns.append(
                ChatTurn(role="system", text=f"{mode}: {summary}")
            )
        self._refresh_context_chars()

    def _refresh_context_chars(self) -> None:
        total = sum(len(turn.text) for turn in self._session_turns)
        self.query_one(StatusBar).context_chars = total

    def _flash_routing(self, mode: ModeName, confidence: float) -> None:
        status = self.query_one(StatusBar)
        status.routing_flash = mode
        status.confidence = confidence
        self.query_one(Conversation).add_system_turn(
            f"Routed to {mode} (confidence {confidence:.2f})"
        )

        def _clear() -> None:
            status.routing_flash = ""
            status.mode = mode
            self._refresh_input_placeholder()

        self.set_timer(ROUTING_FLASH_SECONDS, _clear)

    def _make_service_stream(
        self,
        mode: ModeName,
        text: str,
        user_id: str,
        public: bool,
    ):
        if mode == "ask":
            service = self._build_ask_service()
            return service.run(text, user_id=user_id)
        if mode == "ingest":
            service = (
                self._ingest_service_factory()
                if self._ingest_service_factory
                else IngestService()
            )
            return service.run(text, user_id=user_id)
        # rag
        service = (
            self._rag_service_factory()
            if self._rag_service_factory
            else self._default_rag_service(user_id)
        )
        return service.run(text, user_id=user_id, public=public)

    def _build_ask_service(self) -> "AskService":
        if self._ask_service_factory is not None:
            return self._ask_service_factory()
        # Production path: resolve provider, build the pydantic-ai model.
        from claritymed.core.llm.model import build_model
        from claritymed.orchestrator.services import AskService
        from claritymed.stores.models import resolve_provider

        provider = resolve_provider(override=self._initial_provider_id)
        model = build_model(provider)
        strategy = self._strategy_for_session()
        if self._chat_session is None:
            self._chat_session = ChatSession.new(self._current_user_id)
        return AskService(
            model=model,
            language=self.query_one(StatusBar).language,
            chat_session=self._chat_session,
            provider_id=provider.id,
            model_name=provider.model,
            strategy=strategy,
            provider_config=provider,
        )

    def _strategy_for_session(self):
        """Build the RAG strategy once per session and cache it.

        The retriever owns AsyncQdrantClient instances that should outlive
        a single turn; rebuilding per ``send`` would re-open the qdrant
        file lock and re-create HTTP clients. Returns ``None`` when
        ``rag.enabled=false`` (no caching needed — fast path stays fast).

        Thread-safe: the warm worker (thread) and the first message turn
        (asyncio task) can both reach this; the lock serializes them so
        only one builds the retriever.
        """
        with self._strategy_lock:
            if self._cached_strategy is not None:
                return self._cached_strategy
            from claritymed.core.rag import load_retrieval_config

            cfg = load_retrieval_config()
            if not cfg.rag.enabled:
                return None
            from claritymed.core.rag import build_hybrid_retriever
            from claritymed.core.rag.strategies import build_strategy

            retriever = build_hybrid_retriever(cfg)
            self._cached_strategy = build_strategy(
                retriever,
                config=cfg.strategies,
                max_evidence=cfg.rag.max_evidence,
            )
            return self._cached_strategy

    @staticmethod
    def _default_rag_service(user_id: str) -> RagService:
        from claritymed.stores.user_rag import make_user_rag_store

        return RagService(store=make_user_rag_store(user_id))

    def _resolve_provider_id(self) -> str:
        """Resolve the active provider id from catalog + account default.

        Caller must have validated ``--provider`` upstream (the Typer ``tui``
        subcommand does this so a typo never reaches the App). Resolution
        errors here surface in the status bar rather than crashing mount —
        the user can still switch via ``/user`` or restart with a valid
        ``--provider``.
        """
        try:
            from claritymed.stores.models import resolve_provider

            provider = resolve_provider(override=self._initial_provider_id)
            return provider.id
        except Exception as exc:  # noqa: BLE001
            logger.warning("provider resolution failed: %s", exc)
            return "?"

    def _resolve_provider_kind(self, provider_id: str) -> str:
        if provider_id == "?":
            return "?"
        try:
            from claritymed.stores.models import resolve_provider

            provider = resolve_provider(override=provider_id)
            return provider.kind
        except Exception as exc:  # noqa: BLE001
            logger.warning("provider kind lookup failed: %s", exc)
            return "?"

    # ----- cancellation + toasts -----------------------------------------

    def action_cancel_stream(self) -> None:
        worker = self._stream_worker
        if worker is None or worker.is_finished:
            return
        worker.cancel()
        conv = self.query_one(Conversation)
        conv.cancel_active()
        self._session_turns.append(ChatTurn(role="assistant", text="", cancelled=True))

    def _toast(self, text: str, kind: str = "info") -> None:
        self.mount(Toast(text, kind=kind))


def run() -> None:
    """Entry point used by the Typer ``claritymed tui`` subcommand."""
    ClarityMedApp().run()


__all__ = ["ClarityMedApp", "run"]
