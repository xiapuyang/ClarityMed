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
from claritymed.orchestrator.services import (
    Cancelled,
    Done,
    Error,
    IngestService,
    ModeRouted,
    RagService,
    RetrievalFiltered,
    TokenChunk,
    ToolCompleted,
    ToolStarted,
)
from claritymed.stores.chat_memory import ChatTurn, LanceChatMemoryStore

if TYPE_CHECKING:
    from claritymed.cli.tui.slash_commands import ParsedCommand
    from claritymed.orchestrator.services import AskService

logger = logging.getLogger(__name__)

ModeName = Literal["ingest", "ask", "rag"]
_MODE_CYCLE: tuple[ModeName, ...] = ("ask", "ingest", "rag")
ROUTING_FLASH_SECONDS = 1.5


class ClarityMedApp(App):
    """Top-level Textual application."""

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("shift+tab", "cycle_mode", "Cycle mode", show=True),
        Binding("escape", "cancel_stream", "Cancel", show=False),
        Binding("ctrl+c", "quit", "Quit", show=True),
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
        chat_memory_store=None,
    ) -> None:
        super().__init__()
        self._initial_user_id = user_id or DEFAULT_USER_ID
        self._initial_language = (language or _cfg.default_lang()).lower()
        self._initial_provider_id = provider_id
        self._ask_service_factory = ask_service_factory
        self._ingest_service_factory = ingest_service_factory
        self._rag_service_factory = rag_service_factory
        self._chat_memory_store = chat_memory_store

        self._session_turns: list[ChatTurn] = []
        self._stream_worker: Worker | None = None
        # Track the active user_id at App level (not via StatusBar query) so
        # on_unmount can save the transcript after Textual has already torn
        # down child widgets.
        self._current_user_id: str = self._initial_user_id

    # ----- layout ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="main_row"):
            yield Conversation(id="conversation")
            yield ToolSteps(id="tool_steps")
        yield InputBar()
        yield StatusBar()

    def on_mount(self) -> None:
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

        # Load recent turns from chat memory (v1 stub returns []).
        store = self._chat_memory_store or LanceChatMemoryStore(status.user_id)
        recent = store.load_recent(k=10)
        conv = self.query_one(Conversation)
        if not recent:
            conv.show_empty_state(self._empty_hint(status.mode, status.language))
        else:
            for turn in recent:
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

        self.query_one(InputBar).focus_input()
        self._refresh_input_placeholder()

    def on_unmount(self) -> None:
        # Persist transcript on exit. Use the tracked user_id rather than
        # querying StatusBar — Textual has already unmounted child widgets
        # by the time this runs, so query_one(StatusBar) would raise.
        try:
            store = self._chat_memory_store or LanceChatMemoryStore(
                self._current_user_id
            )
            store.save_turns(self._session_turns)
        except Exception:  # noqa: BLE001
            logger.exception("save_turns failed on TUI shutdown")

    # ----- mode + placeholder ---------------------------------------------

    def action_cycle_mode(self) -> None:
        status = self.query_one(StatusBar)
        idx = _MODE_CYCLE.index(status.mode) if status.mode in _MODE_CYCLE else 0
        status.mode = _MODE_CYCLE[(idx + 1) % len(_MODE_CYCLE)]
        self._refresh_input_placeholder()

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
            self._toast(f"Unknown command: /{parsed.arg or '?'}", kind="error")
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
        if parsed.name == "upload":
            self._open_upload_modal(parsed.arg.strip())
            return
        if parsed.name == "library":
            self._open_library_modal()
            return
        # Unreachable while is_command guards above
        logger.warning("unhandled command: %s", parsed.name)

    def _switch_user(self, user_id: str) -> None:
        # Flush the previous user's transcript before switching — otherwise
        # the in-memory turns would land in the new user's file on shutdown.
        try:
            outgoing = self._chat_memory_store or LanceChatMemoryStore(
                self._current_user_id
            )
            outgoing.save_turns(self._session_turns)
        except Exception:  # noqa: BLE001
            logger.exception("save_turns failed on /user switch")

        self._current_user_id = user_id
        status = self.query_one(StatusBar)
        status.user_id = user_id
        self._session_turns.clear()
        status.context_chars = 0
        conv = self.query_one(Conversation)
        for child in list(conv.children):
            child.remove()
        conv.show_empty_state(self._empty_hint(status.mode, status.language))
        self.query_one(ToolSteps).reset()

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
        try:
            try:
                events = self._make_service_stream(mode, text, status.user_id, public)
            except Exception as exc:  # noqa: BLE001
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
                    elif isinstance(event, RetrievalFiltered):
                        steps.push_filtered(event.total, event.kept, event.filtered_phi)
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
                        return
                    elif isinstance(event, Error):
                        conv.add_error_turn(f"{event.error_type}: {event.message}")
                        return
                    elif isinstance(event, Done):
                        self._on_done(mode, event.final, "".join(final_text_parts))
                        return
            except Exception as exc:  # noqa: BLE001
                conv.add_error_turn(f"stream failed: {exc}")
        finally:
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
            else self._default_rag_service()
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
        return AskService(model=model, language=self.query_one(StatusBar).language)

    @staticmethod
    def _default_rag_service() -> RagService:
        from claritymed.stores.user_rag import UserRagStore

        return RagService(store=UserRagStore.from_defaults())

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
