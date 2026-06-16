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

import faulthandler
import logging
import re
import signal
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from textual import events, work
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
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


_MIME_BY_EXT: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "pdf": "application/pdf",
    "txt": "text/plain",
    "md": "text/markdown",
}


def _guess_mime(ext: str) -> str:
    """ext (without leading dot) → MIME, falling back to octet-stream.

    Local table beats ``mimetypes.guess_type`` here because the latter
    needs a filename and we only have the bare extension at paste time.
    """
    return _MIME_BY_EXT.get(ext.lower(), "application/octet-stream")


def _ocr_step_name(sha256: str) -> str:
    """Stable per-blob step name so push_start / push_complete pair up.

    The short sha keeps the panel readable while still being unique per
    blob — two pasted images get two distinct rows."""
    return f"ocr:{sha256[:8]}"


def _human_size(size: int) -> str:
    """Render a byte count as B / KB / MB for toasts and step rows.

    Used in size-limit toasts and the upload step's ``args_preview`` so
    users see a recognisable magnitude rather than raw byte counts."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


# Ghostty 1.1.0+ joins multi-file drops with a single space; iTerm2 uses
# newlines. Within a path, spaces are shell-escaped as ``\ `` so a bare
# space only separates files when followed by an absolute-path marker
# (``/`` or ``C:\``). Same regex claude-code uses in usePasteHandler.ts.
_DROP_PATH_SPLIT = re.compile(r" (?=/|[A-Za-z]:\\)")


def _strip_outer_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    return text


def _unescape_shell_path(text: str) -> str:
    """Reverse ``\\ `` / ``\\(`` shell escapes a terminal injects when it
    drops a path with spaces or special chars. macOS/Linux only — Windows
    paths use backslash as a separator, leave them alone."""
    if not text or "\\" not in text:
        return text
    # Two passes so a literal ``\\`` survives — sub the doubled form to a
    # placeholder, strip remaining single-backslash escapes, then put
    # literal backslashes back.
    _SENTINEL = "\x00DBL_BS\x00"
    return text.replace("\\\\", _SENTINEL).replace("\\", "").replace(_SENTINEL, "\\")


def _looks_like_drop_attempt(text: str) -> bool:
    """Heuristic for "user dragged something, but it did not resolve to
    a real file."

    Drag-drop in Ghostty / iTerm2 / WezTerm always lands as one or more
    absolute path tokens. The shapes we recognise:

    * ``/Users/...`` / ``~/...`` (POSIX)
    * ``C:\\Users\\...`` (Windows)
    * ``"path"`` / ``'path'`` (terminals that quote paths containing spaces)
    * ``file:///...`` (file URIs — sent by some Linux desktops on drag-drop)
    * Multi-drop: space-followed-by-path-token anywhere in the string

    Bare text without those tokens is left to fall through to the Input
    so non-drop pastes keep their existing behaviour.
    """
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    # Strip a single layer of matching outer quotes so quoted paths
    # ("/foo bar/baz.pdf") look like the plain form to the rest of the
    # checks. The real ``_parse_dropped_paths`` quote-strip runs per
    # candidate; this is just for the "did the user mean to drop?"
    # answer.
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in ("'", '"'):
        stripped = stripped[1:-1].strip()
    if not stripped:
        return False
    if stripped[0] in {"/", "~"}:
        return True
    # file:// URI form
    if stripped.lower().startswith("file://"):
        return True
    # Windows: ``C:\path``
    if len(stripped) >= 3 and stripped[1:3] == ":\\":
        return True
    # Ghostty multi-drop uses space-separated paths; detect any path
    # token in the middle/end of the string.
    return bool(_DROP_PATH_SPLIT.search(text))


def _parse_dropped_paths(text: str) -> list[Path]:
    """Return absolute Paths the user dropped, or [] if none parsed.

    Splits on Ghostty's space-separator and iTerm2's newline-separator,
    un-escapes shell escapes, expands ``~``, decodes ``file://`` URIs,
    and keeps only entries that resolve to an existing file. Empty list
    = treat the paste as text."""
    candidates: list[str] = []
    for chunk in _DROP_PATH_SPLIT.split(text.strip()):
        for line in chunk.split("\n"):
            line = _strip_outer_quotes(line.strip())
            if line:
                candidates.append(_unescape_shell_path(line))
    paths: list[Path] = []
    for raw in candidates:
        decoded = _decode_file_uri(raw)
        try:
            p = Path(decoded).expanduser()
        except (OSError, ValueError):
            continue
        if p.is_file():
            paths.append(p)
    return paths


def _decode_file_uri(text: str) -> str:
    """Convert ``file:///abs/path`` into a plain ``/abs/path``.

    Some desktops (GNOME, KDE) and a few terminals send ``file://`` URIs
    on drag-drop. The hostname segment (``file://localhost/...``) is
    optional and uniformly empty in practice; we tolerate either form.
    Pass-through for anything that doesn't start with the scheme.
    """
    from urllib.parse import unquote, urlparse

    if not text.lower().startswith("file://"):
        return text
    try:
        parsed = urlparse(text)
    except ValueError:
        return text
    # ``urlparse`` returns netloc = "localhost" or "" and path = "/abs".
    # Both yield the same on-disk path.
    return unquote(parsed.path) or text


class ClarityMedApp(App):
    """Top-level Textual application."""

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("shift+tab", "cycle_mode", "Cycle mode", show=True),
        Binding("escape", "cancel_stream", "Cancel", show=False),
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+v", "paste_clipboard", "Paste", show=True),
        Binding("f2", "toggle_steps", "Steps", show=True),
        Binding("f3", "pick_provider", "Provider", show=True),
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
        # Mutable — updated by /provider <id> at runtime.
        self._current_provider_id: str | None = provider_id
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
        # Cached AskService — rebuilt only when provider or user changes.
        # Avoids re-running resolve_provider / build_model / make_translation_provider
        # on every turn (the first build pays the cost; subsequent turns update
        # the session reference in-place and return immediately).
        self._cached_ask_service: "AskService | None" = None
        # Track the active user_id at App level (not via StatusBar query) so
        # on_unmount runs after Textual has already torn down child widgets.
        self._current_user_id: str = self._initial_user_id
        # OcrWorker is lazy-built on first paste — see ``_ensure_ocr_worker``.
        # Headless / one-shot tests never construct one, so we avoid paying
        # the provider-chain build cost on every mount.
        self._ocr_worker = None
        # Per-app monotonic counter so each ingest gets a unique upload row
        # in ToolSteps; using the counter (not the sha) lets us push the
        # ``⟳`` row BEFORE hashing the bytes.
        self._upload_seq: int = 0
        # Cached paste size limit — re-read on demand via _max_paste_bytes()
        # so config edits take effect on next app start, not mid-session.
        self._max_paste_bytes_cache: int | None = None

    # ----- layout ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="main_row"):
            yield Conversation(id="conversation")
            yield ToolSteps(id="tool_steps")
        yield InputBar()
        yield StatusBar()

    def on_mount(self) -> None:
        # Dump all thread stacks to stderr on SIGUSR1 — useful when the TUI
        # freezes: run `kill -USR1 <pid>` to see exactly where each thread is.
        # Uses sys.__stderr__ because Textual replaces sys.stderr with an
        # internal pipe that has no real file descriptor.
        import sys

        if sys.__stderr__ is not None:
            try:
                faulthandler.register(signal.SIGUSR1, file=sys.__stderr__)
            except Exception:  # noqa: BLE001
                pass
        # Boot tracing first turn — no-op when tracing.enabled is false
        # in app.yaml, so headless tests and offline runs stay untouched.
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

        # Warm the paste-time supported-extension set off the UI thread.
        # Cold path is ~120ms (import 6 OCR provider modules + parse
        # ocr.yaml + compute union). Without this, the first paste pays
        # that latency before its placeholder appears in the input,
        # which feels janky on drag-drop because the user is staring
        # straight at the cursor when they release.
        self._warm_paste_pipeline()

        # Eagerly warm the RAG strategy on a worker thread so (a) Qdrant lock
        # conflicts surface at startup rather than 10s into the first turn,
        # and (b) the first message doesn't pay the embedder/qdrant/parent
        # docstore load latency. Skip when RAG is disabled — no spinner, no
        # work. Cheap probe (load yaml only) stays on the main thread.
        from claritymed.core.rag import load_retrieval_config

        if load_retrieval_config().rag.enabled:
            self._warm_rag_strategy()

    @work(thread=True, exclusive=True, group="warm_paste")
    def _warm_paste_pipeline(self) -> None:
        """Pre-compute ``_supported_extensions`` on a background thread.

        Result lands in the same cache the first paste would populate
        synchronously, so the user's first drag-drop / Ctrl+V skips
        the cold-import hit. Failures are silent — if the warm-up
        crashes for some reason, the first paste pays the cold path
        and any error surfaces there instead of taking down the App.
        """
        try:
            self._supported_extensions()
        except Exception:  # noqa: BLE001
            logger.exception(
                "paste-pipeline pre-warm failed; first paste pays cold cost"
            )

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
        # Persistence: ask-mode turns are flushed in real time by AskService
        # via ``ChatSession.append_assistant`` after each turn. ingest / rag
        # turns are transient UI feedback, not chat history. The only thing
        # we own beyond the event loop is the OcrWorker — its background
        # task gets cancelled here so a half-finished extraction doesn't
        # leak past app exit.
        worker = getattr(self, "_ocr_worker", None)
        if worker is not None:
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(worker.stop())
                else:
                    loop.run_until_complete(worker.stop())
            except Exception:  # noqa: BLE001
                logger.exception("OcrWorker stop failed during unmount")
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

    def action_pick_provider(self) -> None:
        """Open the provider picker modal (F3)."""
        self._open_provider_modal()

    def _open_provider_modal(self) -> None:
        from claritymed.cli.tui.modals.provider_modal import ProviderModal

        current = self.query_one(StatusBar).provider_id

        def _handle(result: str | None) -> None:
            if result:
                self._switch_provider(result)

        self.push_screen(ProviderModal(current_provider_id=current), _handle)

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
        parsed = parse(value)
        # Block plain-text messages while a stream is in progress — commands
        # (/clear, /provider, etc.) still go through so the user isn't locked out.
        worker = self._stream_worker
        if worker is not None and not worker.is_finished and not parsed.is_command:
            return
        input_bar.clear()
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
        if parsed.name == "provider":
            arg = parsed.arg.strip()
            if arg:
                self._switch_provider(arg)
            else:
                self._open_provider_modal()
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
        self._cached_ask_service = None
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

    def _switch_provider(self, provider_id: str) -> None:
        if not provider_id:
            self._toast("Usage: /provider <id>", kind="error")
            return
        try:
            from claritymed.stores.models import resolve_provider

            provider = resolve_provider(override=provider_id)
        except Exception as exc:  # noqa: BLE001
            self._toast(f"Unknown provider: {provider_id}", kind="error")
            logger.warning("provider switch failed: %s", exc)
            return
        self._cached_ask_service = None
        self._current_provider_id = provider.id
        status = self.query_one(StatusBar)
        status.provider_id = provider.id
        status.provider_kind = provider.kind
        # Persist to settings.yaml so the choice survives restarts.
        try:
            from claritymed.stores.account import AccountStore

            store = AccountStore(self._current_user_id)
            if store.exists():
                account = store.load()
                store.save(account.model_copy(update={"provider_id": provider.id}))
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to persist provider switch: %s", exc)
        self.query_one(Conversation).add_system_turn(
            f"Provider → {provider.id}  ({provider.model})"
        )

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
        # Preserve any in-flight rows (e.g. a pending OCR job enqueued by
        # an earlier paste). Wiping them here makes the right panel
        # collapse mid-turn and re-appear when OCR finishes — bad UX and
        # the completed row loses its ⟳ → ✓ in-place transition.
        steps.reset(preserve_active=True)
        conv.add_user_turn(text)
        self._session_turns.append(ChatTurn(role="user", text=text))
        self._refresh_context_chars()

        mode: ModeName = force_mode or status.mode  # type: ignore[assignment]
        self.query_one(InputBar).set_streaming(True)
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
        logger.debug("_run_stream: START rid=%s mode=%s", rid, mode)
        try:
            audit_event(
                "request_start",
                payload={"entry": "tui", "mode": mode},
            )
            access.info("tui_turn_start mode=%s", mode)
            try:
                events = await self._make_service_stream(
                    mode, text, status.user_id, public
                )
            except Exception as exc:  # noqa: BLE001
                request_status = "init_error"
                conv.add_error_turn(f"service init failed: {exc}")
                return

            final_text_parts: list[str] = []
            # Byte offset into final_text_parts at the point ask_user_question
            # fired. Used to discard pre-tool text when the user declines.
            _ask_pre_tool_len: int = 0
            try:
                async for event in events:
                    if isinstance(event, ModeRouted):
                        self._flash_routing(event.detected_mode, event.confidence)
                    elif isinstance(event, ToolStarted):
                        if event.tool_name == "ask_user_question":
                            # Track pre-tool text length so we can discard it
                            # if the user declines. The bubble itself is removed
                            # by TextualPromptChannel.ask() just before the modal
                            # opens — that is the canonical clear point.
                            _ask_pre_tool_len = len("".join(final_text_parts))
                            logger.debug(
                                "_run_stream: ToolStarted ask_user_question pre_tool_len=%d",
                                _ask_pre_tool_len,
                            )
                        steps.push_start(event.tool_name, event.args_preview)
                    elif isinstance(event, ToolCompleted):
                        if event.tool_name == "ask_user_question":
                            logger.debug(
                                "_run_stream: ToolCompleted ask_user_question summary=%r",
                                event.summary,
                            )
                        if (
                            event.tool_name == "ask_user_question"
                            and event.summary == "declined"
                        ):
                            # Drop all text the model generated before the
                            # tool call so finalize_active doesn't re-render
                            # the pre-tool answer.
                            combined = "".join(final_text_parts)
                            post = combined[_ask_pre_tool_len:]
                            final_text_parts.clear()
                            if post:
                                final_text_parts.append(post)
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
                            "llm", event.ttft_ms, "streaming…"
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
            logger.debug("_run_stream: FINALLY rid=%s status=%s", rid, request_status)
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
            try:
                self.query_one(InputBar).set_streaming(False)
                self._refresh_input_placeholder()
            except NoMatches:
                pass  # App is tearing down; widgets already unmounted.

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

    async def _make_service_stream(
        self,
        mode: ModeName,
        text: str,
        user_id: str,
        public: bool,
    ):
        if mode == "ask":
            service = await self._build_ask_service()
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

    async def _build_ask_service(self) -> "AskService":
        if self._ask_service_factory is not None:
            return self._ask_service_factory()
        # Return the cached service when the provider hasn't changed, updating
        # only the mutable turn-to-turn state (session ref + language). This
        # avoids re-running resolve_provider / build_model / threading.Lock on
        # every Enter press, which was causing the noticeable submit lag.
        if self._cached_ask_service is not None:
            if self._chat_session is None:
                self._chat_session = ChatSession.new(self._current_user_id)
            self._cached_ask_service._chat_session = self._chat_session
            self._cached_ask_service._language = self.query_one(StatusBar).language
            return self._cached_ask_service
        # First turn or after a provider/user switch — build from scratch.
        # _strategy_for_session acquires a threading.Lock (the warm-up worker
        # may still hold it); running it in a thread executor keeps the asyncio
        # event loop free while we wait, so the UI stays responsive.
        import asyncio

        from claritymed.core.llm.model import build_model
        from claritymed.orchestrator.services import AskService
        from claritymed.stores.models import resolve_provider

        provider = resolve_provider(override=self._current_provider_id)
        model = build_model(provider)
        strategy = await asyncio.to_thread(self._strategy_for_session, model=model)
        if self._chat_session is None:
            self._chat_session = ChatSession.new(self._current_user_id)
        from claritymed.core.rag import load_retrieval_config
        from claritymed.core.translation import make_translation_provider

        mode_name = load_retrieval_config().rag.mode
        from claritymed.cli.tui.prompt_channel import TextualPromptChannel
        from claritymed.cli.tui.tool_approval_channel import (
            TextualToolApprovalChannel,
        )
        from claritymed.config import load_yaml

        profile_context_mode = (
            load_yaml("app.yaml")
            .get("profile_context", {})
            .get("mode", "deterministic")
        )
        from claritymed.orchestrator.features.symptoms_plugin import (
            make_symptoms_factory,
        )
        from claritymed.orchestrator.features.vision_plugin import (
            make_vision_factory,
        )

        _app = self

        def _tui_session_id() -> str | None:
            return (
                _app._chat_session.session_id
                if _app._chat_session is not None
                else None
            )

        service = AskService(
            model=model,
            language=self.query_one(StatusBar).language,
            chat_session=self._chat_session,
            provider_id=provider.id,
            model_name=provider.model,
            strategy=strategy,
            provider_config=provider,
            translation_service=make_translation_provider(
                model, phi_kind=provider.kind
            ),
            rag_mode=mode_name,
            profile_context_mode=profile_context_mode,
            prompt_channel=TextualPromptChannel(self),
            tool_approval_channel=TextualToolApprovalChannel(
                self, language=self.query_one(StatusBar).language
            ),
            symptoms_factory=make_symptoms_factory(),
            vision_factory=make_vision_factory(get_session_id=_tui_session_id),
        )
        self._cached_ask_service = service
        return service

    def _strategy_for_session(self, model=None):
        """Build the RAG strategy once per session and cache it.

        The retriever owns AsyncQdrantClient instances that should outlive
        a single turn; rebuilding per ``send`` would re-open the qdrant
        file lock and re-create HTTP clients. Returns ``None`` when
        ``rag.enabled=false`` (no caching needed — fast path stays fast).

        Thread-safe: the warm worker (thread) and the first message turn
        (asyncio task) can both reach this; the lock serializes them so
        only one builds the retriever.

        ``model`` is forwarded to ``build_strategy`` so HyDE can issue
        its LLM call against the same provider the rest of the turn
        uses; other strategies ignore it.
        """
        _LOCK_TIMEOUT = 30
        logger.debug("_strategy_for_session: waiting for strategy lock")
        if not self._strategy_lock.acquire(timeout=_LOCK_TIMEOUT):
            raise RuntimeError(
                f"strategy lock timed out after {_LOCK_TIMEOUT}s — possible deadlock"
            )
        try:
            logger.debug("_strategy_for_session: lock acquired")
            if self._cached_strategy is not None:
                return self._cached_strategy
            from claritymed.core.rag import load_retrieval_config

            cfg = load_retrieval_config()
            if not cfg.rag.enabled:
                return None
            from claritymed.core.rag import build_hybrid_retriever
            from claritymed.core.rag.strategies import build_strategy

            logger.debug("_strategy_for_session: building retriever")
            retriever = build_hybrid_retriever(cfg)
            logger.debug("_strategy_for_session: building strategy")
            self._cached_strategy = build_strategy(
                retriever,
                config=cfg.strategies,
                max_evidence=cfg.rag.max_evidence,
                model=model,
            )
            logger.debug("_strategy_for_session: strategy ready")
            return self._cached_strategy
        finally:
            self._strategy_lock.release()
            logger.debug("_strategy_for_session: lock released")

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

    # ----- clipboard paste -----------------------------------------------

    def on_paste(self, event: events.Paste) -> None:
        """App-level handler for bracketed-paste (incl. drag-drop in
        Ghostty / iTerm2 / WezTerm). When the dropped payload is one or
        more file paths, route each through ``_ingest_clipboard_bytes``
        (same code path as Ctrl+V's ``FilePath``) and stop the event so
        the path string itself never lands in the Input. Anything else
        bubbles to the focused Input, which inserts it as text.
        """
        text = event.text
        # INFO-level (not debug) so we can diagnose "drag-drop didn't
        # work" reports without asking users to enable debug logging.
        # The first 200 chars is enough to see the path shape /
        # quoting / scheme without flooding the log on giant pastes.
        logger.info(
            "on_paste: len=%d first=%r",
            len(text),
            text[:200] if text else "",
        )
        paths = _parse_dropped_paths(text)
        if not paths:
            # Drag-drop that looked like a path string but resolved to no
            # actual files (folder, non-existent target, or just garbled
            # text). Without a toast the user sees "nothing happen" and
            # assumes the app is broken. The heuristic that distinguishes
            # "drag-drop attempt" from "plain text paste" is the literal
            # presence of a path-prefix token (``/`` or ``C:\`` / quoted
            # form / ``file://``) — bare text falls through to the Input
            # as before.
            if text and _looks_like_drop_attempt(text):
                logger.info(
                    "on_paste: looked like drop but no file resolved (text=%r)",
                    text[:200],
                )
                self._toast(
                    "Could not ingest the dropped item — only existing files "
                    "are supported (folders and broken paths are skipped).",
                    kind="error",
                )
                event.stop()
                event.prevent_default()
            return
        for path in paths:
            # stat() first so we never spend a multi-second read on a
            # file we'd reject anyway. Skip oversize early — the toast
            # in _size_check_or_reject tells the user why nothing
            # appeared in the input.
            try:
                size = path.stat().st_size
            except OSError as exc:
                self._toast(f"Could not stat {path.name}: {exc}", kind="error")
                continue
            if not self._size_check_or_reject(size, path.name):
                continue
            try:
                data = path.read_bytes()
            except OSError as exc:
                self._toast(f"Could not read {path.name}: {exc}", kind="error")
                continue
            ext = path.suffix.lstrip(".") or "bin"
            self._ingest_clipboard_bytes(data, ext=ext, display_name=path.name)
        event.stop()
        event.prevent_default()

    def action_paste_clipboard(self) -> None:
        """Ctrl+V handler — routes clipboard content by type.

        ImageBytes / FilePath → blob + SessionAttachments + OCR enqueue.
        Small text → inserted into the Input. Large text → placeholder.
        Empty → toast hint. All exceptions become toasts so a broken
        platform helper never crashes the TUI.
        """
        from claritymed.cli.tui.paste import (
            Empty,
            FilePath,
            ImageBytes,
            LargeText,
            SmallText,
            read_clipboard,
        )

        try:
            content = read_clipboard()
        except Exception as exc:  # noqa: BLE001
            logger.warning("clipboard read failed: %s", exc)
            self._toast("Clipboard read failed", kind="error")
            return

        if isinstance(content, Empty):
            self._toast("Clipboard is empty", kind="info")
            return
        if isinstance(content, ImageBytes):
            if not self._size_check_or_reject(
                len(content.bytes), f"clipboard image (.{content.ext})"
            ):
                return
            self._ingest_clipboard_bytes(content.bytes, ext=content.ext)
            return
        if isinstance(content, FilePath):
            try:
                size = content.path.stat().st_size
            except OSError as exc:
                self._toast(f"Could not stat {content.path.name}: {exc}", kind="error")
                return
            if not self._size_check_or_reject(size, content.path.name):
                return
            try:
                data = content.path.read_bytes()
            except OSError as exc:
                self._toast(f"Could not read {content.path.name}: {exc}", kind="error")
                return
            ext = content.path.suffix.lstrip(".") or "bin"
            self._ingest_clipboard_bytes(
                data,
                ext=ext,
                display_name=content.path.name,
            )
            return
        if isinstance(content, LargeText):
            self._toast(
                f"Large clipboard text ({len(content.text)} chars) inserted",
                kind="info",
            )
            self._insert_into_input(content.text)
            return
        if isinstance(content, SmallText):
            self._insert_into_input(content.text)
            return

    def _ingest_clipboard_bytes(
        self,
        data: bytes,
        *,
        ext: str,
        display_name: str | None = None,
    ) -> None:
        """Store bytes in the CAS blob pool, register on SessionAttachments,
        enqueue an OCR job. All side effects are loud-on-failure: a missing
        chat_session, missing OcrWorker, or a write error becomes an error
        toast rather than silently dropping the paste."""
        import time

        from claritymed.orchestrator.services.ocr_worker import OcrJob
        from claritymed.stores.session_attachments import (
            SessionAttachments,
        )
        from claritymed.stores.blob_store import BlobStore

        if self._chat_session is None:
            self._toast("No active chat session for paste", kind="error")
            return

        # Supported-ext gate: reject paste targets that no provider can
        # handle BEFORE anything touches blob storage or session state.
        # Extensionless / mislabeled files get a Magika recovery pass
        # — if the detector recognises bytes as something supported, we
        # rewrite ``ext`` and proceed.
        ext, gate_ok = self._gate_or_recover_extension(data, ext, display_name)
        if not gate_ok:
            return

        # Defense-in-depth: entry points (on_paste, action_paste_clipboard)
        # already gate on size, but a future code path that hands us bytes
        # directly would slip past — keep the cheap len() check here.
        filename = display_name or f"clipboard.{ext}"
        if not self._size_check_or_reject(len(data), filename):
            return

        # Push the "uploading" row before any heavy work so the user sees
        # it as soon as the bytes start landing on disk. Counter-based
        # label so we don't need the sha yet. Row flips to ✓ once the
        # blob + session attach succeed; on failure, it flips to ✓ with
        # a "failed: …" summary so a row never gets stranded as ⟳.
        upload_label = self._next_upload_label()
        upload_t0 = time.monotonic()
        size_label = _human_size(len(data))
        try:
            steps = self.query_one(ToolSteps)
            steps.push_start(upload_label, args_preview=f"{filename} ({size_label})")
        except NoMatches:
            pass
        logger.info(
            "upload start filename=%s size=%d ext=%s",
            filename,
            len(data),
            ext,
        )

        user_id = self._current_user_id
        session_id = self._chat_session.session_id
        try:
            blob_store = BlobStore(user_id)
            sha = blob_store.store(data, ext)
        except Exception as exc:  # noqa: BLE001
            logger.exception("blob store failed")
            self._finalize_upload_step(upload_label, summary=f"store failed: {exc}")
            self._toast(f"Could not store pasted blob: {exc}", kind="error")
            return

        mime = _guess_mime(ext)
        try:
            SessionAttachments(user_id, session_id).add(
                sha256=sha,
                filename=filename,
                mime=mime,
                size=len(data),
                source="paste",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("session attachment add failed")
            self._finalize_upload_step(upload_label, summary=f"attach failed: {exc}")
            self._toast(f"Attachment register failed: {exc}", kind="error")
            return

        # Success path: flip ⟳ → ✓ with elapsed ms and the resolved sha
        # so the row tells the user "this many bytes landed under this id".
        upload_duration_ms = int((time.monotonic() - upload_t0) * 1000)
        self._finalize_upload_step(
            upload_label,
            duration_ms=upload_duration_ms,
            summary=f"sha:{sha[:8]} ({size_label})",
        )
        logger.info(
            "upload complete filename=%s size=%d sha=%s duration_ms=%d",
            filename,
            len(data),
            sha[:8],
            upload_duration_ms,
        )

        # Resolve the on-disk content file before any UI side effect, so
        # we fail loud if the blob dir is empty for any reason.
        blob_dir = blob_store.dir(sha)
        content_path = next(
            (
                p
                for p in blob_dir.iterdir()
                if p.name.startswith("content.") and not p.name.endswith(".tmp")
            ),
            None,
        )
        if content_path is None:
            self._toast("Stored blob has no content file", kind="error")
            return

        # Input-bar placeholder. The 8-char prefix is what the user
        # sees while editing and what's persisted to chat_session.jsonl;
        # ``AttachmentsFeature.expand_placeholders`` swaps it for the
        # inline ``<image sha="<full sha>">OCR</image>`` form only at
        # prompt-assembly time, so the wire layer always carries the
        # full sha for downstream tool reference.
        is_image = _guess_mime(ext).startswith("image/")
        placeholder_kind = "Image" if is_image else "File"
        placeholder = f"[{placeholder_kind} sha:{sha[:8]}]"
        self._insert_into_input(placeholder)

        # Plain-text fast-path: csv/md/json/... don't need OCR; reading
        # the file directly is sub-millisecond, so going through the
        # worker queue and chain machinery is pure overhead. Decode
        # in-process and write the sentinel inline; downstream prompt
        # assembly reads ``ocr.md`` the same way it does for real OCR
        # output.
        if self._is_text_extension(ext):
            self._ingest_text_blob(
                blob_store=blob_store,
                user_id=user_id,
                session_id=session_id,
                sha=sha,
                ext=ext,
                content_path=content_path,
                filename=filename,
            )
            return

        worker = self._ensure_ocr_worker()
        if worker is None:
            self._toast(
                f"Pasted {filename}; OCR unavailable — see ~/.claritymed/logs/app.log",
                kind="warning",
            )
            return

        try:
            steps = self.query_one(ToolSteps)
            tool_label = _ocr_step_name(sha)
            steps.push_start(tool_label, args_preview=filename)
        except NoMatches:
            pass

        # The worker captures ContextVars via copy_context() at enqueue
        # time so its later audit_event calls (which require request_id /
        # user_id / language) attribute back to the originating action.
        # TUI events don't enter inject_context(), so we set the three
        # vars just for this call and reset immediately after.
        language = self.query_one(StatusBar).language
        tokens = apply_context(new_request_id(), user_id, language)
        try:
            worker.enqueue(
                OcrJob(
                    user_id=user_id,
                    session_id=session_id,
                    sha256=sha,
                    blob_path=content_path,
                    is_phi=True,
                    original_filename=filename,
                )
            )
        finally:
            reset_context(tokens)
        self._toast(f"Pasted {filename}; OCR queued", kind="info")

    def _insert_into_input(self, text: str) -> None:
        """Splice ``text`` at the Input cursor; non-destructive."""
        try:
            bar = self.query_one(InputBar)
        except NoMatches:
            return
        from textual.widgets import Input

        inp = bar.query_one("#input", Input)
        current = inp.value
        pos = inp.cursor_position
        new_value = current[:pos] + text + current[pos:]
        bar._suppress_next_value = new_value
        inp.value = new_value
        inp.cursor_position = pos + len(text)

    def _supported_extensions(self) -> frozenset[str] | None:
        """Cached union of supported extensions across active OCR chains.

        ``None`` means a catch-all provider is in the chain and the gate
        should not restrict the paste. A failed config load also returns
        ``None`` (fail-open): a broken ``ocr.yaml`` shouldn't block paste
        when the worker would surface the real error anyway.
        """
        cached = getattr(self, "_supported_exts_cache", None)
        if cached is not None:
            return cached[0]
        from claritymed.core.ocr.factory import chain_supported_extensions
        from claritymed.core.schemas.ocr import load_ocr_config

        try:
            cfg = load_ocr_config()
            value = chain_supported_extensions(cfg)
        except Exception:  # noqa: BLE001
            logger.exception("supported-ext gate disabled (config load failed)")
            value = None
        # Wrap in a 1-tuple to distinguish "not yet computed" (attribute
        # absent) from "computed, value is None" (catch-all / disabled).
        self._supported_exts_cache = (value,)
        return value

    def _gate_or_recover_extension(
        self, data: bytes, ext: str, display_name: str | None
    ) -> tuple[str, bool]:
        """Accept, recover, or reject a paste target by extension.

        Returns ``(ext, True)`` to proceed (possibly with a corrected
        extension) or ``(ext, False)`` after toasting the rejection.

        Decision tree:

        1. No gate (catch-all chain or config error) → accept.
        2. ``ext`` is in the accepted set → accept as-is.
        3. Run Magika on the bytes. If Magika returns an extension that
           IS in the accepted set, rewrite ``ext`` and accept; emit an
           audit event so operators can see ext-recovery is happening.
        4. Otherwise → toast and reject. No blob is written, no
           SessionAttachments row is added, no placeholder is inserted.
        """
        accepted = self._supported_extensions()
        if accepted is None:
            return ext, True
        normalized = f".{ext.lstrip('.').lower()}"
        if normalized in accepted:
            return ext, True

        # Recovery pass — only on the unsupported / fallback path so
        # the happy path stays zero-overhead.
        from claritymed.core.filetype.detector import detect

        result = detect(data)
        if result is not None and result.ext is not None and result.ext in accepted:
            recovered = result.ext.lstrip(".")
            self._emit_paste_audit(
                "filetype.detect",
                {
                    "declared_ext": normalized,
                    "detected_ext": result.ext,
                    "label": result.label,
                    "score": round(result.score, 4),
                    "outcome": "recovered",
                },
            )
            return recovered, True

        # Final rejection. Use the display name if we have it so the
        # toast is meaningful for drag-dropped files.
        label = display_name or f"file.{ext}" if ext else "clipboard data"
        self._toast(
            f"Unsupported file type for OCR: {label}",
            kind="error",
        )
        self._emit_paste_audit(
            "filetype.detect",
            {
                "declared_ext": normalized,
                "detected_ext": result.ext if result else None,
                "label": result.label if result else None,
                "score": round(result.score, 4) if result else None,
                "outcome": "rejected",
            },
        )
        return ext, False

    def _emit_paste_audit(self, event: str, payload: dict) -> None:
        """Emit an audit event from a TUI paste handler.

        Paste, drag-drop, and recovery callbacks fire outside the
        ``inject_context()`` boundary the CLI/HTTP entry points use, so
        the audit ``ContextVars`` (request_id / user_id / language) are
        unset and ``audit_event`` raises ``MissingContextError``. Apply
        them transiently around the call so audit lines attribute back
        to the originating action; reset in ``finally`` so we don't
        bleed into other widgets.

        Failures here are non-fatal — losing one audit row is strictly
        better than crashing the paste path on a missing widget mid-
        teardown.
        """
        from claritymed.core.observability.audit import audit_event

        try:
            language = self.query_one(StatusBar).language
        except Exception:  # noqa: BLE001
            language = "en"
        tokens = apply_context(new_request_id(), self._current_user_id, language)
        try:
            audit_event(event, payload)
        except Exception:  # noqa: BLE001
            logger.warning("%s audit failed", event, exc_info=True)
        finally:
            reset_context(tokens)

    def _max_paste_bytes(self) -> int:
        """Cached ``paste.max_file_size_mb`` from app.yaml, in bytes."""
        if self._max_paste_bytes_cache is None:
            self._max_paste_bytes_cache = _cfg.paste_max_file_size_bytes()
        return self._max_paste_bytes_cache

    def _size_check_or_reject(self, size: int, label: str) -> bool:
        """Reject oversized files at the entry point, before any disk read.

        Returns ``True`` when the size is within the configured ceiling.
        On rejection, emits an error toast and a warning log line so the
        user gets immediate feedback and operators can see the rejected
        attempt in ``app.log``. The toast text uses human-friendly sizes
        (KB / MB) so the limit is readable.
        """
        limit = self._max_paste_bytes()
        if size <= limit:
            return True
        self._toast(
            f"File too large: {label} is {_human_size(size)} "
            f"(limit {_human_size(limit)}). Skipped.",
            kind="error",
        )
        logger.warning(
            "upload rejected (oversize): label=%s size=%d limit=%d",
            label,
            size,
            limit,
        )
        return False

    def _next_upload_label(self) -> str:
        """Return a unique ``upload:#N`` label for the ToolSteps row.

        Counter-based instead of sha-based so we can push the ``⟳`` row
        BEFORE hashing the bytes — two concurrent uploads still resolve
        to distinct rows."""
        self._upload_seq += 1
        return f"upload:#{self._upload_seq}"

    def _finalize_upload_step(
        self,
        upload_label: str,
        *,
        duration_ms: int = 0,
        summary: str = "",
    ) -> None:
        """Flip the upload ``⟳`` row to ``✓``.

        Tolerates a missing ToolSteps widget (app tearing down) so paste
        teardown never crashes on a NoMatches lookup."""
        try:
            steps = self.query_one(ToolSteps)
            steps.push_complete(upload_label, duration_ms=duration_ms, summary=summary)
        except NoMatches:
            pass

    def _is_text_extension(self, ext: str) -> bool:
        """Cached lookup against ``OcrConfig.text_extensions``.

        Loading the YAML on every paste is fine (mtime-cached) but we
        cache the normalized set on the App instance so the lookup is
        ``O(1)`` and the next OCR-config reload picks up changes only
        when the App is restarted — paste-time behavior doesn't drift
        mid-session.
        """
        cached = getattr(self, "_text_extensions_cache", None)
        if cached is None:
            from claritymed.core.schemas.ocr import load_ocr_config

            try:
                cfg = load_ocr_config()
                cached = frozenset(cfg.text_extensions)
            except Exception:  # noqa: BLE001
                # If ocr.yaml is bad, fall back to no text fast-path
                # (everything goes through OCR worker, which surfaces the
                # config error there).
                logger.exception("ocr config load failed; text fast-path disabled")
                cached = frozenset()
            self._text_extensions_cache = cached
        return f".{ext.lstrip('.').lower()}" in cached

    def _ingest_text_blob(
        self,
        *,
        blob_store,
        user_id: str,
        session_id: str,
        sha: str,
        ext: str,
        content_path,
        filename: str,
    ) -> None:
        """Read the blob as utf-8 text, write the OCR sentinel inline,
        and fire the same completion path the worker uses.

        Errors=replace decoding: a binary file that slipped into the
        text-extensions allowlist would otherwise crash here; replacing
        invalid bytes yields a useful (if ugly) extraction the user can
        still see, which beats a silent failure.
        """
        from claritymed.orchestrator.services.ocr_worker import OcrCompleted
        from claritymed.stores.session_attachments import (
            SessionAttachments,
        )

        try:
            content_bytes = content_path.read_bytes()
            text = content_bytes.decode("utf-8", errors="replace")
        except OSError as exc:
            logger.exception("text fast-path read failed: %s", exc)
            self._toast(f"Could not read {filename}: {exc}", kind="error")
            return

        status = "done" if text.strip() else "empty"
        blob_store.write_ocr_result(
            sha,
            status=status,
            kind="text",
            ext=ext,
            provider="text",
            chain_tried=["text"],
            reason=None,
            text=text,
            original_filename=filename,
        )
        SessionAttachments(user_id, session_id).mark_ocr_status(
            sha, status, provider="text", reason=None
        )
        self._on_ocr_completed(
            OcrCompleted(
                user_id=user_id,
                session_id=session_id,
                sha256=sha,
                status=status,
                provider="text",
            )
        )
        self._toast(f"Pasted {filename}", kind="info")

    def _ensure_ocr_worker(self):
        """Lazy-build the OcrWorker so headless tests that never paste
        avoid the cost of constructing an OCR provider chain. Returns
        ``None`` if provider construction fails — the caller surfaces a
        toast."""
        worker = getattr(self, "_ocr_worker", None)
        if worker is not None:
            return worker
        try:
            from claritymed.config import CONFIGS_DIR, load_yaml
            from claritymed.core.medical_clip.client import (
                DEFAULT_BASE_URL as MEDICAL_CLIP_DEFAULT_BASE_URL,
                MedicalClipClient,
            )
            from claritymed.core.ocr.factory import make_ocr_provider
            from claritymed.core.vision.ocr_report_detector import (
                load_ocr_report_config,
            )
            from claritymed.orchestrator.services.ocr_worker import OcrWorker

            provider = make_ocr_provider()
            # Medical-clip client construction is unconditional; the
            # server may not be running, but `classify_modality` raises
            # MedicalClipUnreachableError which the worker catches and
            # tags `modality=unknown`. Building the client up-front
            # (vs lazily) keeps OcrWorker free of base_url plumbing.
            medical_clip_base_url = (
                load_yaml("app.yaml")
                .get("medical_clip", {})
                .get("base_url", MEDICAL_CLIP_DEFAULT_BASE_URL)
            )
            medical_clip_client = MedicalClipClient(base_url=medical_clip_base_url)
            ocr_report_config = load_ocr_report_config(CONFIGS_DIR / "vision.yaml")
            worker = OcrWorker(
                provider,
                listener=self._on_ocr_completed,
                medical_clip_client=medical_clip_client,
                ocr_report_config=ocr_report_config,
            )
            worker.start()
            self._ocr_worker = worker
            return worker
        except Exception as exc:  # noqa: BLE001
            logger.exception("OCR worker build failed: %s", exc)
            self._ocr_worker = None
            return None

    def _on_ocr_completed(self, completion) -> None:
        """OcrWorker listener — flips the ToolSteps row to ✓ when the job
        finishes. OcrWorker runs the listener on an ``asyncio.Task`` on
        the same event loop as the App, so direct widget access is safe
        (no ``call_from_thread`` marshal needed)."""
        tool_label = _ocr_step_name(completion.sha256)
        if completion.status == "done":
            summary = f"{completion.provider or 'ocr'} ✓"
        elif completion.status == "empty":
            summary = "no text extracted"
            logger.info("OCR empty for %s", completion.sha256[:8])
        elif completion.status == "failed":
            summary = f"failed: {completion.reason or 'unknown'}"
            logger.warning(
                "OCR failed for %s: %s",
                completion.sha256[:8],
                completion.reason or "unknown",
            )
        else:
            summary = completion.status
        try:
            steps = self.query_one(ToolSteps)
            steps.push_complete(tool_label, summary=summary)
        except NoMatches:
            # App is tearing down; widgets gone.
            pass

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
        # Use Textual's built-in notify() — it handles layering, stacking
        # and dismissal correctly. Our previous home-grown Toast widget
        # rendered at 0×0 because ``dock: bottom`` + ``width/height: auto``
        # collapsed to zero on the Screen, so users never saw the toast.
        # Errors stay visible longer than info/success — a user dropping
        # an unsupported file needs time to read the message; an "OCR
        # queued" confirmation can dismiss faster.
        severity_map = {"info": "information", "warning": "warning", "error": "error"}
        severity = severity_map.get(kind, "information")
        timeout = 8.0 if kind == "error" else 4.0
        self.notify(text, severity=severity, timeout=timeout)


def run() -> None:
    """Entry point used by the Typer ``claritymed tui`` subcommand."""
    ClarityMedApp().run()


__all__ = ["ClarityMedApp", "run"]
