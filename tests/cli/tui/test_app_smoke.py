"""Smoke tests for ``ClarityMedApp`` using Textual's ``App.run_test()``.

These tests exercise mount + widget composition + a couple of basic actions
without touching the LLM or qdrant. Service factories are injected so the
app never tries to resolve a real provider.
"""

from __future__ import annotations

import faulthandler
import sys
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from claritymed.cli.tui import ClarityMedApp
from claritymed.cli.tui.widgets import (
    Conversation,
    InputBar,
    StatusBar,
    ToolSteps,
)
from claritymed.orchestrator.services import (
    ChatSession,
    Done,
    Event,
    TokenChunk,
    ToolCompleted,
    ToolStarted,
)


def _fresh_session(user_id: str = "test") -> ChatSession:
    """A fresh ChatSession in the per-test tmp tree. Per conftest isolation
    every test gets its own DATA_DIR, so no cross-test bleed."""
    return ChatSession.new(user_id)


class _StubAskService:
    """Yields a fixed token stream and a Done so the worker drains cleanly."""

    def __init__(self, chunks: list[str], final: str = "") -> None:
        self._chunks = chunks
        self._final = final or "".join(chunks)

    async def run(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
        yield ToolStarted(tool_name="cite_source")
        for c in self._chunks:
            yield TokenChunk(text=c)
        yield ToolCompleted(tool_name="cite_source", summary="ok")
        yield Done(final=self._final)


@pytest.mark.asyncio
async def test_app_mounts_with_status_bar_and_widgets():
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
        ask_service_factory=lambda: _StubAskService(["hi"]),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one(StatusBar)
        assert status.user_id == "test"
        assert status.language == "en"
        assert status.mode == "ask"
        # All three main panes exist.
        app.query_one(Conversation)
        app.query_one(ToolSteps)
        app.query_one(InputBar)


@pytest.mark.asyncio
async def test_shift_tab_keeps_mode_on_ask():
    """The mode-cycle keybinding is preserved as a stub: only ``ask`` is
    user-reachable now, so cycling is a no-op. The keybinding stays so the
    UX hook is ready when more modes return."""
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one(StatusBar)
        assert status.mode == "ask"
        app.action_cycle_mode()
        await pilot.pause()
        assert status.mode == "ask"


@pytest.mark.asyncio
async def test_slash_help_pushes_help_modal():
    from claritymed.cli.tui.modals.help_modal import HelpModal

    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        input_bar.post_message(InputBar.Submitted("/help"))
        await pilot.pause()
        assert isinstance(app.screen, HelpModal)


@pytest.mark.asyncio
async def test_question_mark_on_empty_input_opens_help_modal():
    from claritymed.cli.tui.modals.help_modal import HelpModal

    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).focus_input()
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpModal)


@pytest.mark.asyncio
async def test_question_mark_after_text_inserts_literally():
    """Once the user has typed anything, ``?`` is a normal character — they
    might be asking a question. Only an empty bar triggers help."""
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(InputBar)
        bar.focus_input()
        await pilot.press("h", "i")
        await pilot.press("question_mark")
        await pilot.pause()
        # Help modal not pushed; the bar contains the literal ``?``.
        assert bar.value().endswith("?")


@pytest.mark.asyncio
async def test_set_mode_updates_status_bar():
    """``set_mode`` is kept as the programmatic mode entry point even though
    only ``ask`` exists today — the hook is what future modes will plug into.
    """
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.set_mode("ask")
        await pilot.pause()
        assert app.query_one(StatusBar).mode == "ask"


@pytest.mark.asyncio
async def test_ask_dispatch_streams_tokens_and_finalizes():
    """After Done, the assistant turn is swapped to a rendered Markdown widget."""
    from textual.widgets import Markdown

    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
        ask_service_factory=lambda: _StubAskService(["hel", "lo"]),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("how are you?"))
        await pilot.pause()
        for _ in range(20):
            await pilot.pause()
            if app._stream_worker is None or app._stream_worker.is_finished:
                break
        conv = app.query_one(Conversation)
        # User turn + Markdown assistant turn.
        markdown_widgets = list(conv.query(Markdown))
        assert markdown_widgets, "assistant turn should be a Markdown widget after Done"
        # The session transcript carries the assembled text.
        assistant_texts = [t.text for t in app._session_turns if t.role == "assistant"]
        assert any("hello" in t for t in assistant_texts)


@pytest.mark.asyncio
async def test_load_recent_renders_history():
    """A pre-existing session is resumed and its on-disk turns render."""
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    seed = ChatSession.new("test")
    seed.append_user("prior question")
    agent = Agent(TestModel(custom_output_text="prior answer"))
    result = await agent.run("prior question")
    from claritymed.orchestrator.services import LatencyTrace

    seed.append_assistant(
        text="prior answer",
        messages_json=result.all_messages_json(),
        model="?",
        provider_id="?",
        usage=result.usage,
        latency=LatencyTrace(total_ms=0),
    )
    resumed = ChatSession.resume("test", seed.session_id)
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=resumed,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        conv = app.query_one(Conversation)
        text_blob = " ".join(
            str(child.renderable)
            for child in conv.children
            if hasattr(child, "renderable")
        )
        assert "prior question" in text_blob
        assert "prior answer" in text_blob


@pytest.mark.asyncio
async def test_unmount_is_a_noop_now():
    """Regression: ``on_unmount`` used to call save_turns on a query that
    raised after Textual tore down child widgets. Persistence has moved
    into AskService (per-LLM-run), so the unmount path must not raise
    even with no chat_session injected."""
    app = ClarityMedApp(user_id="test", language="en")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.set_mode("ask")
        await pilot.pause()
    # Reaching here means unmount completed cleanly.
    assert True


@pytest.mark.asyncio
async def test_slash_user_switches_user_and_clears_history():
    from claritymed.stores.account import init_user

    init_user("test")  # first init → admin role
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/user bob"))
        await pilot.pause()
        status = app.query_one(StatusBar)
        assert status.user_id == "bob"
        assert app._session_turns == []


@pytest.mark.asyncio
async def test_slash_user_no_arg_opens_picker_for_admin():
    from claritymed.cli.tui.modals.user_modal import UserModal
    from claritymed.stores.account import init_user

    init_user("test")  # first init → admin
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/user"))
        await pilot.pause()
        assert isinstance(app.screen, UserModal)


@pytest.mark.asyncio
async def test_slash_user_blocked_for_non_admin():
    from claritymed.stores.account import init_user

    init_user("admin_first")  # first → admin
    init_user("regular")  # second → user
    app = ClarityMedApp(
        user_id="regular",
        language="en",
        chat_session=_fresh_session("regular"),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/user admin_first"))
        await pilot.pause()
        # Non-admin can't /user — no switch.
        assert app.query_one(StatusBar).user_id == "regular"


@pytest.mark.asyncio
async def test_slash_lang_switches_language_and_persists():
    from claritymed.stores.account import AccountStore, init_user

    init_user("test")
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/lang zh"))
        await pilot.pause()
        assert app.query_one(StatusBar).language == "zh"
        # settings.yaml mirrors the new choice so a restart picks it up.
        assert AccountStore("test").load().language == "zh"


@pytest.mark.asyncio
async def test_slash_lang_no_arg_opens_picker():
    from claritymed.cli.tui.modals.language_modal import LanguageModal
    from claritymed.stores.account import init_user

    init_user("test")
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/lang"))
        await pilot.pause()
        assert isinstance(app.screen, LanguageModal)


@pytest.mark.asyncio
async def test_slash_lang_rejects_unsupported_code():
    from claritymed.stores.account import init_user

    init_user("test")
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/lang fr"))
        await pilot.pause()
        # Unknown code is rejected — the StatusBar still shows the original lang.
        assert app.query_one(StatusBar).language == "en"


@pytest.mark.asyncio
async def test_slash_mode_is_no_longer_recognised():
    """``/mode`` was removed; submitting it now lands as an unknown command,
    not a mode switch, and the active mode is unchanged."""
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/mode rag"))
        await pilot.pause()
        assert app.query_one(StatusBar).mode == "ask"


@pytest.mark.asyncio
async def test_unknown_slash_command_shows_inline_error():
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/sproingify"))
        await pilot.pause()
        conv = app.query_one(Conversation)
        # An inline `⏺ Unknown command: /sproingify` bubble replaces the
        # earlier toast so the user can still see the typo after the fact.
        bubbles = [c for c in conv.children if c.__class__.__name__ == "TurnBubble"]
        assert len(bubbles) == 1
        rendered = str(bubbles[0].renderable)
        assert "⏺" in rendered
        assert "Unknown command: /sproingify" in rendered
        assert "error" in bubbles[0].classes


@pytest.mark.asyncio
async def test_slash_clear_rotates_session_and_keeps_old_file_on_disk():
    initial = _fresh_session()
    initial_session_id = initial.session_id
    # Pre-seed the initial session so a file exists on disk to verify
    # /clear preserves it (Claude Code semantics).
    initial.append_user("first session content")
    old_path = initial.path
    assert old_path.exists()

    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=initial,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/clear"))
        await pilot.pause()
        # New session_id was minted
        assert app._chat_session is not None
        assert app._chat_session.session_id != initial_session_id
        # Old file is untouched
        assert old_path.exists()
        assert "first session content" in old_path.read_text("utf-8")
        # In-memory history reset
        assert app._chat_session.message_history() == []
        # Visible conversation reset to the empty-state placeholder
        conv = app.query_one(Conversation)
        assert all("empty" in child.classes for child in conv.children)


@pytest.mark.asyncio
async def test_quit_command_exits_app():
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/quit"))
        await pilot.pause()
        # The app should have queued an exit; subsequent pause drains it.
        # We just assert the call didn't raise.


@pytest.mark.asyncio
async def test_f2_toggles_steps_panel():
    """F2 adds/removes user_collapsed on ToolSteps without clearing content."""
    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
        ask_service_factory=lambda: _StubAskService(["hi"]),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        steps = app.query_one(ToolSteps)
        # Panel starts without the collapsed class.
        assert "user_collapsed" not in steps.classes
        app.action_toggle_steps()
        await pilot.pause()
        assert "user_collapsed" in steps.classes
        # Toggle again restores it.
        app.action_toggle_steps()
        await pilot.pause()
        assert "user_collapsed" not in steps.classes


@pytest.mark.asyncio
async def test_streaming_label_cleared_after_done():
    """'streaming…' on the llm-first-token step is replaced with 'done'
    once the stream worker finishes (Done event received)."""
    from claritymed.orchestrator.services import LlmFirstToken

    class _StreamingStubService:
        async def run(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
            yield LlmFirstToken(ttft_ms=100)
            yield TokenChunk(text="answer")
            yield Done(final="answer")

    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
        ask_service_factory=lambda: _StreamingStubService(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("q"))
        for _ in range(20):
            await pilot.pause()
            if app._stream_worker is None or app._stream_worker.is_finished:
                break
        steps = app.query_one(ToolSteps)
        step_text = " ".join(
            str(child.renderable)
            for child in steps.children
            if hasattr(child, "renderable")
        )
        assert "streaming…" not in step_text
        assert "done" in step_text


# --- _strategy_for_session unit tests (no Textual event loop needed) ---


def test_strategy_for_session_returns_cached():
    """Short-circuits to the cached value without re-acquiring the lock."""
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    sentinel = object()
    app._cached_strategy = sentinel
    assert app._strategy_for_session() is sentinel


def test_strategy_for_session_rag_disabled(monkeypatch):
    """Returns None immediately when rag.enabled=False, then releases the lock."""
    mock_cfg = MagicMock()
    mock_cfg.rag.enabled = False
    monkeypatch.setattr("claritymed.core.rag.load_retrieval_config", lambda: mock_cfg)

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    assert app._strategy_for_session() is None
    assert not app._strategy_lock.locked()


def test_strategy_for_session_lock_timeout():
    """Raises RuntimeError when the lock cannot be acquired within the timeout."""
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    mock_lock = MagicMock()
    mock_lock.acquire.return_value = False
    app._strategy_lock = mock_lock

    with pytest.raises(RuntimeError, match="strategy lock timed out"):
        app._strategy_for_session()


@pytest.mark.asyncio
async def test_on_mount_faulthandler_exception_silenced(monkeypatch):
    """faulthandler.register failures during mount are silently swallowed."""
    if sys.__stderr__ is None:
        pytest.skip("sys.__stderr__ is None in this environment")

    def _bad_register(*args, **kwargs):
        raise RuntimeError("simulated bad file descriptor")

    monkeypatch.setattr(faulthandler, "register", _bad_register)
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(StatusBar)  # mount completed despite faulthandler failure


# ---- clipboard paste action ---------------------------------------------


def _patch_clipboard(monkeypatch, content):
    """Force ``read_clipboard`` to return ``content`` for this test."""
    monkeypatch.setattr(
        "claritymed.cli.tui.paste.read_clipboard", lambda: content, raising=True
    )


class _SyncOcrWorker:
    """Stand-in for OcrWorker that records enqueues without spawning a task."""

    def __init__(self) -> None:
        self.enqueued = []

    def enqueue(self, job) -> None:
        self.enqueued.append(job)

    def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.mark.asyncio
async def test_paste_image_routes_through_blob_and_session(monkeypatch):
    """ImageBytes → BlobStore.store + SessionAttachments row + OCR enqueue."""
    from claritymed.cli.tui.paste import ImageBytes
    from claritymed.orchestrator.services.session_attachments import (
        SessionAttachments,
    )
    from claritymed.stores.blob_store import BlobStore

    _patch_clipboard(monkeypatch, ImageBytes(bytes=b"fake-png-bytes", ext="png"))

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        # Skip real OCR provider construction by pre-seeding the slot.
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()

        # Blob landed: the sha is whatever sha256(b"fake-png-bytes") resolves to.
        rows = SessionAttachments("test", app._chat_session.session_id).list()
        assert len(rows) == 1
        row = rows[0]
        assert row.filename == "clipboard.png"
        assert row.mime == "image/png"
        # Blob is on disk under the per-user CAS pool.
        assert BlobStore("test").path(row.sha256, "png").exists()
        # OCR job was enqueued with matching identifiers.
        assert len(fake_worker.enqueued) == 1
        job = fake_worker.enqueued[0]
        assert job.user_id == "test"
        assert job.sha256 == row.sha256


@pytest.mark.asyncio
async def test_paste_small_text_inserts_into_input(monkeypatch):
    from claritymed.cli.tui.paste import SmallText

    _patch_clipboard(monkeypatch, SmallText(text="hello world"))

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_paste_clipboard()
        await pilot.pause()
        assert "hello world" in app.query_one(InputBar).value()


@pytest.mark.asyncio
async def test_paste_empty_clipboard_emits_toast(monkeypatch):
    from claritymed.cli.tui.paste import Empty

    _patch_clipboard(monkeypatch, Empty())

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        # Action should run without exception even though there's nothing to do.
        app.action_paste_clipboard()
        await pilot.pause()
        # Input is unchanged.
        assert app.query_one(InputBar).value() == ""


@pytest.mark.asyncio
async def test_paste_file_path_routes_through_blob_and_session(monkeypatch, tmp_path):
    from claritymed.cli.tui.paste import FilePath
    from claritymed.orchestrator.services.session_attachments import (
        SessionAttachments,
    )

    sample = tmp_path / "report.pdf"
    sample.write_bytes(b"%PDF-1.4 fake")
    _patch_clipboard(monkeypatch, FilePath(path=sample))

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()
        rows = SessionAttachments("test", app._chat_session.session_id).list()
        assert len(rows) == 1
        assert rows[0].filename == "report.pdf"
        assert rows[0].mime == "application/pdf"
        assert len(fake_worker.enqueued) == 1


@pytest.mark.asyncio
async def test_paste_text_file_uses_fast_path_no_ocr_worker(monkeypatch, tmp_path):
    """Pasting a .csv (text extension) bypasses the OCR worker entirely:
    the sentinel lands inline with provider='text' and the worker queue
    stays empty.

    Regression guard: before the fast-path, every paste — including
    plain-text files — went through the worker chain, wasting an LLM
    OCR call on something we could read in microseconds.
    """
    import json

    from claritymed.cli.tui.paste import FilePath
    from claritymed.orchestrator.services.session_attachments import (
        SessionAttachments,
    )
    from claritymed.stores.blob_store import BlobStore

    sample = tmp_path / "data.csv"
    sample.write_text("col1,col2\n1,2\n3,4\n", encoding="utf-8")
    _patch_clipboard(monkeypatch, FilePath(path=sample))

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()

        rows = SessionAttachments("test", app._chat_session.session_id).list()
        assert len(rows) == 1
        sha = rows[0].sha256

        # Worker queue was NOT touched — the whole point of the fast-path.
        assert fake_worker.enqueued == []

        # Sentinel landed inline with kind="text" + ext="csv" so the
        # reader knows to consult content.csv directly.
        bs = BlobStore("test")
        sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
        assert sentinel["status"] == "done"
        assert sentinel["kind"] == "text"
        assert sentinel["ext"] == "csv"
        assert sentinel["provider"] == "text"
        assert sentinel["chain_tried"] == ["text"]
        # ocr.md is NOT written — content.csv already has the text.
        assert not bs.ocr_path(sha).exists()
        # The unified reader returns the original file contents.
        assert bs.read_extracted_text(sha).startswith("col1,col2")


@pytest.mark.asyncio
async def test_paste_clipboard_read_failure_emits_toast(monkeypatch):
    """An exception from ``read_clipboard`` becomes an error toast, not a crash."""

    def _boom():
        raise RuntimeError("xclip segfault")

    monkeypatch.setattr("claritymed.cli.tui.paste.read_clipboard", _boom, raising=True)
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_paste_clipboard()  # must not raise
        await pilot.pause()


# --- Paste-time supported-ext gate ------------------------------------


@pytest.mark.asyncio
async def test_paste_unsupported_ext_rejected_before_blob_or_attachment(
    monkeypatch, tmp_path
):
    """A file whose extension no OCR provider claims is rejected at paste
    time: no blob written, no SessionAttachments row, no placeholder
    inserted, and a Magika recovery pass that fails to find a supported
    type leaves the rejection in place."""
    from claritymed.cli.tui.paste import FilePath
    from claritymed.orchestrator.services.session_attachments import (
        SessionAttachments,
    )
    from claritymed.stores.blob_store import BlobStore

    # Pure-random bytes — neither a real extension nor a Magika-recoverable
    # type, so neither the declared ``.bin`` nor any detected ext lands
    # in the supported set.
    sample = tmp_path / "random.bin"
    sample.write_bytes(b"\x00\x01\x02\x03random-noise")
    _patch_clipboard(monkeypatch, FilePath(path=sample))

    # Force Magika to "couldn't recover" so the test isolates the gate.
    monkeypatch.setattr("claritymed.core.filetype.detector.detect", lambda _data: None)

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()

        rows = SessionAttachments("test", app._chat_session.session_id).list()
        assert rows == []
        assert fake_worker.enqueued == []
        # Input bar got no placeholder injected.
        assert app.query_one(InputBar).value() == ""
        # And no blob landed on disk for these bytes.
        bs = BlobStore("test")
        import hashlib

        sha = hashlib.sha256(b"\x00\x01\x02\x03random-noise").hexdigest()
        assert not bs.exists(sha)


@pytest.mark.asyncio
async def test_paste_unsupported_ext_surfaces_visible_toast(monkeypatch, tmp_path):
    """Regression: the rejection path must produce a real notification.

    The previous home-grown Toast widget rendered at 0x0 because
    ``dock: bottom`` + ``width/height: auto`` collapsed on the Screen,
    so the audit log got the ``filetype.detect outcome=rejected`` line
    but the user saw nothing. We now route through Textual's
    ``App.notify()`` — assert the notification is queued with the
    expected message, severity, and the longer 8 s timeout reserved
    for errors.
    """
    from claritymed.cli.tui.paste import FilePath

    sample = tmp_path / "doc.dvi"
    sample.write_bytes(b"\xf7\x02fakedvi" + b"\x00" * 64)
    _patch_clipboard(monkeypatch, FilePath(path=sample))
    monkeypatch.setattr("claritymed.core.filetype.detector.detect", lambda _data: None)

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()

        notifications = list(app._notifications)
        rejections = [
            n for n in notifications if "Unsupported file type for OCR" in n.message
        ]
        assert len(rejections) == 1, (
            f"expected exactly one rejection notification, got: "
            f"{[n.message for n in notifications]}"
        )
        assert "doc.dvi" in rejections[0].message
        assert rejections[0].severity == "error"
        assert rejections[0].timeout == 8.0


@pytest.mark.asyncio
async def test_paste_magika_recovers_extension_for_mislabeled_file(
    monkeypatch, tmp_path
):
    """A PNG renamed to ``.bin`` is recovered by Magika: the ext gets
    rewritten to ``.png`` and ingestion proceeds normally."""
    from claritymed.cli.tui.paste import FilePath
    from claritymed.core.filetype.detector import DetectResult
    from claritymed.orchestrator.services.session_attachments import (
        SessionAttachments,
    )

    sample = tmp_path / "mislabeled.bin"
    sample.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-payload")
    _patch_clipboard(monkeypatch, FilePath(path=sample))

    # Stub Magika to deterministically return ".png" — avoids depending
    # on the real model's confidence on this short payload.
    monkeypatch.setattr(
        "claritymed.core.filetype.detector.detect",
        lambda _data: DetectResult(
            ext=".png", label="png", score=0.99, mime_type="image/png"
        ),
    )

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()

        rows = SessionAttachments("test", app._chat_session.session_id).list()
        assert len(rows) == 1
        # Filename keeps the user's original (``mislabeled.bin``) but
        # the stored MIME and blob extension reflect the recovered type.
        assert rows[0].mime == "image/png"
        assert len(fake_worker.enqueued) == 1


def test_parse_dropped_paths_single(tmp_path):
    """Bare absolute path → one Path entry."""
    from claritymed.cli.tui.app import _parse_dropped_paths

    f = tmp_path / "report.pdf"
    f.write_bytes(b"x")
    assert _parse_dropped_paths(str(f)) == [f]


def test_parse_dropped_paths_quoted_and_escaped(tmp_path):
    """Outer quotes stripped, ``\\ `` un-escaped — matches what Ghostty
    injects when a path contains spaces."""
    from claritymed.cli.tui.app import _parse_dropped_paths

    f = tmp_path / "lab report.pdf"
    f.write_bytes(b"x")
    quoted = f'"{f}"'
    escaped = str(f).replace(" ", "\\ ")
    assert _parse_dropped_paths(quoted) == [f]
    assert _parse_dropped_paths(escaped) == [f]


def test_parse_dropped_paths_multi_space_separated(tmp_path):
    """Ghostty 1.1+ joins multi-file drops with a single space, but
    spaces inside a single path are escaped — split only on space when
    followed by an absolute-path marker."""
    from claritymed.cli.tui.app import _parse_dropped_paths

    a = tmp_path / "a.pdf"
    a.write_bytes(b"x")
    b = tmp_path / "b.png"
    b.write_bytes(b"y")
    assert _parse_dropped_paths(f"{a} {b}") == [a, b]


def test_parse_dropped_paths_newline_separated(tmp_path):
    """iTerm2 joins multi-file drops with newlines."""
    from claritymed.cli.tui.app import _parse_dropped_paths

    a = tmp_path / "a.pdf"
    a.write_bytes(b"x")
    b = tmp_path / "b.png"
    b.write_bytes(b"y")
    assert _parse_dropped_paths(f"{a}\n{b}") == [a, b]


def test_parse_dropped_paths_falls_back_for_non_paths():
    """Plain text (no file on disk) returns empty so the caller can
    fall through to the default Input insert-as-text behaviour."""
    from claritymed.cli.tui.app import _parse_dropped_paths

    assert _parse_dropped_paths("hello world") == []
    assert _parse_dropped_paths("/nonexistent/path.pdf") == []


@pytest.mark.asyncio
async def test_on_paste_routes_dropped_pdf_to_ingest(tmp_path):
    """Dragging a PDF into the TUI fires events.Paste with the path text;
    on_paste should swap that for a [File sha:…] placeholder and enqueue
    OCR — same outcome as Ctrl+V with FilePath clipboard content."""
    from textual import events

    from claritymed.orchestrator.services.session_attachments import (
        SessionAttachments,
    )

    pdf = tmp_path / "labs.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = fake_worker
        app.on_paste(events.Paste(str(pdf)))
        await pilot.pause()

        rows = SessionAttachments("test", app._chat_session.session_id).list()
        assert len(rows) == 1
        assert rows[0].filename == "labs.pdf"
        assert rows[0].mime == "application/pdf"
        assert len(fake_worker.enqueued) == 1
        # Path text did NOT land in the Input — the placeholder did.
        assert str(pdf) not in app.query_one(InputBar).value()
        assert app.query_one(InputBar).value().startswith("[File sha:")


@pytest.mark.asyncio
async def test_dispatch_preserves_in_flight_ocr_row(monkeypatch):
    """Submitting a message while an OCR job is still ⟳ in-flight must
    keep that row alive — wiping it makes the right panel collapse and
    the eventual ✓ row reappears as a stray completion. Regression for
    the 'right window disappears, then reappears' UX bug."""
    from claritymed.cli.tui.paste import ImageBytes
    from claritymed.orchestrator.services.ocr_worker import OcrCompleted

    _patch_clipboard(monkeypatch, ImageBytes(bytes=b"png-bytes", ext="png"))

    app = ClarityMedApp(
        user_id="test",
        language="en",
        chat_session=_fresh_session(),
        ask_service_factory=lambda: _StubAskService(["ok"]),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = _SyncOcrWorker()
        app.action_paste_clipboard()
        await pilot.pause()

        steps = app.query_one(ToolSteps)
        [step_key] = [k for k in steps._active.keys() if k.startswith("ocr:")]
        in_flight_widget = steps._active[step_key]
        assert steps.has_class("has_events")

        # Submit a turn while OCR is still pending. The dispatch path
        # used to call reset() without preserve_active and wipe the row.
        # We assert state immediately so the stream worker (kicked off
        # by dispatch) hasn't had time to add its own rows yet.
        app._dispatch_to_service("what does the image say?")
        # In-flight row is still mounted and the panel stays visible.
        assert step_key in steps._active
        assert in_flight_widget in steps.children
        assert steps.has_class("has_events")

        # OCR finishes mid-conversation → ⟳ flips to ✓ in place, not a
        # second row that re-shows the panel.
        sha_prefix = step_key.split(":", 1)[1]
        app._on_ocr_completed(
            OcrCompleted(
                user_id="test",
                session_id=app._chat_session.session_id,
                sha256=sha_prefix + "0" * (64 - len(sha_prefix)),
                status="done",
                provider="PyMuPDFOcrProvider",
            )
        )
        await pilot.pause()
        assert step_key not in steps._active
        # The same widget object got updated in place.
        assert in_flight_widget in steps.children


@pytest.mark.asyncio
async def test_on_paste_plain_text_falls_through(tmp_path):
    """Pasted text that isn't a file path leaves the event unhandled so
    Textual's default Input handler inserts it as normal text."""
    from textual import events

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        ev = events.Paste("not a path, just text")
        app.on_paste(ev)
        await pilot.pause()
        # The handler must not consume non-path pastes — otherwise
        # ordinary text pastes silently disappear into the void.
        assert not ev._stop_propagation


@pytest.mark.asyncio
async def test_paste_short_multiline_inlines_with_newlines():
    """A short multi-line paste (under both placeholder thresholds)
    inserts as-is — newlines preserved, no fold to placeholder. The user
    sees what they pasted and can edit it in place."""
    from textual import events
    from textual.widgets import TextArea

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(InputBar)
        bar.focus_input()
        ta = bar.query_one("#input", TextArea)
        text = "line one\nline two\nline three"  # 3 lines, 28 chars
        ta.post_message(events.Paste(text))
        await pilot.pause()
        assert ta.text == text


@pytest.mark.asyncio
async def test_paste_long_multiline_folds_to_placeholder():
    """Paste with >= ``placeholder_min_lines`` (default 6) collapses to a
    ``[Pasted text #N +M lines]`` visual placeholder. The body is stashed
    on the InputBar and ``expand_pastes`` reconstitutes the full text at
    submit time."""
    from textual import events
    from textual.widgets import TextArea

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(InputBar)
        bar.focus_input()
        ta = bar.query_one("#input", TextArea)
        body = "\n".join(f"line {i}" for i in range(20))
        ta.post_message(events.Paste(body))
        await pilot.pause()
        assert ta.text == "[Pasted text #1 +20 lines]"
        assert bar.expand_pastes(ta.text) == body


@pytest.mark.asyncio
async def test_paste_long_single_line_folds_to_placeholder():
    """A single long line (>= ``placeholder_min_chars``, default 800)
    also folds — without this, a 5000-char URL or token blob would
    horizontally scroll the input forever."""
    from textual import events
    from textual.widgets import TextArea

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(InputBar)
        bar.focus_input()
        ta = bar.query_one("#input", TextArea)
        body = "x" * 1000  # 1 line, 1000 chars
        ta.post_message(events.Paste(body))
        await pilot.pause()
        # Single line → no "+M lines" suffix
        assert ta.text == "[Pasted text #1]"
        assert bar.expand_pastes(ta.text) == body


@pytest.mark.asyncio
async def test_paste_text_over_limit_rejected():
    """A paste larger than ``paste.max_text_chars`` must not land in the
    TextArea. The handler zeroes ``event.text`` and toasts so the user
    knows the paste was dropped on purpose, not silently lost."""
    from textual import events
    from textual.widgets import TextArea

    from claritymed import config as _cfg

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        limit = _cfg.paste_max_text_chars()
        bar = app.query_one(InputBar)
        bar.focus_input()
        ta = bar.query_one("#input", TextArea)
        ta.post_message(events.Paste("x" * (limit + 1)))
        await pilot.pause()
        assert ta.text == ""


@pytest.mark.asyncio
async def test_clear_resets_paste_stash():
    """``InputBar.clear()`` wipes both the visible text and the paste
    stash so the next turn starts from id #1 — protects against id reuse
    across turns and frees the stashed bodies."""
    from textual import events
    from textual.widgets import TextArea

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        bar = app.query_one(InputBar)
        bar.focus_input()
        ta = bar.query_one("#input", TextArea)
        body = "\n".join(f"line {i}" for i in range(10))
        ta.post_message(events.Paste(body))
        await pilot.pause()
        assert ta.text == "[Pasted text #1 +10 lines]"
        bar.clear()
        await pilot.pause()
        assert ta.text == ""
        assert bar._pastes == {}
        assert bar._next_paste_id == 0
        # Next paste starts the counter at #1 again.
        ta.post_message(events.Paste(body))
        await pilot.pause()
        assert ta.text == "[Pasted text #1 +10 lines]"


def test_looks_like_drop_attempt_recognises_unix_and_windows():
    """Path-prefix tokens (``/``, ``~``, ``C:\\``) signal a real drag-drop
    attempt; bare text does not."""
    from claritymed.cli.tui.app import _looks_like_drop_attempt

    assert _looks_like_drop_attempt("/Users/alice/Desktop/report.pdf")
    assert _looks_like_drop_attempt("~/lab.pdf")
    assert _looks_like_drop_attempt(r"C:\Users\alice\report.pdf")
    # Multi-drop: middle/end path token still counts.
    assert _looks_like_drop_attempt("/a/b.pdf /c/d.png")
    # Bare text → not a drop attempt → caller falls through to Input.
    assert not _looks_like_drop_attempt("not a path")
    assert not _looks_like_drop_attempt("")
    assert not _looks_like_drop_attempt("   ")


def test_looks_like_drop_attempt_handles_quoted_paths():
    """macOS Finder and several Linux desktops wrap dropped paths in
    quotes. The heuristic must see through the outer quote to recognise
    the path inside."""
    from claritymed.cli.tui.app import _looks_like_drop_attempt

    assert _looks_like_drop_attempt('"/Users/alice/my report.pdf"')
    assert _looks_like_drop_attempt("'/Users/alice/lab.zip'")
    # Empty quoted string is NOT a drop attempt.
    assert not _looks_like_drop_attempt('""')


def test_looks_like_drop_attempt_handles_file_uri():
    """GNOME / KDE Wayland drag-drop sometimes emits ``file://`` URIs."""
    from claritymed.cli.tui.app import _looks_like_drop_attempt

    assert _looks_like_drop_attempt("file:///home/alice/report.pdf")
    assert _looks_like_drop_attempt("file://localhost/home/alice/report.pdf")


def test_decode_file_uri_returns_plain_path():
    """``file://`` is stripped and percent-encoded chars are decoded."""
    from claritymed.cli.tui.app import _decode_file_uri

    assert _decode_file_uri("file:///home/alice/lab.pdf") == "/home/alice/lab.pdf"
    # Percent-encoded space (``%20``) decodes back to a real space.
    assert (
        _decode_file_uri("file:///home/alice/my%20report.pdf")
        == "/home/alice/my report.pdf"
    )
    # Pass-through for plain paths.
    assert _decode_file_uri("/home/alice/lab.pdf") == "/home/alice/lab.pdf"


def test_parse_dropped_paths_accepts_file_uri(tmp_path):
    """A ``file://`` URI pointing at a real file resolves like the plain
    form so drag-drop from GNOME works."""
    from claritymed.cli.tui.app import _parse_dropped_paths

    f = tmp_path / "report.pdf"
    f.write_bytes(b"x")
    uri = f"file://{f}"
    assert _parse_dropped_paths(uri) == [f]


@pytest.mark.asyncio
async def test_on_paste_dropped_folder_surfaces_error_toast(tmp_path):
    """A dropped folder (or any path that does not resolve to a real
    file) leaves the user with no on-screen reaction in the prior
    behaviour. Now ``on_paste`` consumes the event and shows a toast so
    "I dragged something and nothing happened" stops being a thing."""
    from textual import events

    folder = tmp_path / "case_files"
    folder.mkdir()

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        ev = events.Paste(str(folder))
        app.on_paste(ev)
        await pilot.pause()
        # Path token present → handler consumes the event so the raw
        # folder path doesn't leak into the Input as text.
        assert ev._stop_propagation


@pytest.mark.asyncio
async def test_paste_inserts_sha_placeholder_and_tool_step(monkeypatch):
    """Image paste inserts ``[Image sha:<8-char>]`` at the input cursor
    AND pushes a ToolSteps row. No chat-history line — the right pane
    carries the OCR status. The sha label matches AttachmentsFeature's
    ``(sha <8>)`` form so the user/jsonl/LLM all see the same identifier.
    """
    import hashlib

    from claritymed.cli.tui.paste import ImageBytes
    from claritymed.cli.tui.widgets.input_bar import InputBar

    payload = b"png-bytes"
    sha = hashlib.sha256(payload).hexdigest()
    _patch_clipboard(monkeypatch, ImageBytes(bytes=payload, ext="png"))

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()

        assert app.query_one(InputBar).value() == f"[Image sha:{sha[:8]}]"
        # No system turn for the attachment — chat history is intentionally clean.
        system_turns = [t for t in app._session_turns if t.role == "system"]
        assert not any("attached" in t.text for t in system_turns), [
            t.text for t in system_turns
        ]
        # ToolSteps has a ⟳ row for the OCR job.
        steps = app.query_one(ToolSteps)
        active_keys = list(steps._active.keys())
        assert any(k.startswith("ocr:") for k in active_keys), active_keys


@pytest.mark.asyncio
async def test_paste_two_distinct_images_get_two_sha_placeholders(monkeypatch):
    """Different image bytes hash to different shas → distinct placeholders.
    Same image pasted twice would just append the same sha label twice —
    that's fine, the AttachmentsFeature row is dedup'd on sha.
    """
    import hashlib

    from claritymed.cli.tui.paste import ImageBytes
    from claritymed.cli.tui.widgets.input_bar import InputBar

    payload_a = b"png-bytes-A"
    payload_b = b"png-bytes-B"
    sha_a = hashlib.sha256(payload_a).hexdigest()[:8]
    sha_b = hashlib.sha256(payload_b).hexdigest()[:8]

    contents = iter(
        [
            ImageBytes(bytes=payload_a, ext="png"),
            ImageBytes(bytes=payload_b, ext="png"),
        ]
    )
    monkeypatch.setattr(
        "claritymed.cli.tui.paste.read_clipboard",
        lambda: next(contents),
        raising=True,
    )

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()
        app.action_paste_clipboard()
        await pilot.pause()
        assert (
            app.query_one(InputBar).value() == f"[Image sha:{sha_a}][Image sha:{sha_b}]"
        )


@pytest.mark.asyncio
async def test_paste_still_inserts_placeholder_when_ocr_worker_unavailable(monkeypatch):
    """OCR factory failure (e.g. mineru gated) must not swallow the paste.
    The placeholder still lands in the input — only the right-pane row is
    skipped because there is no worker to track.
    """
    import hashlib

    from claritymed.cli.tui.paste import ImageBytes
    from claritymed.cli.tui.widgets.input_bar import InputBar

    payload = b"png-bytes"
    sha = hashlib.sha256(payload).hexdigest()
    _patch_clipboard(monkeypatch, ImageBytes(bytes=payload, ext="png"))

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_ensure_ocr_worker", lambda: None)
        app.action_paste_clipboard()
        await pilot.pause()
        assert app.query_one(InputBar).value() == f"[Image sha:{sha[:8]}]"


@pytest.mark.asyncio
async def test_paste_pdf_uses_file_sha_placeholder(monkeypatch, tmp_path):
    """Non-image attachments get ``[File sha:<8>]`` so the user/jsonl/LLM
    can tell image vs file at a glance while still keyed on sha."""
    import hashlib

    from claritymed.cli.tui.paste import FilePath
    from claritymed.cli.tui.widgets.input_bar import InputBar

    payload = b"%PDF-1.4 fake"
    sample = tmp_path / "report.pdf"
    sample.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    _patch_clipboard(monkeypatch, FilePath(path=sample))

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()
        assert app.query_one(InputBar).value() == f"[File sha:{sha[:8]}]"


@pytest.mark.asyncio
async def test_ocr_completed_marks_tool_step_done(monkeypatch):
    """The OcrWorker listener flips the ToolSteps row to ✓ on completion.

    Drives ``_on_ocr_completed`` directly to verify the status → summary
    branching (done / empty / failed / unknown all reach the same widget
    call path)."""
    from claritymed.cli.tui.paste import ImageBytes
    from claritymed.orchestrator.services.ocr_worker import OcrCompleted

    _patch_clipboard(monkeypatch, ImageBytes(bytes=b"png-bytes", ext="png"))

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = _SyncOcrWorker()
        app.action_paste_clipboard()
        await pilot.pause()

        # Grab the sha by inspecting the active steps key.
        steps = app.query_one(ToolSteps)
        [step_key] = [k for k in steps._active.keys() if k.startswith("ocr:")]
        sha_prefix = step_key.split(":", 1)[1]

        # Simulate worker completion for that sha.
        app._on_ocr_completed(
            OcrCompleted(
                user_id="test",
                session_id=app._chat_session.session_id,
                sha256=sha_prefix + "0" * (64 - len(sha_prefix)),
                status="done",
                provider="PyMuPDFOcrProvider",
            )
        )
        await pilot.pause()
        # Row moved from _active to completed (no longer in _active).
        assert step_key not in steps._active


@pytest.mark.asyncio
async def test_ocr_completed_failed_status_renders_reason(monkeypatch):
    from claritymed.orchestrator.services.ocr_worker import OcrCompleted

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        steps = app.query_one(ToolSteps)
        sha = "a" * 64
        steps.push_start(f"ocr:{sha[:8]}", args_preview="x.png")
        app._on_ocr_completed(
            OcrCompleted(
                user_id="test",
                session_id=app._chat_session.session_id,
                sha256=sha,
                status="failed",
                reason="tesseract missing",
            )
        )
        await pilot.pause()
        # Step row content should mention the failure reason.
        from textual.widgets import Static

        statics = list(steps.query(Static))
        assert any("tesseract missing" in str(s.renderable) for s in statics)


@pytest.mark.asyncio
async def test_paste_without_chat_session_emits_toast(monkeypatch):
    """No active session → paste refused with an error toast, not a crash."""
    from claritymed.cli.tui.paste import ImageBytes

    _patch_clipboard(monkeypatch, ImageBytes(bytes=b"abc", ext="png"))

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._chat_session = None
        app.action_paste_clipboard()
        await pilot.pause()
        # Did not crash. _session_turns has no attachment row.
        assert all("attached" not in t.text for t in app._session_turns)


@pytest.mark.asyncio
async def test_paste_image_oversize_rejected_before_blob_store(monkeypatch):
    """ImageBytes larger than paste.max_file_size_mb is rejected at the
    entry point — no blob is written, no SessionAttachments row is added,
    no OCR is enqueued. The user sees an error toast instead."""
    from claritymed.cli.tui.paste import ImageBytes
    from claritymed.orchestrator.services.session_attachments import (
        SessionAttachments,
    )

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    # Force a tiny 1KB cap so test bytes don't have to be huge.
    payload = b"x" * 2048
    _patch_clipboard(monkeypatch, ImageBytes(bytes=payload, ext="png"))

    async with app.run_test() as pilot:
        await pilot.pause()
        app._max_paste_bytes_cache = 1024
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()

        rows = SessionAttachments("test", app._chat_session.session_id).list()
        assert rows == []
        assert fake_worker.enqueued == []


@pytest.mark.asyncio
async def test_paste_file_path_oversize_rejected_before_read(monkeypatch, tmp_path):
    """FilePath that stat()s above the cap is rejected without
    read_bytes() — protects the event loop from a multi-second read."""
    from claritymed.cli.tui.paste import FilePath
    from claritymed.orchestrator.services.session_attachments import (
        SessionAttachments,
    )

    sample = tmp_path / "huge.pdf"
    sample.write_bytes(b"%PDF " + b"x" * 4096)
    _patch_clipboard(monkeypatch, FilePath(path=sample))

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._max_paste_bytes_cache = 1024
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()

        rows = SessionAttachments("test", app._chat_session.session_id).list()
        assert rows == []
        assert fake_worker.enqueued == []


@pytest.mark.asyncio
async def test_drag_drop_oversize_rejected_before_read(tmp_path):
    """on_paste sizes each dropped path via stat() and skips oversize
    ones — the file is never read into memory and no row is added."""
    from textual import events

    from claritymed.orchestrator.services.session_attachments import (
        SessionAttachments,
    )

    big = tmp_path / "huge.pdf"
    big.write_bytes(b"%PDF " + b"y" * 8192)

    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._max_paste_bytes_cache = 1024
        app._ocr_worker = _SyncOcrWorker()
        app.on_paste(events.Paste(str(big)))
        await pilot.pause()

        rows = SessionAttachments("test", app._chat_session.session_id).list()
        assert rows == []


@pytest.mark.asyncio
async def test_paste_image_pushes_upload_step_row(monkeypatch):
    """Successful paste leaves a completed ✓ upload row in the right
    panel — visible record that the bytes landed."""
    from claritymed.cli.tui.paste import ImageBytes
    from textual.widgets import Static

    _patch_clipboard(monkeypatch, ImageBytes(bytes=b"ok-bytes", ext="png"))

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="test", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()

        steps = app.query_one(ToolSteps)
        rendered = [str(s.renderable) for s in steps.query(Static)]
        upload_rows = [r for r in rendered if "upload:#" in r]
        # Exactly one upload row, marked complete (✓), referencing the sha.
        assert len(upload_rows) == 1
        assert upload_rows[0].startswith("✓ upload:#")
        assert "sha:" in upload_rows[0]


@pytest.mark.asyncio
async def test_upload_modal_rejects_oversize_file(tmp_path, monkeypatch):
    """/upload modal stat()s the file before read_text(); oversize files
    surface an error inline without dismissing the modal."""
    from claritymed.cli.tui.modals import UploadModal

    sample = tmp_path / "big.md"
    sample.write_bytes(b"y" * 4096)
    # Shrink the cap globally so the read never runs.
    monkeypatch.setattr("claritymed.config.paste_max_file_size_bytes", lambda: 1024)

    class _Host(__import__("textual.app", fromlist=["App"]).App):
        def __init__(self):
            super().__init__()
            self.result = "<unset>"

        def on_mount(self):
            self.push_screen(
                UploadModal(initial_path=str(sample)),
                lambda v: setattr(self, "result", v),
            )

    from textual.widgets import Button, Static

    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#confirm", Button).press()
        await pilot.pause()
        # Modal stayed up (no dismiss with payload).
        assert app.result == "<unset>"
        err = modal.query_one("#error", Static)
        assert "too large" in str(err.renderable).lower()
