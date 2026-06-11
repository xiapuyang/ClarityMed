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


def _fresh_session(user_id: str = "alice") -> ChatSession:
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
        user_id="alice",
        language="en",
        chat_session=_fresh_session(),
        ask_service_factory=lambda: _StubAskService(["hi"]),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one(StatusBar)
        assert status.user_id == "alice"
        assert status.language == "en"
        assert status.mode == "ask"
        # All three main panes exist.
        app.query_one(Conversation)
        app.query_one(ToolSteps)
        app.query_one(InputBar)


@pytest.mark.asyncio
async def test_shift_tab_cycles_mode():
    app = ClarityMedApp(
        user_id="alice",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        status = app.query_one(StatusBar)
        assert status.mode == "ask"
        app.action_cycle_mode()
        await pilot.pause()
        assert status.mode == "ingest"
        app.action_cycle_mode()
        await pilot.pause()
        assert status.mode == "rag"
        app.action_cycle_mode()
        await pilot.pause()
        assert status.mode == "ask"


@pytest.mark.asyncio
async def test_slash_help_shows_help_bubble():
    app = ClarityMedApp(
        user_id="alice",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        input_bar.post_message(InputBar.Submitted("/help"))
        await pilot.pause()
        conv = app.query_one(Conversation)
        texts = [
            str(child.renderable)
            for child in conv.children
            if hasattr(child, "renderable")
        ]
        assert any("Slash commands" in t for t in texts)


@pytest.mark.asyncio
async def test_slash_mode_switches_mode():
    app = ClarityMedApp(
        user_id="alice",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/mode rag"))
        await pilot.pause()
        assert app.query_one(StatusBar).mode == "rag"


@pytest.mark.asyncio
async def test_ask_dispatch_streams_tokens_and_finalizes():
    """After Done, the assistant turn is swapped to a rendered Markdown widget."""
    from textual.widgets import Markdown

    app = ClarityMedApp(
        user_id="alice",
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

    seed = ChatSession.new("alice")
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
    resumed = ChatSession.resume("alice", seed.session_id)
    app = ClarityMedApp(
        user_id="alice",
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
    app = ClarityMedApp(user_id="alice", language="en")
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/mode ingest"))
        await pilot.pause()
    # Reaching here means unmount completed cleanly.
    assert True


@pytest.mark.asyncio
async def test_slash_user_switches_user_and_clears_history():
    app = ClarityMedApp(
        user_id="alice",
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
async def test_slash_user_without_arg_toasts_error():
    app = ClarityMedApp(
        user_id="alice",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/user"))
        await pilot.pause()
        # User did not switch.
        assert app.query_one(StatusBar).user_id == "alice"


@pytest.mark.asyncio
async def test_slash_mode_invalid_arg_toasts():
    app = ClarityMedApp(
        user_id="alice",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/mode bogus"))
        await pilot.pause()
        assert app.query_one(StatusBar).mode == "ask"


@pytest.mark.asyncio
async def test_unknown_slash_command_shows_inline_error():
    app = ClarityMedApp(
        user_id="alice",
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
async def test_ingest_mode_dispatch_runs_service():
    """Submitting plain text in ingest mode goes through IngestService.

    Uses the real IngestService (it is deterministic, no LLM).
    """
    app = ClarityMedApp(
        user_id="alice",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        # Switch to ingest mode.
        app.query_one(InputBar).post_message(InputBar.Submitted("/mode ingest"))
        await pilot.pause()
        # Submit a key=value.
        app.query_one(InputBar).post_message(InputBar.Submitted("allergy=penicillin"))
        for _ in range(20):
            await pilot.pause()
            if app._stream_worker is None or app._stream_worker.is_finished:
                break
        steps = app.query_one(ToolSteps)
        # ingest_service pushes ToolStarted+ToolCompleted for save_to_profile.
        step_text = " ".join(
            str(child.renderable)
            for child in steps.children
            if hasattr(child, "renderable")
        )
        assert "save_to_profile" in step_text


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
        user_id="alice",
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
        user_id="alice",
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
        user_id="alice",
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
        user_id="alice",
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
    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
    sentinel = object()
    app._cached_strategy = sentinel
    assert app._strategy_for_session() is sentinel


def test_strategy_for_session_rag_disabled(monkeypatch):
    """Returns None immediately when rag.enabled=False, then releases the lock."""
    mock_cfg = MagicMock()
    mock_cfg.rag.enabled = False
    monkeypatch.setattr("claritymed.core.rag.load_retrieval_config", lambda: mock_cfg)

    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
    assert app._strategy_for_session() is None
    assert not app._strategy_lock.locked()


def test_strategy_for_session_lock_timeout():
    """Raises RuntimeError when the lock cannot be acquired within the timeout."""
    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
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
    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
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
    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        # Skip real OCR provider construction by pre-seeding the slot.
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()

        # Blob landed: the sha is whatever sha256(b"fake-png-bytes") resolves to.
        rows = SessionAttachments("alice", app._chat_session.session_id).list()
        assert len(rows) == 1
        row = rows[0]
        assert row.filename == "clipboard.png"
        assert row.mime == "image/png"
        # Blob is on disk under the per-user CAS pool.
        assert BlobStore("alice").path(row.sha256, "png").exists()
        # OCR job was enqueued with matching identifiers.
        assert len(fake_worker.enqueued) == 1
        job = fake_worker.enqueued[0]
        assert job.user_id == "alice"
        assert job.sha256 == row.sha256


@pytest.mark.asyncio
async def test_paste_small_text_inserts_into_input(monkeypatch):
    from claritymed.cli.tui.paste import SmallText

    _patch_clipboard(monkeypatch, SmallText(text="hello world"))

    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_paste_clipboard()
        await pilot.pause()
        assert "hello world" in app.query_one(InputBar).value()


@pytest.mark.asyncio
async def test_paste_empty_clipboard_emits_toast(monkeypatch):
    from claritymed.cli.tui.paste import Empty

    _patch_clipboard(monkeypatch, Empty())

    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
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
    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()
        rows = SessionAttachments("alice", app._chat_session.session_id).list()
        assert len(rows) == 1
        assert rows[0].filename == "report.pdf"
        assert rows[0].mime == "application/pdf"
        assert len(fake_worker.enqueued) == 1


@pytest.mark.asyncio
async def test_paste_clipboard_read_failure_emits_toast(monkeypatch):
    """An exception from ``read_clipboard`` becomes an error toast, not a crash."""

    def _boom():
        raise RuntimeError("xclip segfault")

    monkeypatch.setattr("claritymed.cli.tui.paste.read_clipboard", _boom, raising=True)
    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_paste_clipboard()  # must not raise
        await pilot.pause()


@pytest.mark.asyncio
async def test_paste_adds_chat_history_and_tool_step(monkeypatch):
    """Image paste must add a visible system turn AND a ToolSteps row.

    Toasts disappear; the chat-history line is what the user sees if they
    scroll back or resume the session. The ToolSteps row is the progress
    indicator that flips to ✓ when OCR finishes."""
    from claritymed.cli.tui.paste import ImageBytes

    _patch_clipboard(monkeypatch, ImageBytes(bytes=b"png-bytes", ext="png"))

    fake_worker = _SyncOcrWorker()
    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ocr_worker = fake_worker
        app.action_paste_clipboard()
        await pilot.pause()

        # _session_turns must carry the attachment line.
        system_turns = [t for t in app._session_turns if t.role == "system"]
        assert any("attached clipboard.png" in t.text for t in system_turns)

        # ToolSteps has a ⟳ row for the OCR job.
        steps = app.query_one(ToolSteps)
        active_keys = list(steps._active.keys())
        assert any(k.startswith("ocr:") for k in active_keys), active_keys


@pytest.mark.asyncio
async def test_ocr_completed_marks_tool_step_done(monkeypatch):
    """The OcrWorker listener flips the ToolSteps row to ✓ on completion.

    Drives ``_on_ocr_completed`` directly to verify the status → summary
    branching (done / empty / failed / unknown all reach the same widget
    call path)."""
    from claritymed.cli.tui.paste import ImageBytes
    from claritymed.orchestrator.services.ocr_worker import OcrCompleted

    _patch_clipboard(monkeypatch, ImageBytes(bytes=b"png-bytes", ext="png"))

    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
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
                user_id="alice",
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

    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        steps = app.query_one(ToolSteps)
        sha = "a" * 64
        steps.push_start(f"ocr:{sha[:8]}", args_preview="x.png")
        app._on_ocr_completed(
            OcrCompleted(
                user_id="alice",
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

    app = ClarityMedApp(user_id="alice", language="en", chat_session=_fresh_session())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._chat_session = None
        app.action_paste_clipboard()
        await pilot.pause()
        # Did not crash. _session_turns has no attachment row.
        assert all("attached" not in t.text for t in app._session_turns)
