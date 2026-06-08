"""Smoke tests for ``ClarityMedApp`` using Textual's ``App.run_test()``.

These tests exercise mount + widget composition + a couple of basic actions
without touching the LLM or qdrant. Service factories are injected so the
app never tries to resolve a real provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

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
async def test_unknown_slash_command_toasts():
    app = ClarityMedApp(
        user_id="alice",
        language="en",
        chat_session=_fresh_session(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one(InputBar).post_message(InputBar.Submitted("/sproingify"))
        await pilot.pause()
        # The conversation stays empty (no user turn for unknown commands).
        conv = app.query_one(Conversation)
        # only the empty-state placeholder
        assert all("empty" in child.classes for child in conv.children)


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
