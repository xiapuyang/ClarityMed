"""Multi-turn chat history integrity.

Regression coverage for the dead-code bug: _finalize_turn lived after the
try/finally block of an async generator.  The TUI always returns immediately
when it receives Done, which triggers aclose() on the generator and skips
everything after the finally block.  The fix moves _finalize_turn to execute
inside the loop on the Done event, before yielding Done to the consumer.

Every test here uses _consume_until_done() to replicate the TUI pattern.
Using the full-drain idiom (``async for _ in service.run(): pass``) would pass
even with the old broken code, so it cannot serve as a regression guard.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel

from claritymed.core.events import Done
from claritymed.orchestrator.services import AskService, ChatSession


async def _consume_until_done(stream) -> list:
    """Drain events until Done, then stop — exactly what the TUI does.

    The TUI calls return the moment it receives Done.  That exit closes the
    generator via aclose(), which is the trigger for the regression.
    """
    events = []
    async for event in stream:
        events.append(event)
        if isinstance(event, Done):
            return events
    return events


# ---------------------------------------------------------------------------
# Core regression: _finalize_turn must run when consumer breaks on Done
# ---------------------------------------------------------------------------


async def test_message_history_populated_after_tui_style_consumption():
    """_message_history must be non-empty after a TUI-style break on Done.

    This is the minimal reproduction of the original bug: if _finalize_turn
    is only reachable after the generator body rather than inside it, breaking
    on Done leaves _message_history empty and every subsequent turn starts
    with no context.
    """
    session = ChatSession.new("alice")
    service = AskService(
        model=TestModel(custom_output_text="answer one"),
        chat_session=session,
    )

    await _consume_until_done(service.run("first question", user_id="alice"))

    history = session.message_history()
    assert history, (
        "_message_history is empty after TUI-style Done break — "
        "_finalize_turn was not called (dead-code regression)"
    )


async def test_session_file_written_after_tui_style_consumption():
    """The JSONL file must contain a user + assistant pair after TUI break."""
    import json

    session = ChatSession.new("alice")
    service = AskService(
        model=TestModel(custom_output_text="response"),
        chat_session=session,
    )

    await _consume_until_done(service.run("ping", user_id="alice"))

    assert session.path.exists(), "session file was never created"
    kinds = [
        json.loads(line)["type"]
        for line in session.path.read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert "user" in kinds, "user event missing from session file"
    assert "assistant" in kinds, "assistant event missing — append_assistant not called"


# ---------------------------------------------------------------------------
# Turn-to-turn history propagation
# ---------------------------------------------------------------------------


async def test_turn_two_receives_turn_one_prompt_in_history():
    """The user prompt from turn 1 must appear in the history fed to turn 2.

    pydantic-ai's TestModel includes message_history in all_messages(), so
    after turn 2 the session's _message_history contains messages from both
    turns.  Asserting the turn-1 prompt is present proves the history was
    threaded through.
    """
    session = ChatSession.new("alice")
    service = AskService(
        model=TestModel(custom_output_text="ok"),
        chat_session=session,
    )

    await _consume_until_done(service.run("what is hemoglobin?", user_id="alice"))
    await _consume_until_done(
        service.run("and what about hematocrit?", user_id="alice")
    )

    history = session.message_history()
    user_prompts = [
        part.content
        for msg in history
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, UserPromptPart)
    ]
    assert any("hemoglobin" in p for p in user_prompts), (
        "turn-1 prompt not found in session history after turn 2 — "
        "history was not passed to the LLM on turn 2"
    )


async def test_three_turn_history_contains_all_prompts():
    """History must accumulate across three consecutive turns."""
    prompts = ["question one", "question two", "question three"]
    session = ChatSession.new("alice")
    service = AskService(
        model=TestModel(custom_output_text="ok"),
        chat_session=session,
    )

    for prompt in prompts:
        await _consume_until_done(service.run(prompt, user_id="alice"))

    history = session.message_history()
    user_texts = [
        part.content
        for msg in history
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, UserPromptPart)
    ]
    for prompt in prompts:
        assert any(prompt in t for t in user_texts), (
            f"'{prompt}' missing from history after three turns"
        )


async def test_message_count_grows_across_turns():
    """Each turn must add messages; a flat count flags history truncation."""
    session = ChatSession.new("alice")
    service = AskService(
        model=TestModel(custom_output_text="ok"),
        chat_session=session,
    )

    await _consume_until_done(service.run("turn one", user_id="alice"))
    count_after_1 = len(session.message_history())

    await _consume_until_done(service.run("turn two", user_id="alice"))
    count_after_2 = len(session.message_history())

    await _consume_until_done(service.run("turn three", user_id="alice"))
    count_after_3 = len(session.message_history())

    assert count_after_2 > count_after_1, "message count did not grow after turn 2"
    assert count_after_3 > count_after_2, "message count did not grow after turn 3"


# ---------------------------------------------------------------------------
# Resume round-trip
# ---------------------------------------------------------------------------


async def test_cancel_in_foreign_context_emits_cancelled_audit(monkeypatch):
    """Cancelling the stream after the caller resets its ContextVars must
    still emit the ``mode.cancelled`` audit row.

    Regression: the TUI worker's finally calls reset_context() before the
    service generator is aclose()'d by asyncio's finalizer. The aclose may
    fire in a context where request_id / user_id / language are unset, so
    audit_event() in _run_scoped's finally raised MissingContextError and
    the broad except logged "failed to persist cancelled turn for user ...".

    The fix rehydrates ContextVars from values captured at the top of
    _run_scoped. We reproduce the foreign-context cancellation by driving
    _run_scoped directly: pull one event so the generator suspends past
    the captures, reset the caller's ContextVars, then aclose. The finally
    must still successfully emit the cancellation audit row.
    """
    from claritymed.context import (
        apply_context,
        new_request_id,
        reset_context,
    )
    from claritymed.orchestrator.services import ask_service as ask_mod

    real_audit_event = ask_mod.audit_event
    captured: list[str] = []

    def tracking_audit_event(event, payload=None):
        result = real_audit_event(event, payload)
        captured.append(event)
        return result

    monkeypatch.setattr(ask_mod, "audit_event", tracking_audit_event)

    session = ChatSession.new("alice")
    service = AskService(
        model=TestModel(custom_output_text="partial answer"),
        chat_session=session,
    )

    # Drive _run_scoped directly to avoid nested-generator finalization
    # ordering issues — aclose() on the outer run() generator does not
    # synchronously close nested ``async for`` generators, so we'd never
    # see _run_scoped's finally during the test body.
    tokens = apply_context(new_request_id(), "alice", "en")
    stream = service._run_scoped("a question", user_id="alice")
    await stream.__anext__()
    # Caller clears its ContextVars before the generator finalizes,
    # mirroring _run_stream's reset_context(per_turn) on cancel.
    reset_context(tokens)
    await stream.aclose()

    assert "mode.cancelled" in captured, (
        "mode.cancelled audit row never emitted — audit_event likely raised "
        "MissingContextError inside _run_scoped's finally and the broad except "
        "swallowed it. Captured events: " + repr(captured)
    )


async def test_resume_reconstructs_history_for_new_service_instance():
    """ChatSession.resume() must rebuild history so a new AskService instance
    (e.g. after a provider switch) picks up where the previous one left off.

    This is the cross-process / cache-invalidation scenario: the cached
    AskService is discarded and a fresh one is created mid-conversation.
    """
    session = ChatSession.new("alice")
    service_a = AskService(
        model=TestModel(custom_output_text="answer from a"),
        chat_session=session,
    )
    await _consume_until_done(service_a.run("original question", user_id="alice"))

    # Simulate provider switch: new service, new ChatSession resumed from disk.
    resumed = ChatSession.resume("alice", session.session_id)
    service_b = AskService(
        model=TestModel(custom_output_text="answer from b"),
        chat_session=resumed,
    )
    await _consume_until_done(service_b.run("follow-up question", user_id="alice"))

    history = resumed.message_history()
    user_texts = [
        part.content
        for msg in history
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, UserPromptPart)
    ]
    assert any("original question" in t for t in user_texts), (
        "resumed session does not contain turn-1 prompt — "
        "resume() failed to reconstruct message history from disk"
    )
