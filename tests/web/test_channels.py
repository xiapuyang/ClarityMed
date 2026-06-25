"""Unit tests for web-layer prompt/approval channels.

These cover the error and cancellation branches in WebPromptChannel and
WebToolApprovalChannel that the SSE streaming integration tests can't
easily drive (future cancellation, bad payload shape, unknown decision).
"""

from __future__ import annotations

import asyncio

import pytest

from claritymed.core.interaction import (
    InteractiveChannelUnavailable,
    UserDeclinedAnswer,
)
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
    Question,
    QuestionOption,
)
from claritymed.web.channels import (
    WebPromptChannel,
    WebToolApprovalChannel,
    _safe_args,
    build_web_channels,
)


# --- helpers ------------------------------------------------------------


def _make_prompt_channel(rendezvous=None, emit_queue=None):
    return WebPromptChannel(
        user_id="test",
        session_id="sess-1",
        emit_queue=emit_queue or asyncio.Queue(),
        rendezvous=rendezvous if rendezvous is not None else {},
    )


def _make_approval_channel(rendezvous=None, emit_queue=None):
    return WebToolApprovalChannel(
        user_id="test",
        session_id="sess-1",
        emit_queue=emit_queue or asyncio.Queue(),
        rendezvous=rendezvous if rendezvous is not None else {},
    )


def _simple_payload() -> AskUserQuestionInput:
    return AskUserQuestionInput(
        questions=[
            Question(
                question="What symptom?",
                header="Symptom",
                options=[
                    QuestionOption(label="Fever", description="High temp"),
                    QuestionOption(label="Cough", description="Cough"),
                ],
            )
        ]
    )


# --- _safe_args ---------------------------------------------------------


def test_safe_args_short_string_passes_through():
    result = _safe_args({"key": "short"})
    assert result == {"key": "short"}


def test_safe_args_long_string_truncated():
    long_val = "x" * 300
    result = _safe_args({"note": long_val})
    assert len(result["note"]) == 256
    assert result["note"].endswith("...")


def test_safe_args_non_string_untouched():
    result = _safe_args({"count": 42, "flag": True, "data": [1, 2, 3]})
    assert result == {"count": 42, "flag": True, "data": [1, 2, 3]}


def test_safe_args_exactly_256_chars_passes_through():
    val = "a" * 256
    result = _safe_args({"k": val})
    assert result["k"] == val


# --- WebPromptChannel.ask -----------------------------------------------


async def test_prompt_channel_ask_cancelled_raises_user_declined():
    """Future cancellation → UserDeclinedAnswer (not asyncio.CancelledError)."""
    rendezvous = {}
    emit_queue: asyncio.Queue = asyncio.Queue()
    channel = _make_prompt_channel(rendezvous=rendezvous, emit_queue=emit_queue)

    async def _cancel_and_ask():
        async def _canceller():
            # Wait until the interaction is registered then cancel it.
            deadline = asyncio.get_running_loop().time() + 2.0
            while asyncio.get_running_loop().time() < deadline:
                if rendezvous:
                    iid = next(iter(rendezvous))
                    rendezvous[iid]["future"].cancel()
                    return
                await asyncio.sleep(0.01)

        asyncio.ensure_future(_canceller())
        return await channel.ask(_simple_payload())

    with pytest.raises(UserDeclinedAnswer):
        await _cancel_and_ask()


async def test_prompt_channel_ask_invalid_response_raises_user_declined():
    """Bad payload shape from POST /interactions → UserDeclinedAnswer."""
    rendezvous = {}
    emit_queue: asyncio.Queue = asyncio.Queue()
    channel = _make_prompt_channel(rendezvous=rendezvous, emit_queue=emit_queue)

    async def _resolve_bad_payload():
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if rendezvous:
                iid = next(iter(rendezvous))
                rendezvous[iid]["future"].set_result({"bad": "shape"})
                return
            await asyncio.sleep(0.01)

    asyncio.ensure_future(_resolve_bad_payload())

    with pytest.raises(UserDeclinedAnswer, match="invalid response shape"):
        await channel.ask(_simple_payload())


async def test_prompt_channel_ask_valid_response_returns_result():
    """Happy path: future resolves with valid payload → AskUserQuestionResult."""
    rendezvous = {}
    emit_queue: asyncio.Queue = asyncio.Queue()
    channel = _make_prompt_channel(rendezvous=rendezvous, emit_queue=emit_queue)

    async def _resolve():
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if rendezvous:
                iid = next(iter(rendezvous))
                rendezvous[iid]["future"].set_result(
                    {"answers": {"What symptom?": "Fever"}}
                )
                return
            await asyncio.sleep(0.01)

    asyncio.ensure_future(_resolve())
    result = await channel.ask(_simple_payload())
    assert isinstance(result, AskUserQuestionResult)
    assert result.answers["What symptom?"] == "Fever"


async def test_prompt_channel_emits_interaction_requested():
    """ask() pushes an InteractionRequested into the emit queue."""
    from claritymed.core.events import InteractionRequested

    rendezvous = {}
    emit_queue: asyncio.Queue = asyncio.Queue()
    channel = _make_prompt_channel(rendezvous=rendezvous, emit_queue=emit_queue)

    # Cancel immediately so ask() returns quickly.
    async def _cancel():
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if rendezvous:
                iid = next(iter(rendezvous))
                rendezvous[iid]["future"].cancel()
                return
            await asyncio.sleep(0.01)

    asyncio.ensure_future(_cancel())
    with pytest.raises(UserDeclinedAnswer):
        await channel.ask(_simple_payload())

    event = emit_queue.get_nowait()
    assert isinstance(event, InteractionRequested)
    assert event.kind == "ask_user_question"


# --- WebToolApprovalChannel.request ------------------------------------


async def test_approval_channel_cancelled_raises_channel_unavailable():
    """Future cancellation → InteractiveChannelUnavailable."""
    rendezvous = {}
    emit_queue: asyncio.Queue = asyncio.Queue()
    channel = _make_approval_channel(rendezvous=rendezvous, emit_queue=emit_queue)

    async def _cancel():
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if rendezvous:
                iid = next(iter(rendezvous))
                rendezvous[iid]["future"].cancel()
                return
            await asyncio.sleep(0.01)

    asyncio.ensure_future(_cancel())
    with pytest.raises(InteractiveChannelUnavailable):
        await channel.request("save_record", {"field": "allergy"})


async def test_approval_channel_unknown_decision_returns_deny():
    """An unrecognised decision string is safely mapped to 'deny'."""
    rendezvous = {}
    emit_queue: asyncio.Queue = asyncio.Queue()
    channel = _make_approval_channel(rendezvous=rendezvous, emit_queue=emit_queue)

    async def _resolve():
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if rendezvous:
                iid = next(iter(rendezvous))
                rendezvous[iid]["future"].set_result({"decision": "???UNKNOWN???"})
                return
            await asyncio.sleep(0.01)

    asyncio.ensure_future(_resolve())
    decision = await channel.request("save_record", {"x": 1})
    assert decision.decision == "deny"


async def test_approval_channel_valid_once_decision():
    """'once' decision returns correctly."""
    rendezvous = {}
    emit_queue: asyncio.Queue = asyncio.Queue()
    channel = _make_approval_channel(rendezvous=rendezvous, emit_queue=emit_queue)

    async def _resolve():
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if rendezvous:
                iid = next(iter(rendezvous))
                rendezvous[iid]["future"].set_result({"decision": "once"})
                return
            await asyncio.sleep(0.01)

    asyncio.ensure_future(_resolve())
    decision = await channel.request("save_record", {"x": 1})
    assert decision.decision == "once"


async def test_approval_channel_emits_interaction_requested_with_safe_args():
    """request() pushes an InteractionRequested with truncated args."""
    from claritymed.core.events import InteractionRequested

    rendezvous = {}
    emit_queue: asyncio.Queue = asyncio.Queue()
    channel = _make_approval_channel(rendezvous=rendezvous, emit_queue=emit_queue)

    async def _cancel():
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if rendezvous:
                iid = next(iter(rendezvous))
                rendezvous[iid]["future"].cancel()
                return
            await asyncio.sleep(0.01)

    asyncio.ensure_future(_cancel())
    long_arg = "y" * 400
    with pytest.raises(InteractiveChannelUnavailable):
        await channel.request("save_record", {"content": long_arg}, breadcrumb="step1")

    event = emit_queue.get_nowait()
    assert isinstance(event, InteractionRequested)
    assert event.kind == "tool_approval"
    # args should be truncated
    assert len(event.payload["args"]["content"]) == 256


# --- build_web_channels -------------------------------------------------


def test_build_web_channels_returns_both():
    """Smoke test: returns prompt + approval channel with shared state."""
    q: asyncio.Queue = asyncio.Queue()
    rendezvous: dict = {}
    prompt, approval = build_web_channels(
        user_id="u",
        session_id="s",
        emit_queue=q,
        rendezvous=rendezvous,
    )
    assert isinstance(prompt, WebPromptChannel)
    assert isinstance(approval, WebToolApprovalChannel)
