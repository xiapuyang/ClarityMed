"""Unit tests for the ``ask_user_question`` tool body.

The body is exercised directly (no pydantic-ai ``RunContext`` plumbing)
via ``ask_user_question_body``. Channel transports are stubbed so we
can exercise the three branches the LLM cares about: happy path,
channel-unavailable hint, user-declined hint.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from claritymed.core.events import ToolCompleted, ToolStarted
from claritymed.core.interaction.prompt_channel import (
    HeadlessPromptChannel,
    InteractiveChannelUnavailable,
    UserDeclinedAnswer,
)
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
    Question,
    QuestionOption,
)
from claritymed.core.interaction.tools.ask_user_question import (
    ASK_USER_QUESTION_TOOL_NAME,
    ask_user_question_body,
    build_ask_user_question_tool,
)
from claritymed.core.prompts.registry import PromptRegistry


@dataclass
class _FakeDeps:
    """Minimal deps namespace satisfying the TurnState fields the tool reads."""

    prompt_channel: Any | None = None
    tool_calls: dict[str, int] = field(default_factory=dict)
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


def _drain(queue: asyncio.Queue) -> list:
    """Synchronously collect everything sitting on ``queue`` for assertions."""
    out: list = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def _payload() -> AskUserQuestionInput:
    return AskUserQuestionInput(
        questions=[
            Question(
                question="Which library should we use for date formatting?",
                header="Library",
                options=[
                    QuestionOption(label="date-fns", description="Modern, modular."),
                    QuestionOption(label="dayjs", description="Tiny, moment-like."),
                ],
                multi_select=False,
            )
        ]
    )


class _ScriptedChannel:
    """Channel that returns a preset result and records the payload it saw."""

    def __init__(self, result: AskUserQuestionResult) -> None:
        self._result = result
        self.received: AskUserQuestionInput | None = None

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        self.received = payload
        return self._result


class _DeclineChannel:
    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        raise UserDeclinedAnswer("user pressed escape")


class _BoomChannel:
    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        raise RuntimeError("UI exploded")


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_body_returns_json_answers_on_success():
    channel = _ScriptedChannel(
        AskUserQuestionResult(
            answers={"Which library should we use for date formatting?": "dayjs"}
        )
    )
    deps = _FakeDeps(prompt_channel=channel)

    result = await ask_user_question_body(deps, _payload())

    parsed = json.loads(result)
    assert parsed == {
        "answers": {"Which library should we use for date formatting?": "dayjs"}
    }
    # The channel saw the same payload we passed in.
    assert channel.received is not None
    assert channel.received.questions[0].header == "Library"
    # Tool call counter was bumped exactly once.
    assert deps.tool_calls[ASK_USER_QUESTION_TOOL_NAME] == 1
    # Steps panel saw Started + Completed("answered ...") in order.
    events = _drain(deps.event_queue)
    assert isinstance(events[0], ToolStarted)
    assert events[0].tool_name == ASK_USER_QUESTION_TOOL_NAME
    assert events[0].args_preview == "Library"
    assert isinstance(events[-1], ToolCompleted)
    assert events[-1].tool_name == ASK_USER_QUESTION_TOOL_NAME
    assert events[-1].summary.startswith("answered")


# ----------------------------------------------------------------------
# Fallback paths
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_body_returns_unavailable_hint_when_channel_is_none():
    deps = _FakeDeps(prompt_channel=None)

    result = await ask_user_question_body(deps, _payload())

    assert result.startswith("[ask_user_question.unavailable]")
    # Counter still increments — the call did reach us.
    assert deps.tool_calls[ASK_USER_QUESTION_TOOL_NAME] == 1


@pytest.mark.asyncio
async def test_body_returns_unavailable_hint_for_headless_channel():
    deps = _FakeDeps(prompt_channel=HeadlessPromptChannel())

    result = await ask_user_question_body(deps, _payload())

    assert result.startswith("[ask_user_question.unavailable]")


@pytest.mark.asyncio
async def test_body_translates_user_declined():
    deps = _FakeDeps(prompt_channel=_DeclineChannel())

    result = await ask_user_question_body(deps, _payload())

    assert result.startswith("[ask_user_question.declined]")
    assert "Do NOT answer" in result
    events = _drain(deps.event_queue)
    completed = [e for e in events if isinstance(e, ToolCompleted)]
    assert completed and completed[-1].summary == "declined"


@pytest.mark.asyncio
async def test_body_translates_unexpected_channel_error():
    """A buggy channel must not crash the agent run."""
    deps = _FakeDeps(prompt_channel=_BoomChannel())

    result = await ask_user_question_body(deps, _payload())

    assert result.startswith("[ask_user_question.error]")


# ----------------------------------------------------------------------
# Builder wiring
# ----------------------------------------------------------------------


def test_build_tool_pulls_description_from_registry():
    tool = build_ask_user_question_tool(PromptRegistry(), language="en")
    assert tool.name == ASK_USER_QUESTION_TOOL_NAME
    # Substring from the YAML body — anchors that the description is
    # really coming from the prompt file, not a hardcoded string.
    assert "mutually exclusive" in tool.description.lower()


def test_build_tool_pulls_zh_description_from_registry():
    tool = build_ask_user_question_tool(PromptRegistry(), language="zh")
    assert "互斥" in tool.description


# ----------------------------------------------------------------------
# Channel direct
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_headless_channel_raises_unavailable():
    channel = HeadlessPromptChannel()
    with pytest.raises(InteractiveChannelUnavailable):
        await channel.ask(_payload())
