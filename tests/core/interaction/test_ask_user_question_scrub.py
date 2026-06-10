"""Regression tests for the ``ask_user_question`` PHI scrub on return.

Locks in the contract introduced for ce:review P0 #3 — when the LLM's
modal renders an "Other" field, the user may type free text containing
PHI. Every string in ``AskUserQuestionResult.answers`` must flow through
``PhiGuard.scrub_free_text`` before the tool returns to the LLM,
otherwise the next LLM hop replays unscrubbed PHI to the (possibly
cloud) model.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
    Question,
    QuestionOption,
)
from claritymed.core.interaction.tools.ask_user_question import (
    ask_user_question_body,
)


@dataclass
class _FakeDeps:
    prompt_channel: Any | None = None
    tool_calls: dict[str, int] = field(default_factory=dict)
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


class _ReturnExactly:
    """Channel that returns a caller-supplied result verbatim."""

    def __init__(self, result: AskUserQuestionResult) -> None:
        self._result = result

    async def ask(self, _payload: AskUserQuestionInput) -> AskUserQuestionResult:
        return self._result


def _payload() -> AskUserQuestionInput:
    return AskUserQuestionInput(
        questions=[
            Question(
                question="Pick a contact channel?",
                header="Contact",
                options=[
                    QuestionOption(label="phone", description="Call me back."),
                    QuestionOption(label="email", description="Email me."),
                ],
                multi_select=False,
            )
        ]
    )


@pytest.mark.asyncio
async def test_scrub_removes_phone_number_from_other_answer():
    """The CN-mobile regex in safety.yaml should rewrite the free-text
    answer before the tool returns the JSON blob to the LLM."""
    channel = _ReturnExactly(
        AskUserQuestionResult(
            answers={"Pick a contact channel?": "actually call me at 13912345678"}
        )
    )
    deps = _FakeDeps(prompt_channel=channel)

    out = await ask_user_question_body(deps, _payload())
    parsed = json.loads(out)
    answer = parsed["answers"]["Pick a contact channel?"]
    assert "13912345678" not in answer
    assert "[REDACTED:PHONE]" in answer


@pytest.mark.asyncio
async def test_scrub_removes_email_from_list_answer():
    """Multi-select answers come back as a list; every entry must be
    scrubbed."""
    channel = _ReturnExactly(
        AskUserQuestionResult(
            answers={
                "Pick a contact channel?": [
                    "phone",
                    "alice@example.com",
                ]
            }
        )
    )
    deps = _FakeDeps(prompt_channel=channel)

    out = await ask_user_question_body(deps, _payload())
    parsed = json.loads(out)
    answers = parsed["answers"]["Pick a contact channel?"]
    assert isinstance(answers, list)
    assert "phone" in answers
    assert not any("alice@example.com" in a for a in answers)
    assert any("[REDACTED:EMAIL]" in a for a in answers)


@pytest.mark.asyncio
async def test_scrub_passes_clean_picked_labels_through_unchanged():
    """Labels the LLM proposed (no PHI possible) still round-trip cleanly
    through the scrub pass."""
    channel = _ReturnExactly(
        AskUserQuestionResult(answers={"Pick a contact channel?": "email"})
    )
    deps = _FakeDeps(prompt_channel=channel)

    out = await ask_user_question_body(deps, _payload())
    parsed = json.loads(out)
    assert parsed["answers"]["Pick a contact channel?"] == "email"
