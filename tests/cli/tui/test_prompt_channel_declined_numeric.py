"""``TextualPromptChannel._post_declined_note`` regression — render the
numeric range hint instead of empty parens.

Without the numeric branch, declining a numeric question would print
``Age（）`` because ``options`` is ``[]`` for numeric questions — a
visual bug that misrepresents the modal as "nothing was offered".
"""

from __future__ import annotations

from typing import Any

from claritymed.cli.tui.prompt_channel import TextualPromptChannel
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    NumericSpec,
    Question,
    QuestionOption,
)


class _StubConversation:
    """Stand-in widget that captures the text instead of rendering it."""

    def __init__(self) -> None:
        self.system_turns: list[str] = []

    def add_system_turn(self, text: str) -> None:
        self.system_turns.append(text)


class _StubApp:
    """Minimal app that returns the same ``_StubConversation`` for any
    ``query_one(Conversation)`` call. The channel's ``_add_system_turn``
    catches everything else, so we only need this one method."""

    def __init__(self) -> None:
        self.conversation = _StubConversation()

    def query_one(self, _type: Any) -> Any:
        return self.conversation


def test_declined_numeric_renders_range_not_empty_parens() -> None:
    payload = AskUserQuestionInput(
        questions=[
            Question(
                question="How old are you, in years?",
                header="Age",
                numeric=NumericSpec(min=0, max=120, step=1, unit="years"),
            )
        ]
    )
    app = _StubApp()
    channel = TextualPromptChannel(app)  # type: ignore[arg-type]
    channel._post_declined_note(payload)
    assert len(app.conversation.system_turns) == 1
    note = app.conversation.system_turns[0]
    # The range hint must appear and the empty-parens artifact must NOT.
    assert "0-120 years" in note
    assert "（）" not in note
    assert "()" not in note


def test_declined_categorical_still_renders_options() -> None:
    """Regression — adding the numeric branch must not break the
    existing categorical declined-note format."""
    payload = AskUserQuestionInput(
        questions=[
            Question(
                question="Pick one?",
                header="Pick",
                options=[
                    QuestionOption(label="A", description="A."),
                    QuestionOption(label="B", description="B."),
                ],
            )
        ]
    )
    app = _StubApp()
    channel = TextualPromptChannel(app)  # type: ignore[arg-type]
    channel._post_declined_note(payload)
    note = app.conversation.system_turns[0]
    assert "A / B" in note


def test_declined_mixed_batch_renders_both_shapes() -> None:
    """The D10 initial-batch combines numeric + categorical in one
    modal — declining must list both correctly."""
    payload = AskUserQuestionInput(
        questions=[
            Question(
                question="How old are you, in years?",
                header="Age",
                numeric=NumericSpec(min=0, max=120, step=1, unit="years"),
            ),
            Question(
                question="Biological sex?",
                header="Sex",
                options=[
                    QuestionOption(label="Female", description="F"),
                    QuestionOption(label="Male", description="M"),
                ],
            ),
        ]
    )
    app = _StubApp()
    channel = TextualPromptChannel(app)  # type: ignore[arg-type]
    channel._post_declined_note(payload)
    note = app.conversation.system_turns[0]
    assert "0-120 years" in note
    assert "Female / Male" in note


def test_declined_numeric_without_unit_omits_unit() -> None:
    """A pain-scale question with ``unit=None`` should render ``0-10``
    rather than ``0-10 None``."""
    payload = AskUserQuestionInput(
        questions=[
            Question(
                question="Pain on 0-10 scale?",
                header="Pain",
                numeric=NumericSpec(min=0, max=10, step=1),
            )
        ]
    )
    app = _StubApp()
    channel = TextualPromptChannel(app)  # type: ignore[arg-type]
    channel._post_declined_note(payload)
    note = app.conversation.system_turns[0]
    assert "0-10" in note
    assert "None" not in note
