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
    AskUserQuestionResult,
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


def test_answered_numeric_shows_value_and_unit() -> None:
    """Regression — numeric answers stored as '' in answers dict must
    display the real value from numeric_values, not an empty arrow."""
    q_text = "How old are you, in years?"
    payload = AskUserQuestionInput(
        questions=[
            Question(
                question=q_text,
                header="Age",
                numeric=NumericSpec(min=0, max=120, step=1, unit="years"),
            )
        ]
    )
    result = AskUserQuestionResult(
        answers={q_text: ""},
        numeric_values={q_text: 35},
    )
    app = _StubApp()
    channel = TextualPromptChannel(app)  # type: ignore[arg-type]
    channel._post_answered_note(payload, result)
    note = app.conversation.system_turns[0]
    assert "35 years" in note
    # Empty arrow (just ▶ with no value) must not appear.
    assert "▶ \n" not in note
    assert note.strip().endswith("35 years")


def test_answered_numeric_without_unit_omits_unit() -> None:
    """Numeric answer with no unit should not append 'None'."""
    q_text = "Pain 0-10?"
    payload = AskUserQuestionInput(
        questions=[
            Question(
                question=q_text,
                header="Pain",
                numeric=NumericSpec(min=0, max=10, step=1),
            )
        ]
    )
    result = AskUserQuestionResult(
        answers={q_text: ""},
        numeric_values={q_text: 7},
    )
    app = _StubApp()
    channel = TextualPromptChannel(app)  # type: ignore[arg-type]
    channel._post_answered_note(payload, result)
    note = app.conversation.system_turns[0]
    assert "7" in note
    assert "None" not in note


def test_answered_mixed_batch_renders_both_shapes() -> None:
    """Confirmed answer for numeric + categorical in one modal must
    display both correctly."""
    q_age = "How old are you, in years?"
    q_sex = "Biological sex?"
    payload = AskUserQuestionInput(
        questions=[
            Question(
                question=q_age,
                header="Age",
                numeric=NumericSpec(min=0, max=120, step=1, unit="years"),
            ),
            Question(
                question=q_sex,
                header="Sex",
                options=[
                    QuestionOption(label="Female", description="F"),
                    QuestionOption(label="Male", description="M"),
                ],
            ),
        ]
    )
    result = AskUserQuestionResult(
        answers={q_age: "", q_sex: "Female"},
        numeric_values={q_age: 42},
    )
    app = _StubApp()
    channel = TextualPromptChannel(app)  # type: ignore[arg-type]
    channel._post_answered_note(payload, result)
    note = app.conversation.system_turns[0]
    assert "42 years" in note
    assert "Female" in note


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
