"""Tests for ``ask_user_question`` payload schemas.

The schema is the contract the LLM has to satisfy. These tests pin down
the constraints (counts, lengths, uniqueness) so a change to the field
descriptions cannot silently change the validation surface.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    Question,
    QuestionOption,
)


def _opt(label: str, description: str = "an option") -> QuestionOption:
    return QuestionOption(label=label, description=description)


def _question(**overrides) -> Question:
    base = dict(
        question="Which framework should we use for date formatting?",
        header="Library",
        options=[_opt("date-fns"), _opt("dayjs")],
        multi_select=False,
    )
    base.update(overrides)
    return Question(**base)


def test_question_accepts_minimum_valid_shape():
    q = _question()
    assert q.header == "Library"
    assert len(q.options) == 2


def test_question_rejects_header_over_twelve_chars():
    with pytest.raises(ValidationError) as exc:
        _question(header="ThirteenChars")
    assert "header" in str(exc.value).lower()


def test_question_rejects_single_option():
    with pytest.raises(ValidationError):
        _question(options=[_opt("only")])


def test_question_rejects_more_than_four_options():
    too_many = [_opt(f"opt-{i}") for i in range(5)]
    with pytest.raises(ValidationError):
        _question(options=too_many)


def test_question_rejects_duplicate_option_labels_case_insensitive():
    with pytest.raises(ValidationError) as exc:
        _question(options=[_opt("Yes"), _opt("yes")])
    assert "unique" in str(exc.value).lower()


def test_question_rejects_extra_fields():
    """``extra=forbid`` guards against an LLM hallucinating fields."""
    with pytest.raises(ValidationError):
        Question(
            question="What?",
            header="X",
            options=[_opt("a"), _opt("b")],
            multi_select=False,
            answered_already="yes",  # type: ignore[call-arg]
        )


def test_input_requires_at_least_one_question():
    with pytest.raises(ValidationError):
        AskUserQuestionInput(questions=[])


def test_input_caps_questions_at_four():
    too_many = [_question(header=str(i)) for i in range(5)]
    with pytest.raises(ValidationError):
        AskUserQuestionInput(questions=too_many)


def test_input_accepts_four_questions():
    qs = [_question(header=str(i)) for i in range(4)]
    payload = AskUserQuestionInput(questions=qs)
    assert len(payload.questions) == 4
