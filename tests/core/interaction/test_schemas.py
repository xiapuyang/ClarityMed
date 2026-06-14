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


def test_question_accepts_header_up_to_twenty_chars():
    """Bumped from 12 → 20 after benchmark runs against omlx kept burning
    a retry slot on ``Which report?`` (13) / ``Severity level`` (14) —
    headers that fit the TUI chip area but tripped the old limit."""
    # 20 chars exactly — must pass.
    q = _question(header="A" * 20)
    assert q.header == "A" * 20


def test_question_rejects_header_over_twenty_chars():
    with pytest.raises(ValidationError) as exc:
        _question(header="A" * 21)
    assert "header" in str(exc.value).lower()


def test_question_rejects_single_option():
    with pytest.raises(ValidationError):
        _question(options=[_opt("only")])


def test_options_description_teaches_against_one_option_picker():
    """The ``options`` Field description is the only schema-level hint
    the LLM sees inside the tool's JSON Schema. Bench surfaced models
    looping on 1-option pickers because the retry loop cannot
    manufacture an option that doesn't exist — the only escape is
    teaching the model to switch tools BEFORE the call. Pin the
    description carries this guard."""
    from claritymed.core.interaction.schemas import Question

    options_field = Question.model_fields["options"]
    desc = (options_field.description or "").lower()
    assert "do not call this tool" in desc, (
        f"options description must steer model away from this tool when "
        f"only one option exists; got: {desc!r}"
    )


def test_question_rejects_more_than_twenty_options():
    too_many = [_opt(f"opt-{i}") for i in range(21)]
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
