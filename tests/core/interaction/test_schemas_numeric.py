"""Tests for the numeric ``Question`` extension (symptoms-plugin age input).

Pins the contract: ``NumericSpec`` is bounded, ``Question.numeric`` flips
the options shape, ``AskUserQuestionResult.numeric_values`` carries the
per-question numeric payload alongside the categorical answers map.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claritymed.core.interaction.schemas import (
    AskUserQuestionResult,
    NumericSpec,
    Question,
    QuestionOption,
)


def _opt(label: str) -> QuestionOption:
    return QuestionOption(label=label, description="an option")


# --- NumericSpec -----------------------------------------------------------


def test_numeric_spec_accepts_basic_range() -> None:
    spec = NumericSpec(min=0, max=120, step=1, unit="years")
    assert spec.min == 0
    assert spec.max == 120
    assert spec.unit == "years"


def test_numeric_spec_accepts_fractional_step() -> None:
    spec = NumericSpec(min=0, max=10, step=0.5)
    assert spec.step == 0.5


def test_numeric_spec_rejects_min_equal_max() -> None:
    with pytest.raises(ValidationError) as exc:
        NumericSpec(min=5, max=5)
    assert "min" in str(exc.value).lower()


def test_numeric_spec_rejects_min_above_max() -> None:
    with pytest.raises(ValidationError):
        NumericSpec(min=10, max=5)


def test_numeric_spec_rejects_zero_step() -> None:
    with pytest.raises(ValidationError) as exc:
        NumericSpec(min=0, max=10, step=0)
    assert "step" in str(exc.value).lower()


def test_numeric_spec_rejects_negative_step() -> None:
    with pytest.raises(ValidationError):
        NumericSpec(min=0, max=10, step=-1)


def test_numeric_spec_rejects_overlong_unit() -> None:
    with pytest.raises(ValidationError):
        NumericSpec(min=0, max=10, unit="A" * 13)


def test_numeric_spec_rejects_blank_unit() -> None:
    with pytest.raises(ValidationError):
        NumericSpec(min=0, max=10, unit="  ")


# --- Question.numeric ------------------------------------------------------


def test_question_age_numeric_validates() -> None:
    """KTD-13 canonical use: age as NumericSpec(0, 120, 1, 'years')."""
    q = Question(
        question="What is your age in years?",
        header="Age",
        options=[],
        numeric=NumericSpec(min=0, max=120, step=1, unit="years"),
    )
    assert q.numeric is not None
    assert q.options == []


def test_question_pain_scale_numeric_validates() -> None:
    q = Question(
        question="What is your pain on a 0-10 scale?",
        header="Pain",
        options=[],
        numeric=NumericSpec(min=0, max=10, step=1),
    )
    assert q.numeric is not None


def test_question_existing_categorical_still_works() -> None:
    """Regression: existing Question shape without numeric still validates."""
    q = Question(
        question="Which library should we use?",
        header="Library",
        options=[_opt("date-fns"), _opt("dayjs")],
    )
    assert q.numeric is None
    assert len(q.options) == 2


def test_question_rejects_numeric_with_options() -> None:
    """The two answer shapes are mutually exclusive."""
    with pytest.raises(ValidationError) as exc:
        Question(
            question="What is your age?",
            header="Age",
            options=[_opt("under 18"), _opt("18+")],
            numeric=NumericSpec(min=0, max=120),
        )
    assert "options" in str(exc.value).lower()


def test_question_rejects_numeric_with_multi_select() -> None:
    with pytest.raises(ValidationError) as exc:
        Question(
            question="What is your age?",
            header="Age",
            options=[],
            multi_select=True,
            numeric=NumericSpec(min=0, max=120),
        )
    assert "multi_select" in str(exc.value).lower()


def test_question_rejects_categorical_with_zero_options() -> None:
    """Without numeric, the 2-4 options floor still applies."""
    with pytest.raises(ValidationError) as exc:
        Question(question="What?", header="X", options=[])
    assert "2-4" in str(exc.value) or "options" in str(exc.value).lower()


def test_question_rejects_categorical_with_one_option() -> None:
    with pytest.raises(ValidationError):
        Question(question="What?", header="X", options=[_opt("only")])


def test_question_accepts_up_to_twenty_options() -> None:
    """Dataset-driven categorical questions can have up to 20 options."""
    q = Question(
        question="What?",
        header="X",
        options=[_opt(f"o{i}") for i in range(20)],
    )
    assert len(q.options) == 20


def test_question_rejects_categorical_with_twenty_one_options() -> None:
    with pytest.raises(ValidationError):
        Question(
            question="What?",
            header="X",
            options=[_opt(f"o{i}") for i in range(21)],
        )


# --- AskUserQuestionResult.numeric_values ----------------------------------


def test_result_accepts_numeric_values_alongside_answers() -> None:
    """D10 initial-batch: age (numeric) + sex (single-select) in one modal."""
    result = AskUserQuestionResult(
        answers={
            "What is your age?": "",
            "What is your sex assigned at birth?": "female",
        },
        numeric_values={"What is your age?": 35},
    )
    assert result.numeric_values["What is your age?"] == 35
    assert result.answers["What is your sex assigned at birth?"] == "female"


def test_result_numeric_values_default_empty() -> None:
    """Purely categorical modals don't need to populate numeric_values."""
    result = AskUserQuestionResult(answers={"q": "yes"})
    assert result.numeric_values == {}


def test_result_accepts_fractional_numeric_value() -> None:
    result = AskUserQuestionResult(
        answers={"q": ""},
        numeric_values={"q": 7.5},
    )
    assert result.numeric_values["q"] == 7.5


def test_result_rejects_string_numeric_value() -> None:
    """numeric_values is typed; a string here is a category answer in the wrong slot."""
    with pytest.raises(ValidationError):
        AskUserQuestionResult(
            answers={"q": ""},
            numeric_values={"q": "seven"},  # type: ignore[dict-item]
        )


# --- JSON-schema regression --------------------------------------------------


def test_question_json_schema_exposes_numeric_field() -> None:
    """The LLM-visible JSON schema must advertise the numeric field."""
    schema = Question.model_json_schema()
    props = schema.get("properties", {})
    assert "numeric" in props, (
        "Question's JSON schema must include 'numeric' so the LLM "
        f"can emit numeric questions; got properties: {sorted(props.keys())}"
    )


def test_question_options_description_still_steers_against_single_picker() -> None:
    """Regression on the existing description guard from test_schemas.py.

    Loosening the options Field constraint must NOT have erased the
    'DO NOT call this tool' hint that benchmarks proved was load-bearing
    against small-model 1-option-picker loops.
    """
    desc = (Question.model_fields["options"].description or "").lower()
    assert "do not call this tool" in desc
