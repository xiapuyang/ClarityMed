"""Pydantic schemas for the ``ask_user_question`` tool input and result.

The tool's input schema is what the LLM actually sees — pydantic-ai
emits it as the tool's JSON Schema, and field descriptions are visible
to the model. So every constraint here doubles as a teaching aid:
``max_length=12`` on ``header`` isn't just enforcement, it is the only
hint the model gets that headers must be short chips.

Validation errors are not fatal — pydantic-ai retries the tool call with
the error attached, so the model gets a chance to re-emit a well-formed
payload. ``max_retries`` on the registered ``Tool`` bounds that loop.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuestionOption(BaseModel):
    """One mutually-exclusive choice the user can pick."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(
        ...,
        min_length=1,
        max_length=40,
        description=(
            "Short choice text the user will see and pick (1-5 words). "
            "Must be unique within the same question."
        ),
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=160,
        description=(
            "One sentence explaining what this choice means or what will "
            "happen if it is selected."
        ),
    )


class Question(BaseModel):
    """A single question with 2-4 enumerable options.

    Use multiple ``Question`` objects in one ``AskUserQuestionInput`` only
    when the user could reasonably answer them in one sitting (e.g.
    'pick a framework' + 'pick a database' — both setup choices). For
    unrelated questions, prefer one tool call per turn.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(
        ...,
        min_length=4,
        max_length=300,
        description=(
            "The full question, ending with '?'. Must be self-contained: "
            "the user may not recall prior turns."
        ),
    )
    header: str = Field(
        ...,
        min_length=1,
        max_length=12,
        description=(
            "Short chip label, up to 12 characters. Examples: "
            "'Auth method', 'Library', 'Approach'."
        ),
    )
    options: list[QuestionOption] = Field(
        ...,
        min_length=2,
        max_length=4,
        description=(
            "2-4 mutually exclusive options (unless multi_select=true). "
            "Do NOT include an 'Other' option — the UI adds free-text "
            "automatically."
        ),
    )
    multi_select: bool = Field(
        default=False,
        description=(
            "Set true ONLY when choices are genuinely not mutually "
            "exclusive (e.g. 'which symptoms apply'). Default false."
        ),
    )

    @model_validator(mode="after")
    def _no_duplicate_labels(self) -> "Question":
        labels = [opt.label.strip().lower() for opt in self.options]
        if len(set(labels)) != len(labels):
            msg = "Option labels must be unique within a question."
            raise ValueError(msg)
        return self


class AskUserQuestionInput(BaseModel):
    """Tool input: 1-4 related questions to ask the user this turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    questions: list[Question] = Field(
        ...,
        min_length=1,
        max_length=4,
        description=(
            "1-4 questions to present together. Each must be answerable "
            "without the others."
        ),
    )


class AskUserQuestionResult(BaseModel):
    """Tool result returned to the LLM after the user answers.

    ``answers`` maps the original question text (verbatim) to the user's
    selection — a single label string for single-select, a list of label
    strings for multi-select, or any free-form string when the user
    chose the auto-injected 'Other' option.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    answers: dict[str, str | list[str]] = Field(
        ...,
        description="Mapping of question text to user-selected label(s).",
    )
