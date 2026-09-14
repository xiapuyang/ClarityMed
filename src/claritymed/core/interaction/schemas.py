"""Pydantic schemas for the ``ask_user_question`` tool input and result.

The tool's input schema is what the LLM actually sees — pydantic-ai
emits it as the tool's JSON Schema, and field descriptions are visible
to the model. So every constraint here doubles as a teaching aid:
``max_length=30`` on ``header`` isn't just enforcement, it is the only
hint the model gets that headers must be short chips. (Bumped from 12
to 20 after benchmark runs showed small models routinely emit 13-18
char headers like ``Which report?`` / ``Severity level`` that fit the
TUI chip area but tripped the old limit, burning a retry slot. Bumped
again to 30 to accommodate medical phrasing like ``Analyze breast
ultrasound?`` without forcing awkward truncation.)

Validation errors are not fatal — pydantic-ai retries the tool call with
the error attached, so the model gets a chance to re-emit a well-formed
payload. ``max_retries`` on the registered ``Tool`` bounds that loop.

Numeric questions (added for the symptoms plugin's age input) use the
optional :class:`NumericSpec` and leave ``options`` as an empty list.
The 2-4 floor on ``options`` is enforced via a model_validator (not the
Field constraint) so a numeric question's empty list is structurally
valid; the validator still rejects 0/1/5+-option categorical questions.
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
    value: str | None = Field(
        default=None,
        description=(
            "Internal raw identifier for this option (e.g. a dataset value code). "
            "Not shown to the user. When set, callers may pass it back as "
            "``answer_value`` to skip label re-matching on the server."
        ),
    )


class NumericSpec(BaseModel):
    """Numeric input descriptor for :class:`Question`.

    Set on a :class:`Question` when the answer is a number rather than a
    pick from enumerable options (e.g. age in years, pain on a 0-10
    scale). The TUI modal renders an ``Input(type="number")`` with the
    range hint as placeholder; invalid values are rejected inline and
    submit is blocked until the value parses + lies inside ``[min, max]``.

    ``unit`` is appended to the range hint when present (``"0-120 years"``);
    keep it short — the chip area is narrow.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min: float | int = Field(
        ...,
        description="Lower bound, inclusive. Numbers below this are rejected.",
    )
    max: float | int = Field(
        ...,
        description="Upper bound, inclusive. Numbers above this are rejected.",
    )
    step: float | int = Field(
        default=1,
        description=(
            "Granularity of accepted values. Submit is blocked unless "
            "``(value - min) / step`` is integer (within floating-point "
            "tolerance). Default 1."
        ),
    )
    unit: str | None = Field(
        default=None,
        max_length=12,
        description=(
            "Optional unit token rendered next to the range hint, e.g. "
            "'years', 'mg/dL'. Keep under 12 chars."
        ),
    )

    @model_validator(mode="after")
    def _bounds_valid(self) -> "NumericSpec":
        if self.min >= self.max:
            raise ValueError(f"NumericSpec.min ({self.min}) must be < max ({self.max})")
        if self.step <= 0:
            raise ValueError(f"NumericSpec.step ({self.step}) must be > 0")
        if self.unit is not None and not self.unit.strip():
            raise ValueError("NumericSpec.unit, when set, must be non-blank")
        return self


class Question(BaseModel):
    """A single question with 2-4 enumerable options OR a numeric input.

    Use multiple ``Question`` objects in one ``AskUserQuestionInput`` only
    when the user could reasonably answer them in one sitting (e.g.
    'pick a framework' + 'pick a database' — both setup choices). For
    unrelated questions, prefer one tool call per turn.

    Numeric questions set :attr:`numeric` and leave :attr:`options` as
    ``[]`` — the TUI renders an Input widget instead of the option list.
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
        max_length=30,
        description=(
            "Short chip label, up to 30 characters. Examples: "
            "'Auth method', 'Severity level', 'Which report?'."
        ),
    )
    options: list[QuestionOption] = Field(
        default_factory=list,
        description=(
            "2-4 mutually exclusive options (unless multi_select=true). "
            "If you cannot enumerate at least 2 distinct concrete choices, "
            "DO NOT call this tool — answer with a free-text clarifying "
            "question in your reply instead. One-option pickers are not "
            "valid: the user has nothing to pick between. "
            "Do NOT include an 'Other' option — the UI adds free-text "
            "automatically. "
            "Numeric questions (numeric != null) MUST leave this empty: "
            "the modal renders an Input widget instead of options."
        ),
    )
    multi_select: bool = Field(
        default=False,
        description=(
            "Set true ONLY when choices are genuinely not mutually "
            "exclusive (e.g. 'which symptoms apply'). Default false. "
            "Must be false when numeric is set."
        ),
    )
    numeric: NumericSpec | None = Field(
        default=None,
        description=(
            "Set for numeric answers (age in years, pain on 0-10 scale). "
            "When set, options MUST be empty and multi_select MUST be false."
        ),
    )

    @model_validator(mode="after")
    def _options_shape_matches_numeric_flag(self) -> "Question":
        if self.numeric is None:
            # 2-4 is the UX sweet-spot for tool-generated questions; the hard
            # upper bound is 20 to accommodate dataset-driven categorical
            # evidences that can have O(10) values (e.g. DDXPlus travel regions).
            if not (2 <= len(self.options) <= 20):
                raise ValueError(
                    f"Categorical question must have 2-20 options, "
                    f"got {len(self.options)}."
                )
        else:
            # Numeric question — must have no options and not multi_select.
            if self.options:
                raise ValueError(
                    "Numeric questions must leave options empty; "
                    "the modal renders an Input widget, not a picker."
                )
            if self.multi_select:
                raise ValueError("Numeric questions cannot use multi_select.")
        return self

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

    ``numeric_values`` carries answers to numeric questions in the same
    payload (D10 initial-batch can mix a numeric ``age`` with a
    single-select ``sex`` in one modal). ``answers[q]`` is set to ``""``
    for numeric questions; the actual value lives in ``numeric_values[q]``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    answers: dict[str, str | list[str]] = Field(
        ...,
        description="Mapping of question text to user-selected label(s).",
    )
    numeric_values: dict[str, float | int] = Field(
        default_factory=dict,
        description=(
            "Mapping of question text to the user's numeric answer for "
            "questions with NumericSpec set. Empty for purely categorical "
            "modals."
        ),
    )
