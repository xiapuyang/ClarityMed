"""``QuestionModal`` numeric-branch tests.

Two layers:

* Pure helper tests for ``_validate_numeric`` / ``_numeric_range_hint``
  — cheap and exhaustive.
* Pilot-driven integration tests confirming the modal accepts a valid
  value, blocks an invalid one, and writes ``numeric_values`` into the
  dismiss payload per the Unit 11 canonical shape.
"""

from __future__ import annotations

import pytest
from textual.app import App
from textual.widgets import Input, Label

from claritymed.cli.tui.modals.question_modal import (
    QuestionModal,
    _fmt,
    _numeric_range_hint,
    _validate_numeric,
)
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
    NumericSpec,
    Question,
    QuestionOption,
)


# --- helper unit tests -----------------------------------------------------


def test_fmt_strips_trailing_zero_for_integer_bounds() -> None:
    assert _fmt(120) == "120"
    assert _fmt(0.0) == "0"
    assert _fmt(7.5) == "7.5"


def test_numeric_range_hint_with_unit() -> None:
    spec = NumericSpec(min=0, max=120, step=1, unit="years")
    assert _numeric_range_hint(spec) == "0-120 years"


def test_numeric_range_hint_without_unit() -> None:
    spec = NumericSpec(min=0, max=10, step=1)
    assert _numeric_range_hint(spec) == "0-10"


def test_validate_accepts_integer_in_range() -> None:
    spec = NumericSpec(min=0, max=120, step=1, unit="years")
    value, error = _validate_numeric("35", spec)
    assert error is None
    assert value == 35
    assert isinstance(value, int)


def test_validate_rejects_out_of_range() -> None:
    spec = NumericSpec(min=0, max=120, step=1, unit="years")
    value, error = _validate_numeric("130", spec)
    assert value is None
    assert error is not None
    assert "0" in error and "120" in error


def test_validate_rejects_non_numeric() -> None:
    spec = NumericSpec(min=0, max=10, step=1)
    value, error = _validate_numeric("not a number", spec)
    assert value is None
    assert error is not None


def test_validate_accepts_fractional_step() -> None:
    spec = NumericSpec(min=0, max=10, step=0.5)
    value, error = _validate_numeric("7.5", spec)
    assert error is None
    assert value == pytest.approx(7.5)
    assert isinstance(value, float)


def test_validate_rejects_off_step_fractional() -> None:
    spec = NumericSpec(min=0, max=10, step=0.5)
    value, error = _validate_numeric("7.3", spec)
    assert value is None
    assert error is not None
    assert "0.5" in error


def test_validate_accepts_lower_and_upper_bound() -> None:
    spec = NumericSpec(min=0, max=120, step=1)
    assert _validate_numeric("0", spec)[1] is None
    assert _validate_numeric("120", spec)[1] is None


# --- Pilot-driven integration tests ----------------------------------------


def _numeric_payload(
    *,
    min_value: int = 0,
    max_value: int = 120,
    step: int = 1,
    unit: str | None = "years",
) -> AskUserQuestionInput:
    return AskUserQuestionInput(
        questions=[
            Question(
                question="How old are you, in years?",
                header="Age",
                numeric=NumericSpec(min=min_value, max=max_value, step=step, unit=unit),
            )
        ]
    )


def _mixed_payload() -> AskUserQuestionInput:
    """Numeric + categorical in one modal — the D10 initial-batch shape."""
    return AskUserQuestionInput(
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
                    QuestionOption(label="Female", description="Female."),
                    QuestionOption(label="Male", description="Male."),
                ],
            ),
        ]
    )


class _ModalHost(App):
    """Push one modal; capture its dismiss payload."""

    def __init__(self, payload: AskUserQuestionInput) -> None:
        super().__init__()
        self._payload = payload
        self.result: AskUserQuestionResult | None | str = "<unset>"

    def on_mount(self) -> None:
        def _capture(value):
            self.result = value

        self.push_screen(QuestionModal(self._payload), _capture)


@pytest.mark.asyncio
async def test_numeric_modal_accepts_valid_value() -> None:
    app = _ModalHost(_numeric_payload())
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        # Focus the Input then type a valid age.
        picker = modal.query_one("#picker", Input)
        picker.focus()
        await pilot.press("3", "5")
        await pilot.press("enter")
        await pilot.pause()
    assert isinstance(app.result, AskUserQuestionResult)
    assert app.result.numeric_values == {"How old are you, in years?": 35}
    # The canonical Unit 11 shape: the same key in ``answers`` is set to
    # the empty string so PHI scrubbing accounting still touches the
    # numeric question.
    assert app.result.answers == {"How old are you, in years?": ""}


@pytest.mark.asyncio
async def test_numeric_modal_blocks_out_of_range() -> None:
    app = _ModalHost(_numeric_payload())
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        picker = modal.query_one("#picker", Input)
        picker.focus()
        # 130 is outside [0, 120] → submit must be blocked + error shown.
        await pilot.press("1", "3", "0")
        await pilot.press("enter")
        await pilot.pause()
        # Modal still up; result not captured yet.
        assert app.result == "<unset>"
        # Error label visible with the localized message.
        err = modal.query_one("#numeric-error", Label)
        assert err.display is True
        assert "0" in str(err.renderable)


@pytest.mark.asyncio
async def test_numeric_modal_empty_submit_is_noop() -> None:
    """Pressing Enter with no value typed must not close the modal — the
    user is probably still figuring out what to type."""
    app = _ModalHost(_numeric_payload())
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#picker", Input).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.result == "<unset>"


@pytest.mark.asyncio
async def test_numeric_modal_esc_cancels() -> None:
    """Esc must still raise UserDeclinedAnswer-style dismissal — the
    plugin's cancel branch relies on this signal."""
    app = _ModalHost(_numeric_payload())
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#picker", Input).focus()
        await pilot.press("4", "5")
        await pilot.press("escape")
        await pilot.pause()
    assert app.result is None


@pytest.mark.asyncio
async def test_mixed_initial_batch_round_trips_age_and_sex() -> None:
    """The D10 initial-batch shape — numeric page then categorical page —
    produces a result with both ``numeric_values`` and ``answers`` filled."""
    from textual.widgets import OptionList

    app = _ModalHost(_mixed_payload())
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        # Page 1 — age.
        modal.query_one("#picker", Input).focus()
        await pilot.press("4", "2")
        await pilot.press("enter")
        await pilot.pause()
        # Page 2 — sex picker.
        modal = app.screen
        picker = modal.query_one("#picker", OptionList)
        picker.highlighted = 1  # Male
        picker.focus()
        await pilot.press("enter")
        await pilot.pause()
    assert isinstance(app.result, AskUserQuestionResult)
    assert app.result.numeric_values == {"How old are you, in years?": 42}
    assert app.result.answers["Biological sex?"] == "Male"


@pytest.mark.asyncio
async def test_categorical_modal_still_works() -> None:
    """Regression — adding the numeric branch must not break the
    existing single-select path."""
    from textual.widgets import OptionList

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
    app = _ModalHost(payload)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        picker = modal.query_one("#picker", OptionList)
        picker.highlighted = 0
        picker.focus()
        await pilot.press("enter")
        await pilot.pause()
    assert isinstance(app.result, AskUserQuestionResult)
    assert app.result.answers == {"Pick one?": "A"}
    # Numeric map is empty for a purely categorical modal.
    assert app.result.numeric_values == {}
