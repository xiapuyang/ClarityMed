"""Modal that renders an ``AskUserQuestionInput`` and collects answers.

One modal per tool call. Questions are walked one at a time with a
header chip + ``i / N`` indicator, an option picker, and a footer hint
line. UX follows Claude Code's ``AskUserQuestion`` widget:

* Single-select: ``OptionList`` — ↑/↓ moves the highlight, ``Enter``
  locks the highlighted choice and advances (or submits on the last
  page).
* Multi-select: ``SelectionList`` — ↑/↓ moves the highlight, ``Space``
  toggles each option, ``Enter`` submits the page.

No buttons. The modal owns a ``priority=True`` ``Enter`` binding so the
picker widget never absorbs the key (which was the bug that left
``Enter`` doing nothing while a ``RadioSet`` had focus).

Dismissal contracts:

* Submit on the last page → ``dismiss(AskUserQuestionResult(answers=...))``.
* Esc any page → ``dismiss(None)``. The channel translates that into a
  ``UserDeclinedAnswer`` exception so the tool body can return a
  distinct "user declined" hint to the LLM.
"""

from __future__ import annotations

import logging

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, SelectionList
from textual.widgets.option_list import Option

from claritymed.core.i18n import t
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
    NumericSpec,
    Question,
    QuestionOption,
)

# Floating-point tolerance for step validation. Below this, a value is
# treated as "on the step grid" — protects against ``0.1 + 0.2 != 0.3``
# style float jitter for fractional ``step`` configs.
_STEP_EPSILON = 1e-9

logger = logging.getLogger(__name__)


class QuestionModal(ModalScreen[AskUserQuestionResult | None]):
    """Walk the user through 1-4 structured questions, collect their answers."""

    DEFAULT_CSS = """
    QuestionModal {
        align: center middle;
    }
    QuestionModal > Vertical {
        background: $surface;
        border: thick $primary;
        padding: 1 2;
        width: 60;
        max-width: 80%;
        height: auto;
    }
    QuestionModal #header-row {
        height: auto;
        margin-bottom: 1;
    }
    QuestionModal #header-row Label.chip {
        background: $accent;
        color: $text;
        padding: 0 1;
        margin-right: 1;
    }
    QuestionModal #header-row Label.chip-mode {
        background: $boost;
        color: $text-muted;
        padding: 0 1;
        margin-right: 1;
    }
    QuestionModal #question-text {
        margin-bottom: 1;
        height: auto;
    }
    QuestionModal #picker {
        margin-bottom: 1;
        height: auto;
        max-height: 12;
    }
    QuestionModal .hint {
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        # ``priority=True`` makes the modal intercept Enter before the
        # focused picker can absorb it. Without this, OptionList's own
        # "Enter selects" event fires but does not bubble back here as
        # an action — the modal saw nothing and the user was stuck.
        Binding("enter", "advance", "Submit", priority=True),
    ]

    def __init__(self, payload: AskUserQuestionInput) -> None:
        super().__init__()
        self._payload = payload
        self._page = 0
        self._answers: dict[str, str | list[str]] = {}
        # Numeric questions write here in addition to ``_answers`` (which
        # gets the empty string for the same key per the Unit 11 schema
        # contract). Submitted to the LLM via :class:`AskUserQuestionResult`.
        self._numeric_values: dict[str, float | int] = {}

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        logger.debug(
            "QuestionModal.on_mount: modal rendered page=%d/%d q=%r",
            self._page + 1,
            len(self._payload.questions),
            self._current_question.question[:80],
        )

    def compose(self) -> ComposeResult:
        q = self._current_question
        with Vertical():
            with Horizontal(id="header-row"):
                yield Label(q.header, classes="chip")
                yield Label(t(self._badge_key(q)), classes="chip-mode")
                if len(self._payload.questions) > 1:
                    yield Label(
                        f"{self._page + 1} / {len(self._payload.questions)}",
                        id="progress",
                    )
            yield Label(q.question, id="question-text")
            if q.numeric is not None:
                yield Input(
                    placeholder=_numeric_range_hint(q.numeric),
                    type="number",
                    id="picker",
                )
                # Validation error label — hidden until the user types
                # something out of range / off-step.
                err = Label("", id="numeric-error", classes="hint")
                err.display = False
                yield err
                yield Label(t("ask_modal.hint_numeric"), classes="hint")
            elif q.multi_select:
                yield SelectionList[str](
                    *[(self._option_text(opt), opt.label) for opt in q.options],
                    id="picker",
                )
                yield Label(t("ask_modal.hint_multi"), classes="hint")
            else:
                yield OptionList(
                    *[
                        Option(self._option_text(opt), id=f"opt-{i}")
                        for i, opt in enumerate(q.options)
                    ],
                    id="picker",
                )
                yield Label(t("ask_modal.hint_single"), classes="hint")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_cancel(self) -> None:
        logger.debug(
            "QuestionModal.action_cancel: dismissing None (page=%d/%d)",
            self._page + 1,
            len(self._payload.questions),
        )
        self.dismiss(None)

    def action_advance(self) -> None:
        q = self._current_question
        if q.numeric is not None:
            self._handle_numeric_advance(q)
            return
        selection = self._collect_current_selection()
        if selection is None:
            logger.debug(
                "QuestionModal.action_advance: no selection yet, refocusing picker (page=%d/%d)",
                self._page + 1,
                len(self._payload.questions),
            )
            # No choice yet — nudge focus back to the picker rather than
            # advancing with an empty answer (which would lose the
            # question key from the result map).
            self.query_one("#picker").focus()
            return
        self._answers[q.question] = selection
        self._advance_or_submit()

    def _handle_numeric_advance(self, q: Question) -> None:
        """Parse + validate the numeric Input; advance only when valid.

        On parse / range / step failure, surface a localized error on the
        ``#numeric-error`` label and keep focus on the Input — the modal
        does not close until the user enters something acceptable or hits
        Esc.
        """
        assert q.numeric is not None
        picker = self.query_one("#picker", Input)
        raw = picker.value.strip()
        if not raw:
            # Empty submit on a numeric page is a no-op; let the user
            # keep typing rather than swallowing the keystroke.
            return
        value, error = _validate_numeric(raw, q.numeric)
        if error is not None:
            self._set_numeric_error(error)
            picker.focus()
            return
        assert value is not None
        self._numeric_values[q.question] = value
        # Per the Unit 11 contract, ``answers`` still carries the same
        # key (empty string) so the AskService scrub layer doesn't drop
        # numeric questions from its bookkeeping.
        self._answers[q.question] = ""
        self._set_numeric_error("")
        self._advance_or_submit()

    def _set_numeric_error(self, text: str) -> None:
        try:
            label = self.query_one("#numeric-error", Label)
        except Exception:  # noqa: BLE001
            return
        label.update(text)
        label.display = bool(text)

    def _advance_or_submit(self) -> None:
        if self._is_last_page:
            logger.debug(
                "QuestionModal._advance_or_submit: dismissing with %d answers",
                len(self._answers),
            )
            self.dismiss(
                AskUserQuestionResult(
                    answers=self._answers,
                    numeric_values=self._numeric_values,
                )
            )
            return
        self._page += 1
        logger.debug(
            "QuestionModal._advance_or_submit: page %d/%d",
            self._page + 1,
            len(self._payload.questions),
        )
        # ``refresh(recompose=True)`` rebuilds the widgets for the next
        # question; plain ``refresh`` would keep the previous picker.
        self.refresh(recompose=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _current_question(self) -> Question:
        return self._payload.questions[self._page]

    @property
    def _is_last_page(self) -> bool:
        return self._page == len(self._payload.questions) - 1

    def _collect_current_selection(self) -> str | list[str] | None:
        """Return the user's current pick on this page, or None if nothing chosen."""
        q = self._current_question
        if q.multi_select:
            picker_multi = self.query_one("#picker", SelectionList)
            selected = list(picker_multi.selected)
            return selected if selected else None
        picker = self.query_one("#picker", OptionList)
        highlighted = picker.highlighted
        if highlighted is None:
            return None
        if 0 <= highlighted < len(q.options):
            return q.options[highlighted].label
        return None

    @staticmethod
    def _option_text(opt: QuestionOption) -> str:
        """Render one picker row: ``label — description``, or just ``label``
        when the description adds no information beyond the label itself."""
        if opt.description.strip() == opt.label.strip():
            return opt.label
        return f"{opt.label} — {opt.description}"

    @staticmethod
    def _badge_key(q: Question) -> str:
        if q.numeric is not None:
            return "ask_modal.badge_numeric"
        if q.multi_select:
            return "ask_modal.badge_multi"
        return "ask_modal.badge_single"


# --- numeric helpers ------------------------------------------------------


def _numeric_range_hint(spec: NumericSpec) -> str:
    """Render the placeholder shown in the Input widget.

    Mirrors the format used by ``_post_declined_note`` so the user sees
    the same range hint whether they answered or dismissed the modal.
    """
    bounds = f"{_fmt(spec.min)}-{_fmt(spec.max)}"
    return f"{bounds} {spec.unit}" if spec.unit else bounds


def _fmt(value: float | int) -> str:
    """Render a bound without trailing ``.0`` so ``120`` stays ``120``."""
    if isinstance(value, int):
        return str(value)
    if value == int(value):
        return str(int(value))
    return str(value)


def _validate_numeric(
    raw: str, spec: NumericSpec
) -> tuple[float | int | None, str | None]:
    """Parse ``raw`` against ``spec``; return ``(value, error_text)``.

    The error text is already localized — caller drops it straight on
    the ``#numeric-error`` label.
    """
    try:
        value: float | int = float(raw)
    except ValueError:
        return None, t("ask_modal.numeric_invalid")
    if value < spec.min or value > spec.max:
        return None, t(
            "ask_modal.numeric_out_of_range",
            min=_fmt(spec.min),
            max=_fmt(spec.max),
        )
    # Step grid check. ``step=1`` accepts every integer; fractional
    # steps need a tolerance because float math is noisy.
    step = float(spec.step)
    offset = (value - float(spec.min)) / step
    if abs(offset - round(offset)) > _STEP_EPSILON:
        return None, t("ask_modal.numeric_step_mismatch", step=_fmt(spec.step))
    # Preserve int when the spec is integer-shaped — keeps the wire
    # payload cleaner for callers that expect ``int`` (e.g. age).
    if (
        isinstance(spec.min, int)
        and isinstance(spec.max, int)
        and isinstance(spec.step, int)
    ):
        return int(round(value)), None
    return value, None
