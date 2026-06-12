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
from textual.widgets import Label, OptionList, SelectionList
from textual.widgets.option_list import Option

from claritymed.core.i18n import t
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
    Question,
    QuestionOption,
)

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
                badge_key = (
                    "ask_modal.badge_multi"
                    if q.multi_select
                    else "ask_modal.badge_single"
                )
                yield Label(t(badge_key), classes="chip-mode")
                yield Label(
                    f"{self._page + 1} / {len(self._payload.questions)}",
                    id="progress",
                )
            yield Label(q.question, id="question-text")
            if q.multi_select:
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
        self._answers[self._current_question.question] = selection
        if self._is_last_page:
            logger.debug(
                "QuestionModal.action_advance: last page, dismissing with %d answers",
                len(self._answers),
            )
            self.dismiss(AskUserQuestionResult(answers=self._answers))
            return
        self._page += 1
        logger.debug(
            "QuestionModal.action_advance: advancing to page=%d/%d",
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
        """Render one picker row: ``label — description``."""
        return f"{opt.label} — {opt.description}"
