"""Textual implementation of ``PromptChannel``.

The agent loop and the Textual app share one asyncio loop, so awaiting
``app.push_screen_wait`` from inside a pydantic-ai tool body just yields
control back to Textual; the user sees the modal, picks an answer, and
the awaiting coroutine resumes with whatever the modal dismissed with.
No threads, no queues, no signalling beyond the modal's own ``dismiss``.

After the modal closes, the channel writes a short ``system`` turn into
the Conversation widget so the user (and any future reader of the chat
log) can see what they were asked and what they chose. Without that
trace, the next LLM response looks like it pulled a fact out of thin air
because the structured Q&A round-trip lives only inside pydantic-ai's
message history, which the TUI does not render.

Cancellation contract:

* Modal ``dismiss(result)``    → channel returns ``result`` and writes a
  ``↳ 你的回答 / Your answer`` system turn.
* Modal ``dismiss(None)``      → channel raises ``UserDeclinedAnswer``
  after writing a ``↳ 已跳过追问 / Skipped`` system turn.
* Anything else (UI bug)       → channel raises ``InteractiveChannelUnavailable``
  with the original cause attached so the tool body can log it.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from claritymed.cli.tui.modals.question_modal import QuestionModal, _fmt
from claritymed.cli.tui.widgets.conversation import Conversation
from claritymed.core.i18n import t
from claritymed.core.interaction.prompt_channel import (
    InteractiveChannelUnavailable,
    UserDeclinedAnswer,
)
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
)

if TYPE_CHECKING:
    from textual.app import App

logger = logging.getLogger(__name__)


class TextualPromptChannel:
    """Bridge a pydantic-ai tool call to a Textual modal screen."""

    def __init__(self, app: "App") -> None:
        self._app = app

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        q_count = len(payload.questions)
        logger.debug(
            "TextualPromptChannel.ask: ENTER push_screen_wait q_count=%d headers=%r",
            q_count,
            [q.header for q in payload.questions],
        )
        try:
            _t0 = time.monotonic()
            result = await self._app.push_screen_wait(QuestionModal(payload))
            logger.debug(
                "TextualPromptChannel.ask: push_screen_wait returned after %.0fms",
                (time.monotonic() - _t0) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TextualPromptChannel.ask: push_screen_wait RAISED after %.0fms %s: %s",
                (time.monotonic() - _t0) * 1000,
                type(exc).__name__,
                exc,
            )
            # ``push_screen_wait`` should not raise in normal use; if it
            # does (no running screen, app shutting down), treat it as
            # the channel being unavailable so the tool body can fall
            # back to a plain-text hint to the LLM.
            raise InteractiveChannelUnavailable(
                f"Textual prompt channel failed: {exc}"
            ) from exc
        logger.debug(
            "TextualPromptChannel.ask: push_screen_wait RETURNED result=%r",
            result,
        )
        if result is None:
            self._post_declined_note(payload)
            raise UserDeclinedAnswer("User dismissed the question without answering.")
        self._post_answered_note(payload, result)
        return result

    # ------------------------------------------------------------------
    # Chat-trace helpers
    # ------------------------------------------------------------------

    def _post_answered_note(
        self,
        payload: AskUserQuestionInput,
        result: AskUserQuestionResult,
    ) -> None:
        """Append a localized ``Your answer / 你的回答`` system turn.

        Layout matches Claude Code's ``SubmitQuestionsView`` — each
        question is a bullet line, each answer is an arrow line below
        it, so the user can scan ``what was asked → what they picked``
        without re-opening the modal.
        """
        header = t("ask_modal.answer_header")
        lines: list[str] = []
        for q in payload.questions:
            picked = result.answers.get(q.question)
            if picked is None:
                continue
            lines.append(t("ask_modal.answer_question", question=q.question))
            lines.append(t("ask_modal.answer_value", value=self._render_pick(picked)))
        body = header + ("\n" + "\n".join(lines) if lines else "")
        self._add_system_turn(body)

    def _post_declined_note(self, payload: AskUserQuestionInput) -> None:
        """Append a localized declined system turn matching the screenshot format.

        Numeric questions render their range (``0-120 years``) instead of
        the empty parens that ``options`` would produce — without this
        branch the user sees ``Age（）`` on cancel, which mis-suggests the
        modal was blank.
        """
        lines: list[str] = [t("ask_modal.declined_header")]
        for q in payload.questions:
            if q.numeric is not None:
                spec = q.numeric
                bounds = f"{_fmt(spec.min)}-{_fmt(spec.max)}"
                rng = f"{bounds} {spec.unit}" if spec.unit else bounds
                lines.append(
                    t(
                        "ask_modal.declined_question_numeric",
                        question=q.question,
                        range=rng,
                    )
                )
                continue
            opts = " / ".join(opt.label for opt in q.options)
            lines.append(
                t("ask_modal.declined_question", question=q.question, options=opts)
            )
        self._add_system_turn("\n".join(lines))

    @staticmethod
    def _render_pick(picked: str | list[str]) -> str:
        if isinstance(picked, list):
            return ", ".join(picked) if picked else t("ask_modal.multi_empty")
        return picked

    def _add_system_turn(self, text: str) -> None:
        """Insert a system-turn bubble into the active Conversation widget.

        Writing to the widget directly is fine because the Textual app
        and the pydantic-ai tool body share one asyncio loop — there is
        no thread boundary to cross. A try/except wraps it so a missing
        Conversation (degenerate test scaffold) never breaks the agent
        run; the tool result already carries the data the LLM needs.
        """
        try:
            self._app.query_one(Conversation).add_system_turn(text)
        except Exception:  # noqa: BLE001
            pass
