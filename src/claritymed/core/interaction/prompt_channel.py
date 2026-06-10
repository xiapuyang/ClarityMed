"""``PromptChannel`` — abstract transport that hands a question to a UI.

The tool body knows nothing about Textual, Web, or stdin. It just awaits
``channel.ask(payload)`` and returns whatever the channel resolves with.
Hosts plug in their own transport:

* ``cli.tui.prompt_channel.TextualPromptChannel`` pushes a Textual
  ``QuestionModal`` and awaits its dismissal.
* ``HeadlessPromptChannel`` (here) is the safe default for one-shot CLI
  ``ask``, evals, and tests — no UI to render, so it raises
  ``InteractiveChannelUnavailable`` immediately. The tool body catches
  that and returns a user-facing string to the LLM so the model can fall
  back to best-effort answering instead of stalling.

Why a Protocol rather than an ABC: there's no shared implementation to
inherit. Hosts just need to provide an async ``ask``; structural typing
keeps the surface flat and avoids forcing every test fake to subclass.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
)


class InteractiveChannelUnavailable(RuntimeError):
    """Raised by channels that cannot render a UI in the current process.

    The ``ask_user_question`` tool catches this and translates it into a
    user-facing tool result string, so the LLM sees a graceful "no
    channel" signal rather than the whole run aborting.
    """


class UserDeclinedAnswer(RuntimeError):
    """Raised by interactive channels when the user dismisses without choosing.

    Distinct from ``InteractiveChannelUnavailable``: the user is present
    and chose not to answer (Esc, cancel button). The tool body
    translates this into a different message so the LLM can decide
    whether to re-ask, proceed without, or fall back.
    """


@runtime_checkable
class PromptChannel(Protocol):
    """Per-turn transport for ``ask_user_question`` tool calls."""

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        """Render ``payload`` to the user and resolve with their answers.

        Implementations must:
        * Resolve with a complete ``AskUserQuestionResult`` (one answer
          per question in ``payload.questions``) when the user submits.
        * Raise ``UserDeclinedAnswer`` when the user cancels.
        * Raise ``InteractiveChannelUnavailable`` when no UI exists.
        """
        ...


class HeadlessPromptChannel:
    """Default channel for non-interactive contexts (one-shot CLI, evals, tests).

    Always raises ``InteractiveChannelUnavailable``. The tool catches it
    and reports back to the LLM in plain text — the model then continues
    with its best guess rather than dead-ending the turn.
    """

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        msg = (
            "No interactive channel attached to this run; ask the user "
            "in plain text instead of calling this tool."
        )
        raise InteractiveChannelUnavailable(msg)
