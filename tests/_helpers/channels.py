"""Scriptable PromptChannel / ToolApprovalChannel transports for tests.

When a test needs to drive a modal flow that would normally require a Textual
UI (initial-batch numeric+categorical, follow-up confirm, PHI tool approval),
plug in :class:`AutoAnswerChannel` or :class:`AutoApprovalChannel`. They
satisfy the structural :class:`~claritymed.core.interaction.prompt_channel.PromptChannel`
/ :class:`~claritymed.core.interaction.tool_approval_channel.ToolApprovalChannel`
Protocols that production code expects, but return scripted answers from a
queue instead of rendering anything.

Why a queue and not header-keyed scripting: every modal flow in the project
is turn-driven (confirm → initial-batch → server question 1 → …); position
is enough to identify "the next answer" and keeps test setup readable. Add
keyed scripting only when a real test needs out-of-order matching.

Three older one-off stub channels exist in
``tests/orchestrator/features/test_symptoms_plugin.py``,
``tests/orchestrator/features/test_symptoms_plugin_e2e.py``, and
``tests/orchestrator/services/test_ask_service_approvals.py``. New tests
should reuse these helpers rather than growing a fourth.
"""

from __future__ import annotations

from typing import Any

from claritymed.core.interaction.prompt_channel import (
    InteractiveChannelUnavailable,
    UserDeclinedAnswer,
)
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
)
from claritymed.core.interaction.tool_approval_channel import ApprovalDecision

AnswerOrError = AskUserQuestionResult | Exception
DecisionOrError = ApprovalDecision | Exception


class AutoAnswerChannel:
    """Queue-driven scripted ``PromptChannel``.

    Pass a list of :class:`AskUserQuestionResult` instances (and optionally
    :class:`Exception` instances to raise) at construction time. The k-th
    ``ask()`` call returns or raises the k-th item.

    Exhausting the queue raises ``AssertionError`` so a test bug surfaces as
    a failure rather than a hang. Inspect ``.calls`` after the run to assert
    the modal flow the orchestrator drove.

    Common exception cases callers script:

    * :class:`UserDeclinedAnswer` — the user dismissed the modal mid-flow.
    * :class:`InteractiveChannelUnavailable` — no UI exists this turn (the
      headless fallback path).
    """

    def __init__(self, answers: list[AnswerOrError]) -> None:
        self._answers: list[AnswerOrError] = list(answers)
        self.calls: list[AskUserQuestionInput] = []

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        self.calls.append(payload)
        if not self._answers:
            raise AssertionError(
                "AutoAnswerChannel.ask called with no scripted answers left "
                f"(received payload: {[q.header for q in payload.questions]})."
            )
        nxt = self._answers.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class AutoApprovalChannel:
    """Queue-driven scripted ``ToolApprovalChannel``.

    Mirrors :class:`AutoAnswerChannel` for PHI tool approval prompts: pass a
    queue of :class:`ApprovalDecision` (or :class:`Exception`) and inspect
    ``.calls`` after the run for the ``(tool_name, args, breadcrumb)``
    tuples the orchestrator surfaced.
    """

    def __init__(self, decisions: list[DecisionOrError]) -> None:
        self._decisions: list[DecisionOrError] = list(decisions)
        self.calls: list[tuple[str, dict[str, Any], str | None]] = []

    async def request(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        breadcrumb: str | None = None,
    ) -> ApprovalDecision:
        self.calls.append((tool_name, dict(args), breadcrumb))
        if not self._decisions:
            raise AssertionError(
                "AutoApprovalChannel.request called with no scripted "
                f"decisions left (received tool_name={tool_name!r})."
            )
        nxt = self._decisions.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


__all__ = [
    "AutoAnswerChannel",
    "AutoApprovalChannel",
    "AnswerOrError",
    "DecisionOrError",
    "ApprovalDecision",
    "InteractiveChannelUnavailable",
    "UserDeclinedAnswer",
]
