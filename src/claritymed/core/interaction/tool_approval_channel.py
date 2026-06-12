"""``ToolApprovalChannel`` — abstract transport for the PHI-write gate.

When pydantic-ai's ``ApprovalRequiredToolset`` raises
``ApprovalRequired`` for a tool call, the agent run returns a
``DeferredToolRequests`` payload. ``AskService`` walks those calls and
asks the host's channel for a decision per call. The channel knows how
to render the modal (Textual app, web socket, headless prompt) and
returns a structured ``ApprovalDecision`` back into the orchestrator,
which in turn builds ``DeferredToolResults`` for the resume call.

Hosts plug in their own transport:

* ``cli.tui.tool_approval_channel.TextualToolApprovalChannel`` pushes
  a ``ToolApprovalModal`` and awaits its dismissal.
* ``HeadlessToolApprovalChannel`` (here) denies every call — safe
  default for one-shot CLI, evals, and tests where there is no human
  in the loop to approve PHI writes. Callers that want to opt out of
  the gate (e.g. ``claritymed tool <name> --auto-approve`` in
  ``cli/commands/tool.py``) bypass the channel entirely rather than
  installing a permissive variant.

The protocol mirrors ``PromptChannel`` deliberately: same shape, same
cancellation contract, same separation between "unavailable" (no UI)
and "declined" (user is present, said no). Keeping them parallel makes
the TUI wiring symmetric and lets tests share fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from claritymed.core.interaction.prompt_channel import InteractiveChannelUnavailable

Decision = Literal["once", "always_tool", "always_pattern", "modify", "deny"]


@dataclass(frozen=True)
class ApprovalDecision:
    """The user's response to one tool-approval prompt.

    ``modified_args`` is set only when ``decision == "modify"`` and the
    UI supports the per-field edit form (v1 of the TUI modal returns
    ``None`` — the field UI ships post-Unit-7). Downstream
    ``AskService`` treats ``modify`` with ``None`` as "approve once
    with original args", since the user committed to running the call
    but didn't actually edit anything.
    """

    decision: Decision
    modified_args: dict[str, Any] | None = None


@runtime_checkable
class ToolApprovalChannel(Protocol):
    """Per-call transport for PHI tool approval prompts."""

    async def request(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        breadcrumb: str | None = None,
    ) -> ApprovalDecision:
        """Render the approval prompt and resolve with the user's decision.

        Implementations must:

        * Resolve with an ``ApprovalDecision`` when the user picks an
          option (including ``deny`` — that is still a decision).
        * Raise ``InteractiveChannelUnavailable`` when no UI exists
          (one-shot CLI, eval runs). ``AskService`` translates that
          into a ``ToolDenied`` result so the model sees a clean
          refusal rather than the whole run aborting.
        * Propagate ``asyncio.CancelledError`` unchanged. ``AskService``
          catches it, audits ``tool.cancelled_by_shutdown``, and
          returns ``ToolDenied`` for every still-pending call so the
          agent loop terminates rather than hanging on a torn-down
          modal.

        ``breadcrumb`` is a host-rendered hint like ``"Tool 2/5"`` that
        shows the user their position in a multi-tool approval batch.
        Channels that have no UI ignore it.
        """
        ...


class HeadlessToolApprovalChannel:
    """Default channel for non-interactive contexts.

    Raises ``InteractiveChannelUnavailable`` on every call. ``AskService``
    catches the exception and records a ``ToolDenied`` for the
    corresponding tool call, so a headless eval that triggers an
    approval-required tool path sees a clean denial instead of stalling
    on a UI that does not exist.
    """

    async def request(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        breadcrumb: str | None = None,
    ) -> ApprovalDecision:
        raise InteractiveChannelUnavailable(
            f"No tool-approval channel attached: cannot prompt for {tool_name!r}."
        )
