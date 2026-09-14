"""Textual implementation of ``ToolApprovalChannel``.

Wraps ``ToolApprovalModal`` behind the orchestrator-facing channel
protocol. ``AskService`` calls ``await channel.request(...)`` for each
``DeferredToolRequests.approvals`` entry; we push the modal and wait
for the user to dismiss it with one of the three decisions
(``once`` / ``always_tool`` / ``deny``).

Modal dismissed with ``None`` is treated as ``deny`` — Textual hands
back ``None`` for stack-pop on app shutdown or programmatic dismiss
without a value. Denying-by-default is safer than approving-by-default
when the gate is the PHI write defense.

Cancellation:

* If ``push_screen_wait`` raises (no running screen, app teardown),
  re-raise as ``InteractiveChannelUnavailable`` so ``AskService``
  records a clean ``ToolDenied`` instead of letting the agent loop
  hang on a torn-down modal.
* If the consumer cancels (TUI Esc on the whole turn), the awaiting
  ``CancelledError`` propagates unchanged — ``AskService`` already
  audits ``tool.cancelled_by_shutdown`` and short-circuits the
  remaining approvals.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from claritymed.cli.tui.modals.tool_approval_modal import ToolApprovalModal
from claritymed.core.interaction import (
    ApprovalDecision,
    InteractiveChannelUnavailable,
)

if TYPE_CHECKING:
    from textual.app import App

logger = logging.getLogger(__name__)


class TextualToolApprovalChannel:
    """Bridge a deferred tool-approval request to a Textual modal."""

    def __init__(self, app: "App", *, language: str = "en") -> None:
        self._app = app
        self._language = language

    async def request(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        breadcrumb: str | None = None,
    ) -> ApprovalDecision:
        logger.debug(
            "TextualToolApprovalChannel.request: ENTER tool=%s breadcrumb=%r",
            tool_name,
            breadcrumb,
        )
        try:
            t0 = time.monotonic()
            result = await self._app.push_screen_wait(
                ToolApprovalModal(
                    tool_name, args, breadcrumb=breadcrumb, language=self._language
                )
            )
            logger.debug(
                "TextualToolApprovalChannel.request: returned after %.0fms decision=%r",
                (time.monotonic() - t0) * 1000,
                getattr(result, "decision", None),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TextualToolApprovalChannel.request: push_screen_wait RAISED %s: %s",
                type(exc).__name__,
                exc,
            )
            raise InteractiveChannelUnavailable(
                f"Textual tool-approval channel failed: {exc}"
            ) from exc

        if result is None:
            # Programmatic dismiss / app teardown — treat as explicit deny
            # so the deferred-result loop produces a ToolDenied and the
            # agent terminates cleanly rather than hanging on an empty
            # response.
            logger.debug(
                "TextualToolApprovalChannel.request: modal returned None → deny"
            )
            return ApprovalDecision(decision="deny")
        return result
