"""Web-side ``PromptChannel`` / ``ToolApprovalChannel`` implementations.

Each in-flight interaction has one entry in the per-app rendezvous map:

    interactions[interaction_id] = {
        "session_id": str,
        "user_id": str,
        "kind": "ask_user_question" | "tool_approval",
        "future": asyncio.Future,    # resolves with parsed response
        "ask_payload": AskUserQuestionInput | None,   # for question kind
    }

The channel:

1. Mints an ``interaction_id`` (UUID4 hex).
2. Pushes ``InteractionRequested`` into the stream's emit queue.
3. Awaits the future. The future is set by
   ``POST /sessions/{id}/interactions/{interaction_id}`` after schema
   validation, or cancelled when the stream tears down.
4. Translates the future's result into the channel's contract type
   (``AskUserQuestionResult`` / ``ApprovalDecision``) and returns it.

The rendezvous map lives on ``app.state.web_interactions`` so the
``POST /interactions`` endpoint (defined in the chat router) can find
the pending future without going through a per-stream global.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from claritymed.core.events import Event, InteractionRequested
from claritymed.core.interaction import (
    ApprovalDecision,
    InteractiveChannelUnavailable,
    UserDeclinedAnswer,
)
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
)

logger = logging.getLogger(__name__)


class WebPromptChannel:
    """``PromptChannel`` impl that pauses the stream until the SPA replies.

    Constructed once per stream. The same instance is reused if the LLM
    invokes ``ask_user_question`` multiple times in a single turn —
    each invocation mints a fresh interaction_id.
    """

    def __init__(
        self,
        *,
        user_id: str,
        session_id: str,
        emit_queue: asyncio.Queue[Event],
        rendezvous: dict[str, dict[str, Any]],
    ) -> None:
        self._user_id = user_id
        self._session_id = session_id
        self._emit_queue = emit_queue
        self._rendezvous = rendezvous

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        interaction_id = uuid.uuid4().hex
        self._rendezvous[interaction_id] = {
            "session_id": self._session_id,
            "user_id": self._user_id,
            "kind": "ask_user_question",
            "future": future,
            "ask_payload": payload,
        }
        await self._emit_queue.put(
            InteractionRequested(
                interaction_id=interaction_id,
                kind="ask_user_question",
                payload=payload.model_dump(mode="json"),
            )
        )
        try:
            raw = await future
        except asyncio.CancelledError:
            # Stream cancellation drops the rendezvous entry. Reraise so
            # the surrounding tool body's UserDeclinedAnswer path is not
            # bypassed — the LLM sees a "no answer" tool result instead
            # of an empty hang.
            self._rendezvous.pop(interaction_id, None)
            raise UserDeclinedAnswer("interaction cancelled") from None
        # Validate at the channel boundary so the tool body sees a
        # well-typed result regardless of which UI submitted it.
        try:
            return AskUserQuestionResult.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ask_user_question response failed validation")
            raise UserDeclinedAnswer(f"invalid response shape: {exc}") from None


class WebToolApprovalChannel:
    """``ToolApprovalChannel`` impl matched to ``WebPromptChannel``."""

    def __init__(
        self,
        *,
        user_id: str,
        session_id: str,
        emit_queue: asyncio.Queue[Event],
        rendezvous: dict[str, dict[str, Any]],
    ) -> None:
        self._user_id = user_id
        self._session_id = session_id
        self._emit_queue = emit_queue
        self._rendezvous = rendezvous

    async def request(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        breadcrumb: str | None = None,
    ) -> ApprovalDecision:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        interaction_id = uuid.uuid4().hex
        self._rendezvous[interaction_id] = {
            "session_id": self._session_id,
            "user_id": self._user_id,
            "kind": "tool_approval",
            "future": future,
            "tool_name": tool_name,
        }
        await self._emit_queue.put(
            InteractionRequested(
                interaction_id=interaction_id,
                kind="tool_approval",
                payload={
                    "tool_name": tool_name,
                    "args": _safe_args(args),
                    "breadcrumb": breadcrumb or "",
                },
            )
        )
        try:
            raw = await future
        except asyncio.CancelledError:
            self._rendezvous.pop(interaction_id, None)
            # Match the headless contract: cancellation surfaces as
            # "no channel" so AskService records a clean ToolDenied
            # instead of hanging.
            raise InteractiveChannelUnavailable(
                "approval interaction cancelled"
            ) from None
        decision = (raw or {}).get("decision")
        if decision not in {"once", "always_tool", "deny"}:
            logger.warning(
                "approval response had unknown decision %r; treating as deny",
                decision,
            )
            return ApprovalDecision(decision="deny")
        return ApprovalDecision(decision=decision)


def build_web_channels(
    *,
    user_id: str,
    session_id: str,
    emit_queue: asyncio.Queue[Event],
    rendezvous: dict[str, dict[str, Any]],
) -> tuple[WebPromptChannel, WebToolApprovalChannel]:
    """Construct both channels with a shared rendezvous + emit queue."""
    prompt = WebPromptChannel(
        user_id=user_id,
        session_id=session_id,
        emit_queue=emit_queue,
        rendezvous=rendezvous,
    )
    approval = WebToolApprovalChannel(
        user_id=user_id,
        session_id=session_id,
        emit_queue=emit_queue,
        rendezvous=rendezvous,
    )
    return prompt, approval


def _safe_args(args: dict[str, Any]) -> dict[str, Any]:
    """Cap any string field in ``args`` so a PHI-laden value can't blow up the SSE frame.

    Approval modals only need a glance at what the LLM is about to write;
    the full payload still lives in the audit row and the eventual
    SaveRecord tool body. 256 chars is enough for "Patient: J Doe, allergy: peanut"
    style summaries without risking a multi-kilobyte modal title.
    """
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 256:
            out[k] = v[:253] + "..."
        else:
            out[k] = v
    return out
