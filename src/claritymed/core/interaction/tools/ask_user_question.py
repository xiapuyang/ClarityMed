"""Tool: ``ask_user_question`` — surface a structured question to the user.

Thin pydantic-ai shim that awaits the per-turn ``PromptChannel`` attached
to deps. The channel decides how to render (Textual modal, future web
panel, headless stub). Validation of the LLM's payload is delegated to
the ``AskUserQuestionInput`` schema — pydantic-ai's tool-retry loop
gives the model up to ``max_retries`` chances to fix a malformed call
before propagating the error.

Return shape is always a string:

* Happy path: JSON dump of ``AskUserQuestionResult`` (the answers map).
* Channel unavailable: short hint asking the LLM to fall back to plain
  text. The model has been told in its tool description not to retry.
* User declined: short hint telling the LLM the user opted out so it
  can decide whether to re-ask differently or proceed without.

PHI: the modal lets the user pick a structured option OR type free text
under "Other". Any string the user typed counts as free text and MUST
flow through ``PhiGuard.scrub_free_text`` before going back to the LLM.
Picked labels come from the LLM itself, so they cannot carry user PHI;
the scrub still runs over them defensively (it's a regex-cheap pass).
"""

from __future__ import annotations

import logging

from pydantic_ai import RunContext, Tool

from claritymed.core.interaction.prompt_channel import (
    InteractiveChannelUnavailable,
    UserDeclinedAnswer,
)
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
)
from claritymed.core.prompts.registry import PromptRegistry
from claritymed.core.turn_state import TurnState

logger = logging.getLogger(__name__)

ASK_USER_QUESTION_TOOL_NAME = "ask_user_question"
ASK_USER_QUESTION_PROMPT_NAME = "ask_user_question_tool"

# Default cap on how many times pydantic-ai will retry the tool call when
# the LLM emits a malformed payload (e.g. 5 options when max is 4). Two
# retries is enough to recover from a transient schema mistake without
# letting a confused model spam the user with question modals.
_DEFAULT_MAX_RETRIES = 2


async def ask_user_question_body(
    deps: TurnState,
    payload: AskUserQuestionInput,
) -> str:
    """Standalone tool body — testable without constructing a ``RunContext``.

    The pydantic-ai ``Tool`` wrapper built by ``build_ask_user_question_tool``
    is a thin shim that extracts ``ctx.deps`` and calls this. All the
    business logic (channel lookup, error translation, audit counter,
    Steps-panel events) lives here so unit tests can exercise it directly
    with a fake deps namespace.
    """
    # Local import keeps ``core.events`` out of import time — it pulls in
    # pydantic models that are only needed when a tool actually runs.
    from claritymed.core.events import ToolCompleted, ToolStarted

    # Count the call before doing anything else so audit can spot
    # "model called us but the channel was missing" patterns.
    deps.tool_calls[ASK_USER_QUESTION_TOOL_NAME] = (
        deps.tool_calls.get(ASK_USER_QUESTION_TOOL_NAME, 0) + 1
    )

    # Surface the call in the Steps panel — same pattern as
    # ``retrieve_medical_literature``. ``args_preview`` is the header
    # of the first question, which is the most readable signal the LLM
    # produces (chip-style, ≤12 chars, mirrors what the modal will show).
    eq = deps.event_queue
    first_header = payload.questions[0].header if payload.questions else ""
    await eq.put(
        ToolStarted(
            tool_name=ASK_USER_QUESTION_TOOL_NAME,
            args_preview=first_header[:60],
        )
    )

    async def _complete(summary: str) -> None:
        await eq.put(
            ToolCompleted(tool_name=ASK_USER_QUESTION_TOOL_NAME, summary=summary)
        )

    channel = getattr(deps, "prompt_channel", None)
    if channel is None:
        logger.debug(
            "ask_user_question_body: no channel attached, returning unavailable hint"
        )
        await _complete("no channel")
        return (
            "[ask_user_question.unavailable] No prompt channel attached "
            "to this run. Answer in plain text using your best "
            "understanding."
        )

    logger.debug(
        "ask_user_question_body: ENTER channel.ask() channel=%s q_count=%d",
        type(channel).__name__,
        len(payload.questions),
    )
    try:
        result = await channel.ask(payload)
    except InteractiveChannelUnavailable as exc:
        logger.debug("ask_user_question_body: channel unavailable: %s", exc)
        await _complete("channel unavailable")
        return f"[ask_user_question.unavailable] {exc}"
    except UserDeclinedAnswer:
        logger.debug("ask_user_question_body: user declined")
        await _complete("declined")
        return (
            "[ask_user_question.declined] The user dismissed the question modal "
            "without answering. Do NOT answer the question yourself. "
            "Acknowledge that they skipped it and stop."
        )
    except Exception:  # noqa: BLE001
        # A channel raising something unexpected (UI bug, transport
        # error) must not crash the whole agent run — translate it
        # to a recoverable tool result instead.
        logger.exception("ask_user_question_body: prompt channel raised unexpectedly")
        await _complete("error")
        return (
            "[ask_user_question.error] The question UI failed. "
            "Answer in plain text using your best understanding."
        )

    logger.debug(
        "ask_user_question_body: RETURN channel.ask() answered=%d",
        len(result.answers),
    )
    scrubbed = _scrub_result(result)
    await _complete(f"answered ({len(scrubbed.answers)})")
    return scrubbed.model_dump_json()


def _scrub_result(result: AskUserQuestionResult) -> AskUserQuestionResult:
    """Scrub PII from user-supplied free text in the result map.

    The user may have typed free text under the modal's auto-injected
    "Other" option. That string flows back to the LLM verbatim unless we
    scrub it here, breaking the cloud-PHI invariant on the next LLM hop.
    ``PhiGuard.from_config`` reads ``configs/safety.yaml`` for the same
    rules ``AskService`` uses on initial input, so the two paths cannot
    drift.
    """
    from claritymed.core.phi.guard import get_default_guard

    guard = get_default_guard()

    def _scrub_one(value: str) -> str:
        scrubbed, _report = guard.scrub_free_text(value)
        return scrubbed

    cleaned: dict[str, str | list[str]] = {}
    for question, answer in result.answers.items():
        if isinstance(answer, list):
            cleaned[question] = [_scrub_one(a) for a in answer]
        else:
            cleaned[question] = _scrub_one(answer)
    return AskUserQuestionResult(answers=cleaned)


def build_ask_user_question_tool(
    registry: PromptRegistry,
    *,
    language: str = "en",
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> Tool:
    """Build a registered ``Tool`` instance for the ask agent.

    Args:
        registry: PromptRegistry the tool description is read from.
            Description is bilingual; ``language`` picks one.
        language: ``"en"`` or ``"zh"`` — selects the description text.
        max_retries: pydantic-ai retry budget for malformed payloads.

    Returns:
        A ``pydantic_ai.Tool`` ready to be passed to ``Agent(tools=[...])``.
    """
    description = registry.get(
        ASK_USER_QUESTION_PROMPT_NAME,
        language=language,  # type: ignore[arg-type]
    )

    async def ask_user_question(
        ctx: RunContext[TurnState],
        payload: AskUserQuestionInput,
    ) -> str:
        """Surface a structured question to the user and return their answers."""
        return await ask_user_question_body(ctx.deps, payload)

    return Tool(
        ask_user_question,
        takes_ctx=True,
        name=ASK_USER_QUESTION_TOOL_NAME,
        description=description,
        max_retries=max_retries,
    )
