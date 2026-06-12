"""Ask mode: LLM-driven agent with on-demand medical literature retrieval.

The ``retrieve_medical_literature`` tool lets the LLM decide per-query
whether to search the knowledge base rather than being forced on every
turn.  Greetings, chitchat, and follow-up clarifications are handled
purely by the LLM; clinical questions trigger a retrieval round-trip.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Union

from claritymed.core.prompts.registry import PromptRegistry
from claritymed.orchestrator.agents.ask_deps import AskDeps

if TYPE_CHECKING:
    from pydantic_ai import Agent, Tool
    from pydantic_ai.models import Model
    from pydantic_ai.toolsets import AbstractToolset

logger = logging.getLogger(__name__)

ASK_TOOL_NAMES: list[str] = [
    "retrieve_medical_literature",
    "load_patient_facts",
    "load_lab_timeline",
    "cite_source",
    "ask_user_question",
]


def make_ask_agent(
    model: "Model",
    registry: PromptRegistry | None = None,
    language: str = "en",
    *,
    tools: "list[Union[Callable, Tool]] | None" = None,
    toolsets: "list[AbstractToolset[AskDeps]] | None" = None,
    output_type: Any = str,
    extra_prompt_names: list[str] | None = None,
) -> "Agent[AskDeps, Any]":
    """Build the Pydantic AI agent for ask mode.

    Args:
        tools: Mix of plain callables and pre-built ``pydantic_ai.Tool``
            instances to register. ``Tool`` instances are required when
            a tool needs a custom ``description`` (read from the prompt
            registry rather than the docstring) or non-default
            ``max_retries``. The caller (typically ``AskService`` after
            composing ``FeaturePlugin.as_tool`` results plus any
            interaction tools) decides which tools the LLM may invoke
            this turn. Pass ``None`` or ``[]`` for a no-tool agent
            (deterministic-only turns).
        toolsets: Pre-built ``AbstractToolset`` instances (e.g.
            ``ApprovalRequiredToolset`` wrapping the seven ingest write
            tools). Composed alongside ``tools`` so the agent sees the
            union without the caller having to flatten them into plain
            callables. ``None`` / ``[]`` means no toolset.
        output_type: Agent output spec. Defaults to ``str``. Pass
            ``str | DeferredToolRequests`` when the agent has at least
            one approval-gated toolset wired — the framework needs the
            union so it can return ``DeferredToolRequests`` as the
            run output when a tool call raises ``ApprovalRequired``.
        extra_prompt_names: Optional list of registry prompt names to
            append to the base ``ask`` system prompt. AskService passes
            ``["tool_proposal"]`` when the ingest-tools toolset is
            wired so the LLM sees the meta-rules for proposing PHI
            writes alongside the standard answer rules. A missing
            entry logs a warning and is skipped rather than aborting
            the turn.
    """
    from pydantic_ai import Agent

    reg = registry or PromptRegistry()
    parts: list[str] = [reg.get("ask", language=language)]  # type: ignore[arg-type]
    for extra in extra_prompt_names or []:
        try:
            parts.append(reg.get(extra, language=language))  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            logger.warning("extra system prompt %r missing; skipping", extra)
    system_prompt = "\n\n".join(parts)

    agent: "Agent[AskDeps, Any]" = Agent(
        model,
        output_type=output_type,
        system_prompt=system_prompt,
        deps_type=AskDeps,
        tools=list(tools or []),
        toolsets=list(toolsets or []),
    )
    return agent
