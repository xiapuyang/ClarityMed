"""Ask mode: LLM-driven agent with on-demand medical literature retrieval.

The ``retrieve_medical_literature`` tool lets the LLM decide per-query
whether to search the knowledge base rather than being forced on every
turn.  Greetings, chitchat, and follow-up clarifications are handled
purely by the LLM; clinical questions trigger a retrieval round-trip.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from claritymed.core.prompts.registry import PromptRegistry
from claritymed.orchestrator.agents.ask_deps import AskDeps

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)

ASK_TOOL_NAMES: list[str] = [
    "retrieve_medical_literature",
    "load_patient_facts",
    "load_lab_timeline",
    "cite_source",
]


def make_ask_agent(
    model: "Model",
    registry: PromptRegistry | None = None,
    language: str = "en",
    *,
    tools: "list[Callable] | None" = None,
) -> "Agent[AskDeps, str]":
    """Build the Pydantic AI agent for ask mode.

    Args:
        tools: Pydantic-AI tool callables to register on the agent. The
            caller (typically ``AskService`` after composing
            ``FeaturePlugin.as_tool`` results) decides which tools the
            LLM may invoke this turn. Pass ``None`` or ``[]`` for a
            no-tool agent (deterministic-only turns).
    """
    from pydantic_ai import Agent

    reg = registry or PromptRegistry()
    system_prompt = reg.get("ask", language=language)  # type: ignore[arg-type]

    agent: "Agent[AskDeps, str]" = Agent(
        model,
        output_type=str,
        system_prompt=system_prompt,
        deps_type=AskDeps,
    )
    for tool in tools or []:
        agent.tool(tool)
    return agent
