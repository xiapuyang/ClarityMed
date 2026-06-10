"""Ask mode: LLM-driven agent with on-demand medical literature retrieval.

The ``retrieve_medical_literature`` tool lets the LLM decide per-query
whether to search the knowledge base rather than being forced on every
turn.  Greetings, chitchat, and follow-up clarifications are handled
purely by the LLM; clinical questions trigger a retrieval round-trip.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Union

from claritymed.core.prompts.registry import PromptRegistry
from claritymed.orchestrator.agents.ask_deps import AskDeps

if TYPE_CHECKING:
    from pydantic_ai import Agent, Tool
    from pydantic_ai.models import Model

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
) -> "Agent[AskDeps, str]":
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
    """
    from pydantic_ai import Agent

    reg = registry or PromptRegistry()
    system_prompt = reg.get("ask", language=language)  # type: ignore[arg-type]

    agent: "Agent[AskDeps, str]" = Agent(
        model,
        output_type=str,
        system_prompt=system_prompt,
        deps_type=AskDeps,
        tools=list(tools or []),
    )
    return agent
