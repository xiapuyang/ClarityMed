"""Ask mode: the only path that streams LLM-generated text to the user.

Phase 1 keeps the Agent shape but does not register retrieval tools — the
service layer is responsible for assembling the retrieved evidence and
passing it as user context. The real retrieval tools land with the text_rag
plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from claritymed.core.prompts.registry import PromptRegistry

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model

ASK_TOOL_NAMES: list[str] = [
    "retrieve_system_rag_stub",
    "retrieve_user_rag_stub",
    "load_patient_facts",
    "load_lab_timeline",
    "cite_source",
]


def make_ask_agent(
    model: "Model",
    registry: PromptRegistry | None = None,
    language: str = "en",
) -> "Agent[None, str]":
    """Build the Pydantic AI agent for ask mode.

    Phase 1 returns a text-streaming Agent (``output_type=str``). The
    service layer wraps the streamed text into a ``GroundedAnswer`` after
    the run finishes — uncertainty fusion, citation linking, and red-flag
    detection are post-processing steps that the agent does not own.
    """
    from pydantic_ai import Agent

    reg = registry or PromptRegistry()
    system_prompt = reg.get("ask", language=language)  # type: ignore[arg-type]

    agent = Agent(
        model,
        output_type=str,
        system_prompt=system_prompt,
    )
    return agent
