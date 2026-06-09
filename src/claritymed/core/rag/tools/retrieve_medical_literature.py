"""Tool: retrieve_medical_literature.

Thin pydantic-ai shim over the shared retrieval pipeline
(``core.rag.retrieval_pipeline``). Frames the LLM tool-call protocol
with ``ToolStarted`` / ``ToolCompleted`` events, then delegates
retrieval + filtering + audit to the shared helper so
``RagFeature.pre_invoke`` (deterministic mode) and this tool run
identical pipelines.

Returned by ``RagFeature.as_tool`` when ``rag.mode=tool``; registered
on the agent by ``AskService`` via ``make_ask_agent(tools=[...])``.
"""

from __future__ import annotations

import logging

from pydantic_ai import RunContext

from claritymed.core.rag.retrieval_pipeline import format_evidence, perform_retrieval

# Runtime import (not TYPE_CHECKING) so pydantic-ai can resolve the
# ``RunContext[TurnState]`` annotation when this tool is registered on
# an Agent. ``TurnState`` is a Protocol with no orchestrator deps, so
# importing it at module scope is safe — no cycle, no static layer
# violation.
from claritymed.core.turn_state import TurnState

logger = logging.getLogger(__name__)


async def retrieve_medical_literature(ctx: RunContext[TurnState], query: str) -> str:
    """Search the medical knowledge base for evidence relevant to the query.

    Call this for medical questions requiring clinical evidence, drug
    information, differential diagnosis, treatment protocols, or
    disease-specific guidance.  Skip for greetings, chitchat, and
    clearly non-medical topics.
    """
    deps = ctx.deps
    from claritymed.core.events import ToolCompleted, ToolStarted

    # Count the invocation as early as possible so the
    # ``tool_announced_but_skipped`` detector at turn end has a true
    # signal even when the call fails halfway through.
    deps.tool_calls["retrieve_medical_literature"] = (
        deps.tool_calls.get("retrieve_medical_literature", 0) + 1
    )

    eq = deps.event_queue
    # Always emit ToolStarted so the Steps panel reflects the LLM's call
    # even when RAG is disabled — makes "RAG not configured" visible vs
    # the tool silently not being called at all.
    await eq.put(
        ToolStarted(tool_name="retrieve_medical_literature", args_preview=query[:60])
    )

    if deps.strategy is None:
        await eq.put(
            ToolCompleted(
                tool_name="retrieve_medical_literature", summary="RAG disabled"
            )
        )
        return ""

    safe_chunks = await perform_retrieval(deps, query)
    # Accumulate per-call chunks onto deps so AskService can render the
    # cumulative Sources block. The tool may be invoked 1..N times per
    # turn; ``deps.retrieved_chunks`` is the union across calls, while
    # the per-call ``safe_chunks`` is what this round's LLM sees.
    deps.retrieved_chunks.extend(safe_chunks)
    return format_evidence(safe_chunks)
