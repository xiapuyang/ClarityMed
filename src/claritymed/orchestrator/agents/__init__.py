"""Agent factories per mode.

Only the ask mode owns a Pydantic AI ``Agent``. The rag mode runs
deterministic dispatch directly inside ``RagService`` — no LLM call,
no agent factory.
"""

from claritymed.orchestrator.agents.ask_agent import make_ask_agent
from claritymed.orchestrator.agents.ask_deps import AskDeps

__all__ = [
    "AskDeps",
    "make_ask_agent",
]
