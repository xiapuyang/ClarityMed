"""Agent factories per mode.

Each mode owns one Pydantic AI ``Agent``. The agent is only ever
instantiated lazily — when the service layer decides to invoke the LLM —
because ingest and rag modes default to ``allow_llm_inference=false`` and
short-circuit to deterministic tool dispatch.
"""

from claritymed.orchestrator.agents.ask_agent import make_ask_agent
from claritymed.orchestrator.agents.ask_deps import AskDeps
from claritymed.orchestrator.agents.ingest_agent import (
    INGEST_TOOL_NAMES,
    make_ingest_agent,
    save_to_lab_record,
    save_to_profile,
    save_to_vision_record,
)
from claritymed.orchestrator.agents.rag_agent import (
    RAG_TOOL_NAMES,
    embed_and_store,
    make_rag_agent,
)

__all__ = [
    "AskDeps",
    "INGEST_TOOL_NAMES",
    "RAG_TOOL_NAMES",
    "embed_and_store",
    "make_ask_agent",
    "make_ingest_agent",
    "make_rag_agent",
    "save_to_lab_record",
    "save_to_profile",
    "save_to_vision_record",
]
