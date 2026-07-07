"""Service layer between the UI (TUI / Typer) and the agents.

Every entry point — Typer CLI subcommands, Textual TUI, future AG-UI web —
calls the appropriate service, never the agent directly. The service owns
ContextVar wiring (``request_id``, ``user_id``, ``language``), the R18 PHI
scrubbing step before any LLM call, audit log writes, and the
``AsyncIterator[Event]`` protocol that streams progress to the caller.
"""

from claritymed.core.observability.latency import LatencyTrace, build_step_records
from claritymed.orchestrator.services.ask_service import AskService
from claritymed.orchestrator.services.chat_session import (
    ChatSession,
    ChatTurn,
    SessionMeta,
)
from claritymed.orchestrator.services.factory import (
    build_ask_service,
    build_rag_strategy,
)
from claritymed.core.events import (
    Cancelled,
    DifferentialReady,
    Done,
    Error,
    Event,
    LlmCallStarted,
    LlmFirstToken,
    RetrievalCompleted,
    RetrievalFiltered,
    RetrievalPending,
    RetrievalStarted,
    TokenChunk,
    ToolCompleted,
    ToolStarted,
)
from claritymed.orchestrator.services.rag_service import RagService

__all__ = [
    "AskService",
    "Cancelled",
    "ChatSession",
    "ChatTurn",
    "DifferentialReady",
    "Done",
    "Error",
    "Event",
    "LatencyTrace",
    "RagService",
    "LlmCallStarted",
    "LlmFirstToken",
    "RetrievalCompleted",
    "RetrievalFiltered",
    "RetrievalPending",
    "RetrievalStarted",
    "SessionMeta",
    "TokenChunk",
    "ToolCompleted",
    "ToolStarted",
    "build_ask_service",
    "build_rag_strategy",
    "build_step_records",
]
