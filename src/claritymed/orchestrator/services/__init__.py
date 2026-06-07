"""Service layer between the UI (TUI / Typer) and the agents.

Every entry point — Typer CLI subcommands, Textual TUI, future AG-UI web —
calls the appropriate service, never the agent directly. The service owns
ContextVar wiring (``request_id``, ``user_id``, ``language``), the R18 PHI
scrubbing step before any LLM call, audit log writes, and the
``AsyncIterator[Event]`` protocol that streams progress to the caller.
"""

from claritymed.orchestrator.services.ask_service import AskService
from claritymed.orchestrator.services.events import (
    Cancelled,
    Done,
    Error,
    Event,
    ModeRouted,
    RetrievalFiltered,
    TokenChunk,
    ToolCompleted,
    ToolStarted,
)
from claritymed.orchestrator.services.ingest_service import IngestService
from claritymed.orchestrator.services.rag_service import RagService

__all__ = [
    "AskService",
    "Cancelled",
    "Done",
    "Error",
    "Event",
    "IngestService",
    "ModeRouted",
    "RagService",
    "RetrievalFiltered",
    "TokenChunk",
    "ToolCompleted",
    "ToolStarted",
]
