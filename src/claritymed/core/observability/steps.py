"""Lightweight step-timing sink for core services.

Any ``core/`` service that makes an LLM call (translation, reranking,
grading…) wraps it in ``step(name, details)`` to record timing and
metadata.  Caller sets up a ``capture_steps()`` scope; collected steps
are converted to ``ToolStarted``/``ToolCompleted`` events by the
orchestrator boundary (AskService) which knows about those event types.

This keeps the dependency arrow clean:
  core/observability/steps  ←  any core service
  orchestrator/ask_service  →  core/observability/steps  (reads sink)
  orchestrator/ask_service  →  orchestrator/services/events (ToolStarted…)

Pattern inside a core service::

    from claritymed.core.observability.steps import step

    with step("translate.query", details="translate/en") as s:
        result = await agent.run(text)
        s.summary = "done"
    return result.output
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Generator

_sink: ContextVar[list["StepRecord"] | None] = ContextVar("_step_sink", default=None)


@dataclass
class StepRecord:
    name: str
    details: str = ""
    duration_ms: int = 0
    summary: str = ""
    failed: bool = False


class step:
    """Sync context manager — works with ``await`` inside the ``with`` block.

    Records timing + outcome into the nearest ``capture_steps()`` scope.
    No-op when no scope is active so core services are safe to call
    outside an ask request (tests, CLI one-shots, etc.).
    """

    def __init__(self, name: str, details: str = "") -> None:
        self.name = name
        self.details = details
        self.summary: str = ""
        self._t0: float = 0.0

    def __enter__(self) -> "step":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = int((time.perf_counter() - self._t0) * 1000)
        sink = _sink.get()
        if sink is not None:
            sink.append(
                StepRecord(
                    name=self.name,
                    details=self.details,
                    duration_ms=duration_ms,
                    summary=self.summary,
                    failed=exc_type is not None,
                )
            )


@contextmanager
def capture_steps() -> Generator[list[StepRecord], None, None]:
    """Collect all ``step`` records emitted inside this scope."""
    records: list[StepRecord] = []
    token = _sink.set(records)
    try:
        yield records
    finally:
        _sink.reset(token)
