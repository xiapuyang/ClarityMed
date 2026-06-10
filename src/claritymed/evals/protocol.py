"""Benchmark protocol — the abstract shape every eval runner satisfies.

``LmEvalRunner`` (Phase 1) implements this protocol over lm-evaluation-
harness; a future RAG-only comparison runner (Phase 2) will too. Pinning
the shape early means the CLI dispatch in ``cli/eval.py`` can swap
runners without conditional branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from claritymed.core.schemas import ProviderConfig


@dataclass(frozen=True)
class RunResult:
    """Summary of one benchmark run, written to the CLI and audit log."""

    provider_id: str
    task_id: str
    n_questions: int
    accuracy: float
    output_path: Path
    started_at: datetime
    duration_s: float


class Benchmark(Protocol):
    """A benchmark runner. One method, one run."""

    def run(
        self,
        provider: "ProviderConfig",
        task_id: str,
        limit: int | None,
    ) -> RunResult:
        """Score ``provider`` on ``task_id``, writing JSONL + emitting audit."""
        ...
