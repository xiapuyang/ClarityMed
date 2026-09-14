"""Evaluation-suite catalog: declares WHICH benchmarks the runner will execute.

Mirrors ``configs/evals.yaml``. The shape is intentionally tiny — task
definitions themselves live as lm-evaluation-harness YAML files under
``src/claritymed/evals/tasks/<task_id>.yaml`` so adding a new benchmark is
a YAML-only change (the §9-layer-5 extensibility claim from
``docs/plans/2026-06-09-002-feat-evals-medqa-plan.md``).

Fields:

* ``tasks`` — default task id list for ``claritymed eval`` invocations.
* ``output_dir`` — relative-to-project-root directory for per-run JSONL.
* ``judge_provider_id`` — reserved for Phase 2 / deepeval-judge reuse.
* ``default_limit`` — question cap when CLI ``--limit`` is omitted.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvalsConfig(BaseModel):
    """Parsed ``configs/evals.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: list[str] = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    judge_provider_id: str | None = Field(default=None)
    default_limit: int | None = Field(default=None, ge=0)
