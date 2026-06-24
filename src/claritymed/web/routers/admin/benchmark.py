"""Admin benchmark endpoints.

Two read endpoints + one trigger:

* ``GET /admin/benchmark/runs`` — list of past runs by scanning
  ``SHARED_DIR/evals/``. A run dir without a recognizable result
  manifest is reported as ``incomplete`` rather than dropped.
* ``GET /admin/benchmark/runs/{run_id}`` — drilldown for one run.
* ``POST /admin/benchmark/runs`` — kicks a ``benchmark_run`` Job; the
  runner is wired in ``app.py`` lifespan.

We deliberately keep the runner CLI args open-ended; the SPA's form
serializes whatever shape the operator picked into the job's ``params``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from claritymed import config as _cfg
from claritymed.core.observability.audit import audit_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmark", tags=["admin", "benchmark"])

RESULT_MANIFEST_NAMES = ("result.json", "results.json", "summary.json")


class BenchmarkRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str | None = None
    provider_id: str | None = None
    sample_size: int | None = None
    runner: str = "lm_eval_runner"
    extra_args: list[str] | None = None


def _evals_root():
    return _cfg.SHARED_DIR / "evals"


def _read_manifest(run_dir):
    for name in RESULT_MANIFEST_NAMES:
        path = run_dir / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    return None


@router.get("/runs")
async def list_runs() -> dict[str, Any]:
    root = _evals_root()
    items: list[dict[str, Any]] = []
    if root.exists():
        for run_dir in sorted(root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            manifest = _read_manifest(run_dir)
            items.append(
                {
                    "run_id": run_dir.name,
                    "complete": manifest is not None,
                    "summary": manifest,
                }
            )
    audit_event("admin.benchmark.read", payload={"count": len(items)})
    return {"items": items, "total_count": len(items)}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    if "/" in run_id or run_id.startswith("."):
        raise HTTPException(status_code=400, detail="invalid run_id")
    run_dir = _evals_root() / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    manifest = _read_manifest(run_dir)
    files = sorted(p.name for p in run_dir.iterdir())
    audit_event("admin.benchmark.read", payload={"run_id": run_id})
    return {
        "run_id": run_id,
        "manifest": manifest,
        "files": files,
        "complete": manifest is not None,
    }


@router.post("/runs")
async def trigger_run(req: BenchmarkRunRequest, request: Request) -> dict[str, Any]:
    registry = request.app.state.jobs
    if not registry.has_runner("benchmark_run"):
        raise HTTPException(
            status_code=503, detail="benchmark_run runner not registered"
        )
    spec = await registry.submit(
        "benchmark_run", req.model_dump(mode="python", exclude_none=True)
    )
    return spec.to_json()
