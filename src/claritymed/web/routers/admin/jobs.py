"""Admin jobs router — list, get, cancel.

POST endpoints to *create* jobs live with their respective kinds
(``/admin/rag/collections/upsert``, ``/admin/benchmark/runs``) so
authentication, validation, and request bodies stay close to the
domain. This router is read + cancel only.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/jobs", tags=["admin", "jobs"])


def _spec_to_response(spec: Any) -> dict[str, Any]:
    return spec.to_json()


@router.get("")
async def list_jobs(
    request: Request,
    state: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """Return jobs sorted by newest first.

    Filter by ``state`` (queued/running/done/failed/cancelled/crashed)
    or ``kind`` (rag_ingest/benchmark_run). Both are
    optional; the SPA polls with ``state=running`` while any are visible.
    """
    registry = request.app.state.jobs
    items = registry.list(state=state, kind=kind, limit=limit)
    return {
        "items": [_spec_to_response(s) for s in items],
        "total": len(items),
    }


@router.get("/{job_id}")
async def get_job(job_id: str, request: Request) -> dict[str, Any]:
    registry = request.app.state.jobs
    spec = registry.get(job_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    return _spec_to_response(spec)


@router.delete("/{job_id}")
async def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    registry = request.app.state.jobs
    spec = registry.get(job_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"job {job_id!r} not found")
    if not spec.cancellable:
        raise HTTPException(status_code=400, detail="job is not cancellable")
    updated = await registry.cancel(job_id)
    return _spec_to_response(updated or spec)
