"""Admin landing-page composition endpoint.

One ``GET /api/v1/admin/overview`` call returns everything the
landing dashboard needs: provider count, user count, recent jobs,
recent audit events, and a placeholder for the servers card (filled
by U10). The endpoint stitches the data in-process — no HTTP fan-out
back to ourselves, which would double the latency budget on first
paint.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
from fastapi import APIRouter, Request

from claritymed import config as _cfg
from claritymed.stores.account import AccountStore
from claritymed.stores.models import load_models
from claritymed.stores.paths import list_user_ids
from claritymed.web.admin.jobs import JobRegistry
from claritymed.web.admin.servers_graph import NODES_PROCESS
from claritymed.web.routers.admin.servers import probe_node

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/overview", tags=["admin", "overview"])

AUDIT_TAIL_DEFAULT = 10
JOBS_RECENT_LIMIT = 5


@router.get("")
async def overview(request: Request) -> dict[str, Any]:
    return {
        "providers": _providers_card(),
        "users": _users_card(),
        "recent_jobs": _recent_jobs_card(request.app.state.jobs),
        "audit_tail": _audit_tail(),
        "servers": await _servers_card(),
    }


def _providers_card() -> dict[str, Any]:
    catalog = load_models()
    return {
        "count": len(catalog.providers),
        "default": catalog.default_provider,
    }


def _users_card() -> dict[str, Any]:
    ids = list_user_ids()
    admin_count = 0
    for uid in ids:
        try:
            if AccountStore(uid).load().role == "admin":
                admin_count += 1
        except FileNotFoundError:
            continue
    return {
        "count": len(ids),
        "admin_count": admin_count,
    }


def _recent_jobs_card(registry: JobRegistry) -> dict[str, Any]:
    specs = registry.list(limit=JOBS_RECENT_LIMIT)
    return {
        "items": [spec.to_json() for spec in specs],
        "active": registry.has_running_or_queued(),
    }


def _audit_tail(n: int = AUDIT_TAIL_DEFAULT) -> dict[str, Any]:
    path = _cfg.LOG_DIR / "audit.log"
    if not path.exists():
        return {"items": []}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"items": []}
    items: list[dict[str, Any]] = []
    # Each line carries the stdlib logging preamble before the JSON
    # payload (see ``AUDIT_FMT`` in core/observability/logging.py);
    # slice from the first ``{`` before json.loads. Same trick the
    # admin audit viewer uses.
    for raw in reversed(lines):
        line = raw.strip()
        if not line:
            continue
        brace = line.find("{")
        if brace < 0:
            continue
        try:
            items.append(json.loads(line[brace:]))
        except json.JSONDecodeError:
            continue
        if len(items) >= n:
            break
    return {"items": items}


async def _servers_card() -> dict[str, Any]:
    """Probe each process node's ``/health`` and return a compact summary.

    Reuses the same ``probe_node`` helper as ``/admin/servers`` so both
    surfaces agree on what ``up`` means. We drop ``edges`` here — the
    overview card only renders the per-node status pills, not the
    topology graph.
    """
    async with httpx.AsyncClient() as client:
        nodes = await asyncio.gather(
            *(probe_node(client, node) for node in NODES_PROCESS)
        )
    return {"ready": True, "nodes": list(nodes)}
