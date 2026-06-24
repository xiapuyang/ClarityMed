"""Admin servers health + dependency graph endpoint.

``GET /api/v1/admin/servers`` probes every process node's ``/health``
endpoint in parallel (httpx.AsyncClient, 500ms timeout each) and
assembles a graph payload the SPA renders via Mermaid. Logical nodes
have no port — their status is always ``n/a``; the SPA computes a
derived display state from their dependencies if it wants to.

Auto-refresh is the SPA's job (10s polling, configured in the hook).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from fastapi import APIRouter

from claritymed.core.observability.audit import audit_event
from claritymed.web.admin.servers_graph import (
    NODES_LOGICAL,
    NODES_PROCESS,
    ServerNode,
    edges,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/servers", tags=["admin", "servers"])

HEALTHCHECK_TIMEOUT_S = 0.5


async def _probe(client: httpx.AsyncClient, node: ServerNode) -> dict[str, Any]:
    port = node.resolved_port()
    if port is None:
        return {
            "id": node.id,
            "kind": node.kind,
            "label": node.label,
            "status": "down",
            "detail": "no port configured",
        }
    url = f"http://{node.host}:{port}{node.health_path}"
    try:
        response = await client.get(url, timeout=HEALTHCHECK_TIMEOUT_S)
    except httpx.TimeoutException:
        return {
            "id": node.id,
            "kind": node.kind,
            "label": node.label,
            "status": "timeout",
            "port": port,
        }
    except httpx.HTTPError as exc:
        return {
            "id": node.id,
            "kind": node.kind,
            "label": node.label,
            "status": "down",
            "port": port,
            "detail": str(exc.__class__.__name__),
        }
    payload: dict[str, Any] = {}
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        payload = {}
    return {
        "id": node.id,
        "kind": node.kind,
        "label": node.label,
        "status": "up" if response.status_code == 200 else "down",
        "port": port,
        "uptime_s": payload.get("uptime_s"),
        "pid": payload.get("pid"),
        "manifest_sha": payload.get("manifest_sha"),
    }


@router.get("")
async def get_servers() -> dict[str, Any]:
    """Return the full server graph plus per-node status."""
    nodes_payload: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        probes = await asyncio.gather(*(_probe(client, node) for node in NODES_PROCESS))
    nodes_payload.extend(probes)
    for node in NODES_LOGICAL:
        nodes_payload.append(
            {
                "id": node.id,
                "kind": node.kind,
                "label": node.label,
                "status": "n/a",
            }
        )
    audit_event(
        "admin.servers.healthcheck",
        payload={"count": len(NODES_PROCESS)},
    )
    return {
        "nodes": nodes_payload,
        "edges": [{"from": src, "to": dst} for src, dst in edges()],
    }
