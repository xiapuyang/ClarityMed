"""Tests for ``/api/v1/admin/servers``.

The probes target localhost:8082..8086 which aren't running under the
test harness — so we let httpx fail naturally with connect/timeout
errors and assert the router degrades cleanly to ``down``/``timeout``
states. This is more honest than patching httpx (which would mask the
exact error envelope shape the production code uses).
"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_servers_requires_admin(web_client, non_admin_cookies):
    response = await web_client.get("/api/v1/admin/servers", cookies=non_admin_cookies)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_servers_returns_valid_payload(web_client, admin_cookies):
    """Sanity: the endpoint always returns a payload with the expected
    shape regardless of whether subprocess servers happen to be running
    on the dev machine.
    """
    response = await web_client.get("/api/v1/admin/servers", cookies=admin_cookies)
    assert response.status_code == 200
    body = response.json()
    by_id = {n["id"]: n for n in body["nodes"]}
    assert by_id["embedder"]["status"] in ("up", "down", "timeout")
    assert by_id["reranker"]["status"] in ("up", "down", "timeout")
    assert by_id["rag"]["status"] == "n/a"
    assert {"from": "reranker", "to": "embedder"} in body["edges"]
    assert {"from": "chat", "to": "rag"} in body["edges"]


@pytest.mark.asyncio
async def test_servers_graph_includes_process_nodes(web_client, admin_cookies):
    response = await web_client.get("/api/v1/admin/servers", cookies=admin_cookies)
    body = response.json()
    process_ids = {n["id"] for n in body["nodes"] if n["kind"] == "process"}
    assert process_ids == {
        "embedder",
        "reranker",
        "symptoms",
        "vision",
        "medical_clip",
        "omlx",
    }
    logical_ids = {n["id"] for n in body["nodes"] if n["kind"] == "logical"}
    assert "rag" in logical_ids
    assert "chat" in logical_ids
