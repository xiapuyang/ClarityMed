"""Tests for ``/api/v1/admin/servers``.

The probes target localhost:8082..8086 which aren't running under the
test harness — so we let httpx fail naturally with connect/timeout
errors and assert the router degrades cleanly to ``down``/``timeout``
states. This is more honest than patching httpx (which would mask the
exact error envelope shape the production code uses).
"""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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
        "claritymed_web",
        "claritymed_ui",
        "admin_ui",
    }
    logical_ids = {n["id"] for n in body["nodes"] if n["kind"] == "logical"}
    assert "rag" in logical_ids
    assert "chat" in logical_ids


# --- _pid_from_port unit tests -------------------------------------------


def test_pid_from_port_lsof_not_found():
    """FileNotFoundError from lsof (not installed) → None."""
    from claritymed.web.routers.admin.servers import _pid_from_port

    with patch("subprocess.run", side_effect=FileNotFoundError("lsof not found")):
        result = _pid_from_port(9999)
    assert result is None


def test_pid_from_port_timeout_expired():
    """subprocess.TimeoutExpired → None."""
    from claritymed.web.routers.admin.servers import _pid_from_port

    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["lsof"], timeout=0.5),
    ):
        result = _pid_from_port(9999)
    assert result is None


def test_pid_from_port_no_digit_in_output():
    """lsof returns output with no digit-only tokens → None."""
    from claritymed.web.routers.admin.servers import _pid_from_port

    mock_result = MagicMock()
    mock_result.stdout = b""
    with patch("subprocess.run", return_value=mock_result):
        result = _pid_from_port(9999)
    assert result is None


# --- probe_node unit tests -----------------------------------------------


@pytest.mark.asyncio
async def test_probe_node_no_port_returns_down():
    """resolved_port() → None means no health endpoint → down with detail."""
    from claritymed.web.admin.servers_graph import ServerNode
    from claritymed.web.routers.admin.servers import probe_node

    node = MagicMock(spec=ServerNode)
    node.id = "test-node"
    node.kind = "process"
    node.label = "Test"
    node.resolved_port.return_value = None

    client = MagicMock(spec=httpx.AsyncClient)
    result = await probe_node(client, node)
    assert result["status"] == "down"
    assert result["detail"] == "no port configured"


@pytest.mark.asyncio
async def test_probe_node_timeout_returns_timeout():
    """httpx.TimeoutException → status=timeout."""
    from claritymed.web.admin.servers_graph import ServerNode
    from claritymed.web.routers.admin.servers import probe_node

    node = MagicMock(spec=ServerNode)
    node.id = "test-node"
    node.kind = "process"
    node.label = "Test"
    node.host = "localhost"
    node.health_path = "/health"
    node.resolved_port.return_value = 19999

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.side_effect = httpx.TimeoutException("timed out")

    result = await probe_node(client, node)
    assert result["status"] == "timeout"
    assert result["port"] == 19999


@pytest.mark.asyncio
async def test_probe_node_connection_error_returns_down():
    """httpx.HTTPError (ConnectError) → status=down with error class name."""
    from claritymed.web.admin.servers_graph import ServerNode
    from claritymed.web.routers.admin.servers import probe_node

    node = MagicMock(spec=ServerNode)
    node.id = "test-node"
    node.kind = "process"
    node.label = "Test"
    node.host = "localhost"
    node.health_path = "/health"
    node.resolved_port.return_value = 19998

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.side_effect = httpx.ConnectError("connection refused")

    result = await probe_node(client, node)
    assert result["status"] == "down"
    assert "ConnectError" in result["detail"]
