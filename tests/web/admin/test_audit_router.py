"""Tests for ``/api/v1/admin/audit``."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from claritymed import config as _cfg


def _write_audit_log(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def _make_event(
    kind: str,
    *,
    user_id: str = "test",
    request_id: str | None = None,
    created_at: datetime | None = None,
    payload: dict | None = None,
) -> dict:
    ts = created_at or datetime.now(timezone.utc)
    return {
        "kind": kind,
        "payload": payload or {},
        "request_id": request_id or ts.strftime("%Y%m%d%H%M%S") + "ABCDEF12",
        "user_id": user_id,
        "language": "en",
        "created_at": ts.isoformat(),
        "trace_id": None,
        "span_id": None,
    }


@pytest.fixture
def audit_log_path(monkeypatch, tmp_path):
    log_dir = tmp_path / "audit_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_cfg, "LOG_DIR", log_dir)
    return log_dir / "audit.log"


@pytest.mark.asyncio
async def test_list_audit_empty(web_client, admin_cookies, audit_log_path):
    response = await web_client.get("/api/v1/admin/audit", cookies=admin_cookies)
    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["total_count"] == 0
    assert response.headers["X-Total-Count"] == "0"


@pytest.mark.asyncio
async def test_list_audit_requires_admin(web_client, non_admin_cookies):
    response = await web_client.get("/api/v1/admin/audit", cookies=non_admin_cookies)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_audit_pagination(web_client, admin_cookies, audit_log_path):
    base = datetime.now(timezone.utc)
    events = [
        _make_event("admin.config.read", created_at=base - timedelta(minutes=i))
        for i in range(20)
    ]
    _write_audit_log(audit_log_path, events)

    page_one = await web_client.get(
        "/api/v1/admin/audit?offset=0&limit=10", cookies=admin_cookies
    )
    assert page_one.status_code == 200
    body = page_one.json()
    assert body["total_count"] == 20
    assert len(body["items"]) == 10

    page_two = await web_client.get(
        "/api/v1/admin/audit?offset=10&limit=10", cookies=admin_cookies
    )
    body_two = page_two.json()
    assert len(body_two["items"]) == 10
    assert body_two["items"][0]["request_id"] != body["items"][0]["request_id"]


@pytest.mark.asyncio
async def test_list_audit_filter_by_kind(web_client, admin_cookies, audit_log_path):
    events = [
        _make_event("admin.config.read"),
        _make_event("admin.config.write"),
        _make_event("admin.user.update"),
    ]
    _write_audit_log(audit_log_path, events)
    response = await web_client.get(
        "/api/v1/admin/audit?kind=admin.config.write", cookies=admin_cookies
    )
    body = response.json()
    assert body["total_count"] == 1
    assert body["items"][0]["kind"] == "admin.config.write"


@pytest.mark.asyncio
async def test_list_audit_filter_by_actor(web_client, admin_cookies, audit_log_path):
    events = [
        _make_event("admin.config.read", user_id="alice"),
        _make_event("admin.config.read", user_id="bob"),
    ]
    _write_audit_log(audit_log_path, events)
    response = await web_client.get(
        "/api/v1/admin/audit?actor=bob", cookies=admin_cookies
    )
    body = response.json()
    assert body["total_count"] == 1
    assert body["items"][0]["user_id"] == "bob"


@pytest.mark.asyncio
async def test_list_audit_filter_by_time_range(
    web_client, admin_cookies, audit_log_path
):
    now = datetime.now(timezone.utc)
    events = [
        _make_event("admin.config.read", created_at=now - timedelta(hours=2)),
        _make_event("admin.config.read", created_at=now),
    ]
    _write_audit_log(audit_log_path, events)
    cutoff = (now - timedelta(hours=1)).isoformat()
    response = await web_client.get(
        "/api/v1/admin/audit", params={"since": cutoff}, cookies=admin_cookies
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 1


@pytest.mark.asyncio
async def test_list_audit_malformed_since_returns_422(
    web_client, admin_cookies, audit_log_path
):
    response = await web_client.get(
        "/api/v1/admin/audit?since=not-a-date", cookies=admin_cookies
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_audit_offset_beyond_total(
    web_client, admin_cookies, audit_log_path
):
    _write_audit_log(audit_log_path, [_make_event("admin.config.read")])
    response = await web_client.get(
        "/api/v1/admin/audit?offset=100", cookies=admin_cookies
    )
    body = response.json()
    assert body["total_count"] == 1
    assert body["items"] == []
