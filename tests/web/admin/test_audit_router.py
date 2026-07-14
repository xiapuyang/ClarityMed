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
async def test_list_audit_accessible_to_any_user(web_client, non_admin_cookies):
    response = await web_client.get("/api/v1/admin/audit", cookies=non_admin_cookies)
    assert response.status_code == 200


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


@pytest.mark.asyncio
async def test_list_audit_filter_by_request_id(
    web_client, admin_cookies, audit_log_path
):
    target_rid = "20260607213555TARGETRID"
    events = [
        _make_event("mode.ask", request_id="20260607213555OTHERRID"),
        _make_event("mode.ask", request_id=target_rid),
        _make_event("request_end", request_id=target_rid),
    ]
    _write_audit_log(audit_log_path, events)
    response = await web_client.get(
        f"/api/v1/admin/audit?request_id={target_rid}", cookies=admin_cookies
    )
    body = response.json()
    assert body["total_count"] == 2
    assert all(ev["request_id"] == target_rid for ev in body["items"])


@pytest.mark.asyncio
async def test_list_audit_returns_distinct_actors(
    web_client, admin_cookies, audit_log_path
):
    events = [
        _make_event("mode.ask", user_id="alice"),
        _make_event("mode.ask", user_id="bob"),
        _make_event("mode.ask", user_id="alice"),
    ]
    _write_audit_log(audit_log_path, events)
    response = await web_client.get("/api/v1/admin/audit", cookies=admin_cookies)
    body = response.json()
    # Distinct + sorted; covers the whole log, not just the current page.
    assert body["distinct_actors"] == ["alice", "bob"]


@pytest.mark.asyncio
async def test_list_audit_parses_logging_prefixed_lines(
    web_client, admin_cookies, audit_log_path
):
    """The on-disk audit.log carries the stdlib logging preamble
    (see ``AUDIT_FMT`` in core/observability/logging.py); the parser
    must strip it before json.loads. Regression test for the
    silently-empty admin viewer.
    """
    event = _make_event("admin.config.read", user_id="alice")
    prefixed = (
        "2026-06-07 17:35:55,386 "
        f"[{event['request_id']}][{event['user_id']}] [en] " + json.dumps(event)
    )
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    audit_log_path.write_text(prefixed + "\n", encoding="utf-8")
    response = await web_client.get("/api/v1/admin/audit", cookies=admin_cookies)
    body = response.json()
    assert body["total_count"] == 1
    assert body["items"][0]["kind"] == "admin.config.read"
    assert body["items"][0]["user_id"] == "alice"


@pytest.mark.asyncio
async def test_load_events_skips_blank_lines(web_client, admin_cookies, audit_log_path):
    """Blank lines in audit.log are skipped without error."""
    good = json.dumps(_make_event("mode.ask"))
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    audit_log_path.write_text(f"\n{good}\n\n", encoding="utf-8")
    response = await web_client.get("/api/v1/admin/audit", cookies=admin_cookies)
    body = response.json()
    assert body["total_count"] == 1


@pytest.mark.asyncio
async def test_load_events_skips_no_brace_lines(
    web_client, admin_cookies, audit_log_path
):
    """Lines with no '{' character (e.g. rotation markers) are silently skipped."""
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    good = json.dumps(_make_event("mode.ask"))
    audit_log_path.write_text(f"LOG ROTATION MARKER\n{good}\n", encoding="utf-8")
    response = await web_client.get("/api/v1/admin/audit", cookies=admin_cookies)
    body = response.json()
    assert body["total_count"] == 1


@pytest.mark.asyncio
async def test_load_events_skips_invalid_json(
    web_client, admin_cookies, audit_log_path
):
    """Lines with '{' but unparseable JSON are silently skipped."""
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    good = json.dumps(_make_event("mode.ask"))
    audit_log_path.write_text(f"{{invalid json here\n{good}\n", encoding="utf-8")
    response = await web_client.get("/api/v1/admin/audit", cookies=admin_cookies)
    body = response.json()
    assert body["total_count"] == 1


@pytest.mark.asyncio
async def test_match_skips_event_without_created_at_on_time_filter(
    web_client, admin_cookies, audit_log_path
):
    """Events missing created_at are excluded when a time filter is active."""
    event = _make_event("mode.ask")
    del event["created_at"]
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    audit_log_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    response = await web_client.get(
        "/api/v1/admin/audit",
        params={"since": "2020-01-01T00:00:00+00:00"},
        cookies=admin_cookies,
    )
    body = response.json()
    assert body["total_count"] == 0


@pytest.mark.asyncio
async def test_match_skips_event_with_invalid_created_at(
    web_client, admin_cookies, audit_log_path
):
    """Events with a non-ISO created_at are excluded under a time filter."""
    event = _make_event("mode.ask")
    event["created_at"] = "not-a-date"
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    audit_log_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    response = await web_client.get(
        "/api/v1/admin/audit",
        params={"since": "2020-01-01T00:00:00+00:00"},
        cookies=admin_cookies,
    )
    body = response.json()
    assert body["total_count"] == 0


@pytest.mark.asyncio
async def test_match_filters_by_until_excludes_later_events(
    web_client, admin_cookies, audit_log_path
):
    """Events after 'until' are excluded; events before it are kept."""
    now = datetime.now(timezone.utc)
    events = [
        _make_event("mode.ask", created_at=now - timedelta(hours=2)),  # before until
        _make_event("mode.ask", created_at=now),  # after until
    ]
    _write_audit_log(audit_log_path, events)
    until = (now - timedelta(hours=1)).isoformat()
    response = await web_client.get(
        "/api/v1/admin/audit",
        params={"until": until},
        cookies=admin_cookies,
    )
    body = response.json()
    assert body["total_count"] == 1
