"""Tests for ``/api/v1/admin/overview``."""

from __future__ import annotations

import json

import pytest

from claritymed import config as _cfg


@pytest.mark.asyncio
async def test_overview_accessible_to_any_user(web_client, non_admin_cookies):
    response = await web_client.get("/api/v1/admin/overview", cookies=non_admin_cookies)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_overview_empty_defaults(web_client, admin_cookies):
    response = await web_client.get("/api/v1/admin/overview", cookies=admin_cookies)
    assert response.status_code == 200
    body = response.json()
    assert "providers" in body
    assert "users" in body
    assert "recent_jobs" in body
    assert "audit_tail" in body
    assert "servers" in body
    assert body["recent_jobs"]["active"] is False
    # Servers card is now wired up — endpoint always probes. Subprocess
    # servers aren't running under the test harness, so nodes report
    # down/timeout, but ``ready`` reflects that the probe ran.
    assert body["servers"]["ready"] is True
    assert isinstance(body["servers"]["nodes"], list)
    assert len(body["servers"]["nodes"]) > 0


@pytest.mark.asyncio
async def test_overview_audit_tail(web_client, admin_cookies, tmp_path, monkeypatch):
    log_dir = tmp_path / "audit_logs"
    log_dir.mkdir()
    audit_path = log_dir / "audit.log"
    events = [
        {
            "kind": "admin.config.read",
            "payload": {},
            "request_id": f"reqid{i:016d}".upper(),
            "user_id": "test",
            "language": "en",
            "created_at": f"2026-06-24T00:00:{i:02d}Z",
            "trace_id": None,
            "span_id": None,
        }
        for i in range(15)
    ]
    audit_path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    monkeypatch.setattr(_cfg, "LOG_DIR", log_dir)
    response = await web_client.get("/api/v1/admin/overview", cookies=admin_cookies)
    body = response.json()
    # AUDIT_TAIL_DEFAULT = 10.
    assert len(body["audit_tail"]["items"]) == 10


@pytest.mark.asyncio
async def test_overview_user_admin_count(web_client, admin_cookies, non_admin_user):
    response = await web_client.get("/api/v1/admin/overview", cookies=admin_cookies)
    body = response.json()
    assert body["users"]["count"] >= 2
    assert body["users"]["admin_count"] >= 1


@pytest.mark.asyncio
async def test_overview_users_card_skips_missing_account(
    web_client, admin_cookies, monkeypatch
):
    """_users_card skips a user id whose account file is missing (FileNotFoundError)."""
    from claritymed.web.routers.admin import overview as overview_mod

    monkeypatch.setattr(
        overview_mod, "list_user_ids", lambda: ["test", "ghost-no-account"]
    )
    response = await web_client.get("/api/v1/admin/overview", cookies=admin_cookies)
    assert response.status_code == 200
    body = response.json()
    # ghost-no-account has no settings.yaml → skipped; only 'test' counted
    assert body["users"]["count"] == 2
    assert body["users"]["admin_count"] >= 1


@pytest.mark.asyncio
async def test_overview_audit_tail_oserror_returns_empty(
    web_client, admin_cookies, tmp_path, monkeypatch
):
    """If audit.log is unreadable (OSError), audit_tail returns empty items."""
    log_dir = tmp_path / "audit_logs"
    log_dir.mkdir()
    # Creating audit.log as a directory makes read_text raise IsADirectoryError.
    (log_dir / "audit.log").mkdir()
    monkeypatch.setattr(_cfg, "LOG_DIR", log_dir)
    response = await web_client.get("/api/v1/admin/overview", cookies=admin_cookies)
    assert response.status_code == 200
    body = response.json()
    assert body["audit_tail"]["items"] == []


@pytest.mark.asyncio
async def test_overview_audit_tail_skips_blank_and_no_brace_lines(
    web_client, admin_cookies, tmp_path, monkeypatch
):
    """Blank lines and lines without '{' are skipped in the audit tail."""
    log_dir = tmp_path / "audit_logs"
    log_dir.mkdir()
    event = {
        "kind": "mode.ask",
        "payload": {},
        "request_id": "20260624000000AABBCCDD",
        "user_id": "test",
        "language": "en",
        "created_at": "2026-06-24T00:00:00+00:00",
        "trace_id": None,
        "span_id": None,
    }
    (log_dir / "audit.log").write_text(
        f"\nNO BRACE LINE\n{json.dumps(event)}\n\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_cfg, "LOG_DIR", log_dir)
    response = await web_client.get("/api/v1/admin/overview", cookies=admin_cookies)
    body = response.json()
    assert body["audit_tail"]["items"] == [event]


@pytest.mark.asyncio
async def test_overview_audit_tail_skips_invalid_json(
    web_client, admin_cookies, tmp_path, monkeypatch
):
    """Lines with '{' but invalid JSON are silently skipped in audit tail."""
    log_dir = tmp_path / "audit_logs"
    log_dir.mkdir()
    event = {
        "kind": "mode.ask",
        "payload": {},
        "request_id": "20260624000000EEFF0011",
        "user_id": "test",
        "language": "en",
        "created_at": "2026-06-24T00:00:01+00:00",
        "trace_id": None,
        "span_id": None,
    }
    (log_dir / "audit.log").write_text(
        f"{{invalid json!\n{json.dumps(event)}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_cfg, "LOG_DIR", log_dir)
    response = await web_client.get("/api/v1/admin/overview", cookies=admin_cookies)
    body = response.json()
    assert body["audit_tail"]["items"] == [event]
