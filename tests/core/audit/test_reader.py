"""Smoke tests for ``read_audit_events`` JSONL parsing + filters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claritymed.core.audit import read_audit_events


def _write_audit_lines(log_dir: Path, name: str, events: list[dict]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / name
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, ensure_ascii=False))
            fh.write("\n")


def test_reads_jsonl_and_skips_blank_lines(tmp_path):
    _write_audit_lines(
        tmp_path,
        "audit.log",
        [
            {
                "kind": "mode.ask",
                "user_id": "alice",
                "created_at": "2026-06-09T10:00:00Z",
            },
            {
                "kind": "mode.ask",
                "user_id": "bob",
                "created_at": "2026-06-09T11:00:00Z",
            },
        ],
    )
    events = list(read_audit_events(log_dir=tmp_path))
    assert len(events) == 2
    assert events[0]["user_id"] == "alice"


def test_filters_by_user_id(tmp_path):
    _write_audit_lines(
        tmp_path,
        "audit.log",
        [
            {
                "kind": "mode.ask",
                "user_id": "alice",
                "created_at": "2026-06-09T10:00:00Z",
            },
            {
                "kind": "mode.ask",
                "user_id": "bob",
                "created_at": "2026-06-09T10:00:00Z",
            },
        ],
    )
    events = list(read_audit_events(log_dir=tmp_path, user_id="alice"))
    assert len(events) == 1
    assert events[0]["user_id"] == "alice"


def test_filters_by_time_window(tmp_path):
    _write_audit_lines(
        tmp_path,
        "audit.log",
        [
            {"kind": "mode.ask", "user_id": "x", "created_at": "2026-06-05T10:00:00Z"},
            {"kind": "mode.ask", "user_id": "x", "created_at": "2026-06-09T10:00:00Z"},
            {"kind": "mode.ask", "user_id": "x", "created_at": "2026-06-15T10:00:00Z"},
        ],
    )
    events = list(
        read_audit_events(log_dir=tmp_path, since="2026-06-08", until="2026-06-10")
    )
    assert len(events) == 1


def test_raises_when_log_dir_missing(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        list(read_audit_events(log_dir=missing))


def test_skips_malformed_lines(tmp_path):
    log = tmp_path / "audit.log"
    log.write_text(
        '{"kind":"mode.ask","user_id":"a","created_at":"2026-06-09T10:00:00Z"}\n'
        "not-json\n"
        "\n"
        '{"kind":"mode.ask","user_id":"b","created_at":"2026-06-09T10:00:00Z"}\n',
        encoding="utf-8",
    )
    events = list(read_audit_events(log_dir=tmp_path))
    assert len(events) == 2


def test_walks_rotated_files_in_mtime_order(tmp_path):
    """``audit.log.1`` written first, then ``audit.log`` — yields in
    modification-time order (older first)."""
    import time

    (tmp_path / "audit.log.1").write_text(
        '{"kind":"mode.ask","user_id":"old","created_at":"2026-06-01T00:00:00Z"}\n',
        encoding="utf-8",
    )
    time.sleep(0.02)
    (tmp_path / "audit.log").write_text(
        '{"kind":"mode.ask","user_id":"new","created_at":"2026-06-09T00:00:00Z"}\n',
        encoding="utf-8",
    )
    events = list(read_audit_events(log_dir=tmp_path))
    assert [e["user_id"] for e in events] == ["old", "new"]
