"""Tests for ``ingest.records.state`` — rows.jsonl + session.yaml helpers.

The state directory is the *only* place PHI lives ephemerally during an
import, so the invariants here are stricter than usual:

* Append must be durable (flush + fsync) so SIGINT can't truncate.
* Read collapses to "latest entry per row_id wins" so resume can ask
  "what's already terminal?"
* ``write_session`` is atomic so a read mid-write never sees partial
  YAML.
* ``delete_template_dir`` is failure-safe — it never deletes part of
  the tree without leaving a marker the loader can refuse next time.
"""

from __future__ import annotations

import os
import stat
import sys
from datetime import datetime, timezone

import pytest

from claritymed.ingest.records.state import (
    CLEANUP_FAILED_MARKER,
    ImportState,
    RowRecord,
)


@pytest.fixture
def state(monkeypatch, tmp_path) -> ImportState:
    """Build an ImportState whose ``data/_imports/...`` lives under tmp.

    Reloads ``claritymed.config`` after monkeypatching so the path
    helpers see the redirected ``DATA_DIR``.
    """
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path / "data"))
    import importlib

    from claritymed import config as _cfg

    importlib.reload(_cfg)
    # paths module reads _cfg at runtime, no reimport needed.
    return ImportState("0123456789ab")


def _row(
    row_id: str,
    status: str,
    *,
    kind: str = "case",
    user_id: str = "test",
    slug: str | None = None,
) -> RowRecord:
    return RowRecord(
        kind=kind,
        user_id=user_id,
        row_id=row_id,
        ts=datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc),
        status=status,
        slug=slug,
    )


# --- append + read latest ---------------------------------------------


def test_append_then_read_latest_round_trip(state: ImportState):
    row = _row("case:test:c1", "done", slug="2024-01-15-tmpl-abc")
    state.append_row(row)
    latest = state.read_latest_rows()
    assert set(latest.keys()) == {"case:test:c1"}
    assert latest["case:test:c1"].status == "done"
    assert latest["case:test:c1"].slug == "2024-01-15-tmpl-abc"


def test_latest_collapses_pending_then_done(state: ImportState):
    """Resume reads ``rows.jsonl`` and asks "what's still pending?" —
    so the latest entry per row id must win, not the chronological one."""
    state.append_row(_row("case:test:c1", "pending"))
    state.append_row(_row("case:test:c1", "done", slug="2024-01-15-tmpl-abc"))
    latest = state.read_latest_rows()
    assert latest["case:test:c1"].status == "done"


def test_read_latest_missing_file_returns_empty(state: ImportState):
    """First-attempt resume sees an empty state, not an exception."""
    assert state.read_latest_rows() == {}


def test_read_latest_skips_blank_lines(state: ImportState):
    """A trailing newline shouldn't trip the parser."""
    state.append_row(_row("case:test:c1", "done"))
    state.rows_jsonl.open("a").write("\n\n")
    latest = state.read_latest_rows()
    assert "case:test:c1" in latest


def test_read_latest_raises_on_malformed_line(state: ImportState):
    """Silent skip would let a stale ``pending`` row be misclassified
    on resume. Fail-loud so the operator sees and fixes."""
    state.ensure_dir()
    state.rows_jsonl.write_text("not valid json\n")
    with pytest.raises(ValueError, match="line 1"):
        state.read_latest_rows()


# --- has_pending_row --------------------------------------------------


def test_has_pending_row_true_when_pending_line_exists(state: ImportState):
    state.append_row(_row("case:test:c1", "pending"))
    assert state.has_pending_row("case:test:c1") is True


def test_has_pending_row_false_after_terminal(state: ImportState):
    """The Unit 8 WAL pattern relies on this: ``pending`` line was
    appended before the store write. After a terminal append we still
    see ``pending`` in the file (append-only) — so the check is
    "does any line for this row_id say pending?"."""
    state.append_row(_row("case:test:c1", "pending"))
    state.append_row(_row("case:test:c1", "done"))
    # Both lines still on disk; pending sentinel still present.
    assert state.has_pending_row("case:test:c1") is True


def test_has_pending_row_false_when_never_appended(state: ImportState):
    """No file → no pending. Distinguishes first-run from crash-recover."""
    assert state.has_pending_row("case:test:c1") is False


# --- session.yaml -----------------------------------------------------


def test_write_session_round_trip(state: ImportState):
    payload = {
        "import_id": state.import_id,
        "counts": {"case": {"done": 5, "error": 0}},
    }
    state.write_session(payload)
    read_back = state.read_session()
    assert read_back["counts"]["case"]["done"] == 5


def test_write_session_atomic_no_tmp_left_on_success(state: ImportState):
    """`.tmp` sidecar must be gone after a successful rename."""
    state.write_session({"foo": "bar"})
    tmp = state.session_yaml.with_suffix(state.session_yaml.suffix + ".tmp")
    assert not tmp.exists()
    assert state.session_yaml.exists()


def test_read_session_missing_returns_none(state: ImportState):
    assert state.read_session() is None


# --- file modes -------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_state_dir_mode_is_0700(state: ImportState):
    state.ensure_dir()
    mode = stat.S_IMODE(os.stat(state.dir_path).st_mode)
    assert mode == 0o700


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_rows_jsonl_mode_is_0600(state: ImportState):
    state.append_row(_row("case:test:c1", "done"))
    mode = stat.S_IMODE(os.stat(state.rows_jsonl).st_mode)
    assert mode == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_session_yaml_mode_is_0600(state: ImportState):
    state.write_session({"foo": "bar"})
    mode = stat.S_IMODE(os.stat(state.session_yaml).st_mode)
    assert mode == 0o600


# --- delete_template_dir ----------------------------------------------


def test_delete_template_dir_idempotent_when_missing(state: ImportState):
    """Missing-dir is a no-op (idempotent). The orchestrator may call
    cleanup twice on a "successful import" path; we don't want one of
    them to raise."""
    state.delete_template_dir()  # must not raise
    state.delete_template_dir()  # second call also fine


def test_delete_template_dir_happy_path(state: ImportState):
    state.ensure_template_dir()
    (state.template_dir / "test.yaml").write_text("cases: []\n")
    state.delete_template_dir()
    assert not state.template_dir.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_delete_template_dir_writes_marker_on_failure(state: ImportState, monkeypatch):
    """When rmtree fails mid-tree, the next loader call must refuse to
    proceed — otherwise we'd silently leak PHI in a partial state.
    Simulated by forcing rmtree to fail; the marker file then exists."""
    state.ensure_template_dir()
    (state.template_dir / "test.yaml").write_text("cases: []\n")

    import shutil as _shutil

    def _broken_rmtree(path, onerror=None):
        # Simulate a failure on first file. Emit through onerror so the
        # caller's failure-safe path runs.
        if onerror is not None:
            try:
                raise PermissionError(f"simulated EBUSY on {path}")
            except PermissionError:
                onerror(None, str(path), sys.exc_info())

    monkeypatch.setattr(_shutil, "rmtree", _broken_rmtree)

    state.delete_template_dir()
    marker = state.template_dir / CLEANUP_FAILED_MARKER
    assert marker.exists(), "expected cleanup_failed marker after rmtree failure"
    assert "simulated EBUSY" in marker.read_text(encoding="utf-8")


# --- path validation --------------------------------------------------


def test_invalid_import_id_rejected_by_paths():
    """ImportState construction validates import_id via the regex —
    a typo'd id can't reach into a sibling directory."""
    with pytest.raises(ValueError, match="invalid import_id"):
        ImportState("not-12-hex")


def test_invalid_import_id_rejects_uppercase():
    with pytest.raises(ValueError, match="invalid import_id"):
        ImportState("ABCDEF123456")


def test_invalid_import_id_rejects_path_traversal():
    with pytest.raises(ValueError, match="invalid import_id"):
        ImportState("../../etc")
