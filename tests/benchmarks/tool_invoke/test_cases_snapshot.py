"""Per-run cases snapshot — read / write / build round-trip + the
upload-path integration that the lookup CLI depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.benchmarks.tool_invoke.cases_snapshot import (
    SCHEMA_VERSION,
    SNAPSHOT_FILENAME,
    CaseSnapshotEntry,
    CasesSnapshot,
    build_snapshot,
    read_snapshot,
    write_snapshot,
)


# --- helpers --------------------------------------------------------


class _FakeCase:
    """Minimal stand-in for a real Case dataclass instance."""

    def __init__(
        self,
        *,
        name: str,
        revision: int = 1,
        tier: str = "base",
        expected_behavior: str = "call_tool",
        expected_tool: str | None = None,
        expected_tools: list[str] | None = None,
        prompts: dict[str, str] | None = None,
        seed: object | None = None,
    ) -> None:
        self.name = name
        self.revision = revision
        self.tier = tier
        self.expected_behavior = expected_behavior
        self.expected_tool = expected_tool
        self.expected_tools = expected_tools or []
        self.prompts = prompts or {"en": "EN", "zh": "ZH"}
        self.seed = seed
        # Real predicate so inspect.getsource has something to grab.
        self.args_predicate = _example_predicate


def _example_predicate(args: dict) -> tuple[bool, str]:
    """Sample predicate used to verify source capture survives round-trip."""
    return True, "always-pass marker"


# --- tests ----------------------------------------------------------


def test_build_snapshot_captures_predicate_source():
    """Snapshot entries carry args_predicate_src so a future reader
    sees the *grader logic*, not just the case name."""
    snap = build_snapshot(
        runner="ingest",
        bench_ts="20260617_120000_001",
        cases=[_FakeCase(name="save_allergy", revision=2)],
    )
    assert len(snap.cases) == 1
    entry = snap.cases[0]
    assert entry.case_id == "save_allergy@v2"
    assert entry.name == "save_allergy"
    assert entry.revision == 2
    # ``inspect.getsource`` should have captured the marker comment.
    assert "always-pass marker" in entry.args_predicate_src
    # ``content_sha256`` is deterministic — recomputing for the same
    # case produces the same hash.
    snap2 = build_snapshot(
        runner="ingest",
        bench_ts="x",  # bench_ts doesn't affect content hash
        cases=[_FakeCase(name="save_allergy", revision=2)],
    )
    assert snap2.cases[0].content_sha256 == entry.content_sha256


def test_snapshot_round_trip(tmp_path: Path):
    """write_snapshot → read_snapshot returns the same content."""
    original = build_snapshot(
        runner="ingest",
        bench_ts="20260617_120000_001",
        cases=[
            _FakeCase(name="save_allergy", revision=2),
            _FakeCase(name="save_medication", revision=1),
        ],
    )
    write_snapshot(tmp_path, original)
    loaded = read_snapshot(tmp_path)
    assert loaded is not None
    assert loaded.runner == "ingest"
    assert loaded.bench_ts == "20260617_120000_001"
    assert [e.case_id for e in loaded.cases] == [
        "save_allergy@v2",
        "save_medication@v1",
    ]


def test_read_snapshot_returns_none_when_absent(tmp_path: Path):
    """Legacy runs don't have a snapshot — silently return None so the
    uploader can fall back to trial-derived examples."""
    assert read_snapshot(tmp_path) is None


def test_read_snapshot_fails_on_unknown_schema(tmp_path: Path):
    """Future shape change must fail loud, not silently lose fields."""
    (tmp_path / SNAPSHOT_FILENAME).write_text(
        '{"schema_version": 999, "runner": "ingest", "bench_ts": "x", "cases": []}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema_version"):
        read_snapshot(tmp_path)


def test_snapshot_entry_includes_prompts_dict():
    """The full prompts dict (templates, before any seed substitution)
    must survive into the entry so Phoenix examples can render it."""
    snap = build_snapshot(
        runner="ingest",
        bench_ts="x",
        cases=[
            _FakeCase(
                name="weight_kg_update",
                prompts={"en": "set weight to {kg} kg", "zh": "设体重为 {kg} kg"},
            )
        ],
    )
    assert snap.cases[0].prompts == {
        "en": "set weight to {kg} kg",
        "zh": "设体重为 {kg} kg",
    }


def test_schema_version_is_int():
    """Sanity: the module-level constant the writer stamps in is an int."""
    assert isinstance(SCHEMA_VERSION, int)
    snap = CasesSnapshot(runner="x", bench_ts="y", cases=[])
    assert snap.schema_version == SCHEMA_VERSION


def test_snapshot_entry_is_frozen():
    """CaseSnapshotEntry is frozen so tests / consumers can't mutate
    an entry after it's been hashed into the snapshot file."""
    entry = CaseSnapshotEntry(
        case_id="x@v1",
        name="x",
        revision=1,
        content_sha256="a" * 64,
        tier="base",
        expected_behavior="call_tool",
        args_predicate_src="def f(): pass",
    )
    with pytest.raises((TypeError, ValueError)):
        entry.revision = 2  # type: ignore[misc]
