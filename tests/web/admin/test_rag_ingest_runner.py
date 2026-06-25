"""Unit tests for the rag_ingest job runner.

Covers _request_from_params, _cleanup_uploads, and the run() coroutine.
ingest_system_rag is always patched so no real qdrant/embedder is needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from claritymed.ingest.corpus.base import IngestStats
from claritymed.ingest.system_rag import SystemRagIngestResult
from claritymed.web.admin.job_runners.rag_ingest import (
    _cleanup_uploads,
    _request_from_params,
    run,
)


# --- _request_from_params -----------------------------------------------


def test_request_from_params_happy_path(tmp_path):
    p = tmp_path / "doc.pdf"
    p.touch()
    req = _request_from_params(
        {
            "name": "my_collection",
            "file_paths": [str(p)],
            "topics": ["oncology"],
            "language": "en",
            "cross_lingual": True,
            "authority_tier": 1,
            "license": "CC-BY",
            "dedupe_cosine_threshold": 0.9,
        }
    )
    assert req.name == "my_collection"
    assert req.files == [p]
    assert req.topics == ["oncology"]
    assert req.language == "en"
    assert req.cross_lingual is True
    assert req.authority_tier == 1
    assert req.license == "CC-BY"
    assert req.dedupe_cosine_threshold == pytest.approx(0.9)


def test_request_from_params_missing_name():
    with pytest.raises(ValueError, match="name"):
        _request_from_params({"file_paths": ["/some/path.pdf"]})


def test_request_from_params_missing_files():
    with pytest.raises(ValueError, match="file_paths"):
        _request_from_params({"name": "test_col", "file_paths": []})


def test_request_from_params_defaults():
    """Fields omitted from params get historical defaults."""
    req = _request_from_params(
        {"name": "default_col", "file_paths": ["/some/file.pdf"]}
    )
    assert req.topics == []
    assert req.language is None
    assert req.cross_lingual is False
    assert req.authority_tier is None
    assert req.dedupe_cosine_threshold == pytest.approx(0.0)


# --- _cleanup_uploads ---------------------------------------------------


def test_cleanup_uploads_removes_existing_dir(tmp_path):
    d = tmp_path / "uploads" / "abc123"
    d.mkdir(parents=True)
    (d / "file.txt").write_text("data")
    _cleanup_uploads(str(d))
    assert not d.exists()


def test_cleanup_uploads_nonexistent_dir_is_noop(tmp_path):
    _cleanup_uploads(str(tmp_path / "does_not_exist"))


def test_cleanup_uploads_none_is_noop():
    _cleanup_uploads(None)


# --- helpers for run() tests --------------------------------------------


class _FakeSpec:
    id = "fake-job-id"

    def __init__(self, params):
        self.params = params


class _FakeRegistry:
    def __init__(self):
        self.stdout_lines: list[str] = []
        self.updates: list[dict] = []

    def update(self, job_id, *, progress):  # noqa: ARG002
        self.updates.append({"progress": progress})

    def append_stdout(self, job_id, line):  # noqa: ARG002
        self.stdout_lines.append(line)


def _make_result(
    *,
    children_written=8,
    children_deduped=0,
    centroid_refreshed=False,
    is_new_collection=False,
    yaml_snippet="name: test_col\n",
) -> SystemRagIngestResult:
    stats = IngestStats(
        source="test",
        docs_processed=2,
        parents_written=4,
        children_written=children_written,
        docs_skipped=0,
        docs_resumed=0,
        children_deduped=children_deduped,
    )
    return SystemRagIngestResult(
        stats=stats,
        yaml_snippet=yaml_snippet,
        is_new_collection=is_new_collection,
        centroid_refreshed=centroid_refreshed,
    )


# --- run() tests --------------------------------------------------------


async def test_run_happy_path(tmp_path):
    """Existing collection: no yaml append, stdout has summary, upload dir cleaned."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "doc.pdf").write_text("content")

    spec = _FakeSpec(
        {
            "name": "existing_col",
            "file_paths": [str(upload_dir / "doc.pdf")],
            "upload_dir": str(upload_dir),
        }
    )
    registry = _FakeRegistry()
    result = _make_result(children_written=5, is_new_collection=False)

    with patch(
        "claritymed.web.admin.job_runners.rag_ingest.ingest_system_rag",
        new=AsyncMock(return_value=result),
    ):
        await run(spec, registry)

    assert not upload_dir.exists(), "upload_dir should be cleaned up"
    assert any("summary" in line for line in registry.stdout_lines)
    assert any("done" in u["progress"] for u in registry.updates)


async def test_run_new_collection_appends_yaml(tmp_path):
    """New collection with children: yaml snippet is auto-appended."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "doc.pdf").write_text("content")

    spec = _FakeSpec(
        {
            "name": "new_col",
            "file_paths": [str(upload_dir / "doc.pdf")],
            "upload_dir": str(upload_dir),
        }
    )
    registry = _FakeRegistry()
    result = _make_result(children_written=3, is_new_collection=True)

    with (
        patch(
            "claritymed.web.admin.job_runners.rag_ingest.ingest_system_rag",
            new=AsyncMock(return_value=result),
        ),
        patch(
            "claritymed.web.admin.job_runners.rag_ingest.append_system_rag_collection",
            return_value=True,
        ) as mock_append,
    ):
        await run(spec, registry)

    mock_append.assert_called_once()
    assert any("appended new collection" in line for line in registry.stdout_lines)


async def test_run_new_collection_already_in_yaml(tmp_path):
    """New collection but append_system_rag_collection returns False (already present)."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "doc.pdf").write_text("content")

    spec = _FakeSpec(
        {
            "name": "dup_col",
            "file_paths": [str(upload_dir / "doc.pdf")],
            "upload_dir": str(upload_dir),
        }
    )
    registry = _FakeRegistry()
    result = _make_result(children_written=3, is_new_collection=True)

    with (
        patch(
            "claritymed.web.admin.job_runners.rag_ingest.ingest_system_rag",
            new=AsyncMock(return_value=result),
        ),
        patch(
            "claritymed.web.admin.job_runners.rag_ingest.append_system_rag_collection",
            return_value=False,
        ),
    ):
        await run(spec, registry)

    assert any("already present" in line for line in registry.stdout_lines)


async def test_run_new_collection_no_children_skips_yaml(tmp_path):
    """New collection but zero children written: yaml append is skipped."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "doc.pdf").write_text("content")

    spec = _FakeSpec(
        {
            "name": "empty_col",
            "file_paths": [str(upload_dir / "doc.pdf")],
            "upload_dir": str(upload_dir),
        }
    )
    registry = _FakeRegistry()
    result = _make_result(children_written=0, is_new_collection=True)

    with (
        patch(
            "claritymed.web.admin.job_runners.rag_ingest.ingest_system_rag",
            new=AsyncMock(return_value=result),
        ),
        patch(
            "claritymed.web.admin.job_runners.rag_ingest.append_system_rag_collection",
        ) as mock_append,
    ):
        await run(spec, registry)

    mock_append.assert_not_called()
    assert any("skipping yaml append" in line for line in registry.stdout_lines)


async def test_run_centroid_refreshed_logged(tmp_path):
    """centroid_refreshed=True emits a dedicated stdout line."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "doc.pdf").write_text("content")

    spec = _FakeSpec(
        {
            "name": "some_col",
            "file_paths": [str(upload_dir / "doc.pdf")],
            "upload_dir": str(upload_dir),
        }
    )
    registry = _FakeRegistry()
    result = _make_result(centroid_refreshed=True)

    with patch(
        "claritymed.web.admin.job_runners.rag_ingest.ingest_system_rag",
        new=AsyncMock(return_value=result),
    ):
        await run(spec, registry)

    assert any("centroid refreshed" in line for line in registry.stdout_lines)


async def test_run_cleanup_on_ingest_error(tmp_path):
    """upload_dir is removed even when ingest_system_rag raises."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "file.txt").write_text("data")

    spec = _FakeSpec(
        {
            "name": "boom_col",
            "file_paths": [str(upload_dir / "file.txt")],
            "upload_dir": str(upload_dir),
        }
    )
    registry = _FakeRegistry()

    with patch(
        "claritymed.web.admin.job_runners.rag_ingest.ingest_system_rag",
        new=AsyncMock(side_effect=RuntimeError("qdrant down")),
    ):
        with pytest.raises(RuntimeError, match="qdrant down"):
            await run(spec, registry)

    assert not upload_dir.exists(), "upload dir must be cleaned even on error"


async def test_run_yaml_append_error_is_surfaced_in_stdout(tmp_path):
    """yaml append failure is caught and logged to stdout, not raised."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    (upload_dir / "doc.pdf").write_text("content")

    spec = _FakeSpec(
        {
            "name": "new_col_err",
            "file_paths": [str(upload_dir / "doc.pdf")],
            "upload_dir": str(upload_dir),
        }
    )
    registry = _FakeRegistry()
    result = _make_result(children_written=2, is_new_collection=True)

    with (
        patch(
            "claritymed.web.admin.job_runners.rag_ingest.ingest_system_rag",
            new=AsyncMock(return_value=result),
        ),
        patch(
            "claritymed.web.admin.job_runners.rag_ingest.append_system_rag_collection",
            side_effect=OSError("disk full"),
        ),
    ):
        await run(spec, registry)  # must not raise

    assert any("append failed" in line for line in registry.stdout_lines)
