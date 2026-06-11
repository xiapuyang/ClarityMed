"""Tests for ``ToolDispatcher``."""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.errors import PathOutsideUserDomain, UnknownSha256
from claritymed.orchestrator.services.tool_dispatcher import (
    ApprovalGateResult,
    ToolDispatcher,
    manifest_references_sha,
)
from claritymed.stores.blob_store import BlobStore


@pytest.fixture
def _ctx():
    tokens = apply_context("20260611000000ABCDEF12", "alice", "en")
    yield
    reset_context(tokens)


def test_validate_args_unknown_tool_raises(_ctx):
    d = ToolDispatcher()
    with pytest.raises(ValueError, match="unknown tool"):
        d.validate_args("bogus_tool", {})


def test_validate_args_returns_pydantic_model(_ctx):
    d = ToolDispatcher()
    args = {"substance": "penicillin", "severity": "severe", "source": "self_report"}
    parsed = d.validate_args("save_allergy", args)
    assert parsed.substance == "penicillin"


def test_validate_args_rejects_malformed(_ctx):
    d = ToolDispatcher()
    with pytest.raises(ValueError, match="invalid args"):
        d.validate_args(
            "save_allergy",
            {"substance": "p", "severity": "bogus", "source": "self_report"},
        )


def test_check_shas_empty_attachments_passes(_ctx):
    d = ToolDispatcher()
    d.check_shas({"attachments": []})


def test_check_shas_missing_sha_raises(_ctx):
    d = ToolDispatcher()
    sha = "a" * 64
    with pytest.raises(UnknownSha256):
        d.check_shas({"attachments": [{"sha256": sha, "filename": "x.pdf"}]})


def test_check_shas_session_sha_passes(_ctx):
    sha = "a" * 64
    d = ToolDispatcher(session_attachments=lambda: {sha})
    d.check_shas({"attachments": [{"sha256": sha, "filename": "x.pdf"}]})


def test_check_shas_blob_pool_sha_passes(_ctx):
    """A sha that's already on disk in blobs/<sha[:2]>/<sha>/ resolves."""
    bs = BlobStore("alice")
    sha = bs.store(b"hello", "txt")
    d = ToolDispatcher()
    d.check_shas({"attachments": [{"sha256": sha, "filename": "h.txt"}]})


def test_check_record_path_outside_raises(_ctx):
    d = ToolDispatcher()
    with pytest.raises(PathOutsideUserDomain):
        d.check_record_path({"record_path": "../../../etc/passwd"})


def test_check_record_path_inside_passes(_ctx):
    d = ToolDispatcher()
    d.check_record_path({"record_path": "exam-reports/2026-06-11-ab12cd34"})


def test_check_record_path_symlink_raises(tmp_path: Path, _ctx):
    """Symlinking into the user's own dir from outside must still fail."""
    from claritymed.stores.paths import user_records_dir

    root = user_records_dir("alice")
    root.mkdir(parents=True, exist_ok=True)
    # Create a real dir outside, then a symlink inside the user's tree
    # pointing at it.
    outside = tmp_path / "outside"
    outside.mkdir()
    target = root / "exam-reports" / "linked"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside)
    d = ToolDispatcher()
    with pytest.raises(PathOutsideUserDomain):
        d.check_record_path({"record_path": "exam-reports/linked"})


def test_gate_returns_allowed_when_rule_matches(_ctx):
    args = {
        "substance": "penicillin",
        "severity": "severe",
        "source": "self_report",
    }
    rule_match = lambda name, a: "rule-1" if name == "save_allergy" else None  # noqa: E731
    d = ToolDispatcher(rule_match=rule_match)
    result = d.gate("save_allergy", args)
    assert isinstance(result, ApprovalGateResult)
    assert result.allowed is True
    assert result.rule_id == "rule-1"


def test_gate_returns_not_allowed_when_no_rule(_ctx):
    args = {
        "substance": "penicillin",
        "severity": "severe",
        "source": "self_report",
    }
    d = ToolDispatcher(rule_match=lambda *_: None)
    result = d.gate("save_allergy", args)
    assert result.allowed is False
    assert result.rule_id is None


def test_manifest_references_sha_false_for_fresh_user(_ctx):
    """Empty records/library directories → no sha is referenced."""
    assert manifest_references_sha("alice", "a" * 64) is False
