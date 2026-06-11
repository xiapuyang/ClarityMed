"""Tests for ``claritymed.stores.paths``."""

from __future__ import annotations

import pytest

from claritymed import config as _cfg
from claritymed.errors import InvalidUserIdError
from claritymed.stores import paths


def test_user_root_normal():
    assert paths.user_root("alice") == _cfg.DATA_DIR / "users" / "alice"


def test_user_root_with_special_chars():
    assert paths.user_root("alice_v2-1") == _cfg.DATA_DIR / "users" / "alice_v2-1"


def test_user_root_rejects_path_traversal():
    with pytest.raises(InvalidUserIdError):
        paths.user_root("../etc/passwd")


def test_user_root_rejects_empty():
    with pytest.raises(InvalidUserIdError):
        paths.user_root("")


def test_user_root_rejects_too_long():
    with pytest.raises(InvalidUserIdError):
        paths.user_root("a" * 33)


def test_user_root_rejects_slash():
    with pytest.raises(InvalidUserIdError):
        paths.user_root("alice/bob")


def test_shared_paths_resolve_under_shared_dir():
    assert paths.shared_root() == _cfg.SHARED_DIR
    assert paths.shared_knowledge_raw_dir() == _cfg.SHARED_DIR / "knowledge" / "raw"
    assert (
        paths.shared_vision_models_dir("rash", "v1")
        == _cfg.SHARED_DIR / "vision_models" / "rash" / "v1"
    )


def test_user_rag_qdrant_dir_is_per_user():
    """Each user's local-mode user_rag storage must live in their own dir.

    This is the file-isolation guarantee: two users' qdrant directories
    are physically separate, so a bug that picks the wrong path can't
    cross-leak data. The path also stays under ``user_root`` so existing
    per-user OS file-mode bits apply.
    """
    alice = paths.user_rag_qdrant_dir("alice")
    bob = paths.user_rag_qdrant_dir("bob")
    assert alice != bob
    assert alice == paths.user_root("alice") / "qdrant"
    assert bob == paths.user_root("bob") / "qdrant"


def test_list_user_ids_ignores_dangling_empty_dir():
    """A directory with no settings.yaml must not count as a user — that's
    the first-admin-bootstrap correctness guard."""
    users_root = _cfg.DATA_DIR / "users"
    (users_root / "ghost").mkdir(parents=True)  # bare dir, no settings.yaml
    assert paths.list_user_ids() == []


def test_data_and_shared_disjoint():
    data = _cfg.DATA_DIR.resolve()
    shared = _cfg.SHARED_DIR.resolve()
    assert not data.is_relative_to(shared)
    assert not shared.is_relative_to(data)


# --- v1 PHI storage helpers --------------------------------------------


def test_user_records_dir_with_category():
    assert (
        paths.user_records_dir("alice", "exam-reports")
        == _cfg.DATA_DIR / "users" / "alice" / "records" / "exam-reports"
    )


def test_user_records_dir_without_category():
    assert (
        paths.user_records_dir("alice") == _cfg.DATA_DIR / "users" / "alice" / "records"
    )


def test_user_record_dir_composes_three_levels():
    p = paths.user_record_dir("alice", "exam-reports", "2026-06-11-ab12cd34")
    assert p.name == "2026-06-11-ab12cd34"
    assert p.parent.name == "exam-reports"
    assert p.parent.parent.name == "records"


def test_user_record_dir_rejects_traversal_slug():
    with pytest.raises(ValueError):
        paths.user_record_dir("alice", "exam-reports", "..evil")


def test_user_record_dir_rejects_traversal_category():
    with pytest.raises(ValueError):
        paths.user_record_dir("alice", "../etc", "x")


def test_user_library_dir_mirrors_records():
    assert (
        paths.user_library_dir("alice", "papers")
        == _cfg.DATA_DIR / "users" / "alice" / "library" / "papers"
    )


def test_user_blob_dir_is_two_level_cas():
    sha = "ab" + "0" * 62
    p = paths.user_blob_dir("alice", sha)
    assert p == paths.user_blobs_dir("alice") / "ab" / sha


def test_user_blob_path_rejects_non_hex_sha():
    with pytest.raises(ValueError):
        paths.user_blob_path("alice", "not-a-sha", "pdf")


def test_user_blob_path_rejects_wrong_length_sha():
    with pytest.raises(ValueError):
        paths.user_blob_path("alice", "ab" * 31, "pdf")


def test_user_blob_path_rejects_bad_extension():
    with pytest.raises(ValueError):
        paths.user_blob_path("alice", "a" * 64, "")
    with pytest.raises(ValueError):
        paths.user_blob_path("alice", "a" * 64, ".pdf")
    with pytest.raises(ValueError):
        paths.user_blob_path("alice", "a" * 64, "sub/dir")


def test_user_session_attachments_path():
    sid = "1234-uuid"
    assert (
        paths.user_session_attachments_path("alice", sid)
        == _cfg.DATA_DIR / "users" / "alice" / "session" / sid / "attachments.json"
    )


def test_user_session_dir_rejects_traversal():
    with pytest.raises(ValueError):
        paths.user_session_dir("alice", "../etc")


def test_user_audit_payload_path():
    rid = "20260611000000ABCDEF12"
    assert (
        paths.user_audit_payload_path("alice", rid)
        == _cfg.DATA_DIR / "users" / "alice" / "audit_payloads" / f"{rid}.json"
    )


def test_user_audit_payload_path_rejects_traversal_request_id():
    with pytest.raises(ValueError):
        paths.user_audit_payload_path("alice", "../etc/passwd")


def test_user_parent_docstores_are_distinct():
    """PHI and library docstore JSON files must be physically separate."""
    phi = paths.user_parent_docstore_phi_path("alice")
    lib = paths.user_parent_docstore_library_path("alice")
    assert phi != lib
    assert phi.parent == lib.parent == paths.user_root("alice")
