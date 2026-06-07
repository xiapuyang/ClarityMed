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
    assert paths.shared_qdrant_dir() == _cfg.SHARED_DIR / "qdrant"
    assert paths.shared_knowledge_raw_dir() == _cfg.SHARED_DIR / "knowledge" / "raw"
    assert (
        paths.shared_vision_models_dir("rash", "v1")
        == _cfg.SHARED_DIR / "vision_models" / "rash" / "v1"
    )


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
