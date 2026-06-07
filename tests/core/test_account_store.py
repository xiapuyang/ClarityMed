"""Tests for ``claritymed.stores.account``."""

from __future__ import annotations

import time

import pytest

from claritymed.context import MissingContextError
from claritymed.errors import UserIdMismatch
from claritymed.stores.account import (
    AccountStore,
    current_account,
    init_user,
    reset_account_cache,
)
from claritymed.stores.paths import user_settings_path


def test_first_user_is_admin():
    a = init_user("alice")
    assert a.role == "admin"


def test_second_user_defaults_to_user():
    init_user("alice")
    b = init_user("bob")
    assert b.role == "user"


def test_init_user_idempotent():
    first = init_user("alice", display_name="Alice")
    again = init_user("alice", display_name="Renamed")  # second call must not overwrite
    assert again.display_name == first.display_name == "Alice"


def test_dangling_directory_does_not_count():
    """Deleting settings.yaml but leaving the dir must still treat next as first."""
    init_user("alice")
    user_settings_path("alice").unlink()
    # 'dave' arrives after alice's dir exists but with no settings.yaml —
    # list_user_ids() returns []; dave becomes admin.
    dave = init_user("dave")
    assert dave.role == "admin"


def test_save_rejects_user_id_mismatch():
    init_user("alice")
    init_user("bob")
    alice_store = AccountStore("alice")
    bob_account = AccountStore("bob").load()
    with pytest.raises(UserIdMismatch):
        alice_store.save(bob_account)


def test_current_account_requires_context():
    init_user("alice")
    with pytest.raises(MissingContextError):
        current_account()


def test_current_account_returns_account(as_, alice):
    with as_(alice):
        ac = current_account()
        assert ac.user_id == "alice"
        assert ac.role == "admin"


def test_current_account_hot_reloads_after_mtime_bump(as_, alice):
    """Editing settings.yaml on disk must be picked up on the next call."""
    with as_(alice):
        before = current_account()
        assert before.role == "admin"
        # Demote alice on disk + bump mtime.
        store = AccountStore("alice")
        demoted = before.model_copy(update={"role": "user"})
        store.save(demoted)
        new_mtime = time.time() + 5
        import os

        os.utime(store.path, (new_mtime, new_mtime))
        # Cache key is (user_id, mtime); fresh mtime invalidates.
        after = current_account()
        assert after.role == "user"
    reset_account_cache()
