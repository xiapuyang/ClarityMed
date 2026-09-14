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


def test_load_ignores_unknown_top_level_keys():
    """settings.yaml is shared with SettingsStore (``approvals``) and may
    carry fields from older schema versions (``cloud_provider_opt_in``).
    AccountStore.load must filter to known Account fields, not crash."""
    init_user("alice")
    path = user_settings_path("alice")
    import yaml as _yaml

    raw = _yaml.safe_load(path.read_text())
    raw["cloud_provider_opt_in"] = False  # removed-from-schema legacy field
    raw["approvals"] = {  # owned by SettingsStore — never an Account concern
        "rules": [{"id": "x", "tool": "t", "action": "allow", "args_pattern": {}}]
    }
    path.write_text(_yaml.safe_dump(raw, sort_keys=False))
    acct = AccountStore("alice").load()
    assert acct.user_id == "alice"
    assert acct.role == "admin"


def test_save_preserves_unknown_top_level_keys():
    """Round-trip: an ``approvals`` block written by SettingsStore must
    survive a subsequent AccountStore.save (e.g. when the user switches
    provider). Without the merge, save would clobber it."""
    init_user("alice")
    path = user_settings_path("alice")
    import yaml as _yaml

    raw = _yaml.safe_load(path.read_text())
    raw["approvals"] = {"rules": [{"id": "rule-1", "tool": "save_record"}]}
    path.write_text(_yaml.safe_dump(raw, sort_keys=False))

    store = AccountStore("alice")
    acct = store.load()
    # Simulate the /provider switch flow: app reads, modifies, saves.
    store.save(acct.model_copy(update={"provider_id": "ollama"}))

    after = _yaml.safe_load(path.read_text())
    assert after["provider_id"] == "ollama"
    assert after["approvals"] == {"rules": [{"id": "rule-1", "tool": "save_record"}]}


def test_save_cross_store_roundtrip_does_not_lose_approvals():
    """End-to-end: SettingsStore.add_rule then AccountStore.save then
    SettingsStore.list_rules must still see the rule. This is the
    scenario the architecture bug actually broke in production."""
    from claritymed.stores.settings_store import SettingsStore

    init_user("alice")
    settings = SettingsStore("alice")
    rule = settings.add_rule(
        tool="save_allergy",
        args_pattern={"severity": "mild"},
        ttl_hours=24,
    )
    # Now a /provider switch persists the Account.
    acc_store = AccountStore("alice")
    acc_store.save(acc_store.load().model_copy(update={"provider_id": "ollama"}))
    # The rule must still be there.
    survivors = settings.list_rules()
    assert [r.id for r in survivors] == [rule.id]


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
