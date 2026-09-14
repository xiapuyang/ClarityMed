"""Tests for ``stores.settings_store.SettingsStore``."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.stores.settings_store import (
    _RULES_PER_TOOL_CAP,
    ApprovalRule,
    SettingsStore,
)


@pytest.fixture
def _ctx():
    tokens = apply_context("20260611000000ABCDEF12", "test", "en")
    yield
    reset_context(tokens)


def test_add_rule_round_trip(_ctx):
    store = SettingsStore("test")
    rule = store.add_rule("save_allergy", {"severity": "mild"})
    assert isinstance(rule, ApprovalRule)
    assert rule.tool == "save_allergy"
    rules = store.list_rules()
    assert any(r.id == rule.id for r in rules)


def test_match_rule_returns_first_match(_ctx):
    store = SettingsStore("test")
    store.add_rule("save_allergy", {"severity": "mild"})
    hit = store.match_rule(
        "save_allergy",
        {"substance": "peanut", "severity": "mild", "source": "self_report"},
    )
    assert hit is not None
    assert hit.tool == "save_allergy"


def test_match_rule_no_match_on_different_pattern(_ctx):
    store = SettingsStore("test")
    store.add_rule("save_allergy", {"severity": "mild"})
    hit = store.match_rule("save_allergy", {"severity": "severe"})
    assert hit is None


def test_match_rule_no_match_on_different_tool(_ctx):
    store = SettingsStore("test")
    store.add_rule("save_allergy", {"severity": "mild"})
    hit = store.match_rule("save_medication", {"severity": "mild"})
    assert hit is None


def test_opaque_keys_excluded_from_pattern(_ctx):
    """sha256 / record_path / attachments are stripped from the pattern."""
    store = SettingsStore("test")
    rule = store.add_rule(
        "save_record",
        {"category": "exam-reports", "sha256": "a" * 64},
    )
    assert "sha256" not in rule.args_pattern
    assert rule.args_pattern == {"category": "exam-reports"}


def test_replace_on_duplicate_pattern(_ctx):
    """Adding a rule with the same (tool, pattern) replaces — doesn't stack."""
    store = SettingsStore("test")
    r1 = store.add_rule("save_allergy", {"severity": "mild"})
    r2 = store.add_rule("save_allergy", {"severity": "mild"})
    rules = store.list_rules()
    assert len(rules) == 1
    assert r1.id != r2.id  # new rule replaces, gets new id
    assert rules[0].id == r2.id


def test_revoke_rule_returns_true_on_hit(_ctx):
    store = SettingsStore("test")
    rule = store.add_rule("save_allergy", {"severity": "mild"})
    assert store.revoke_rule(rule.id) is True
    assert store.list_rules() == []


def test_revoke_rule_returns_false_on_miss(_ctx):
    store = SettingsStore("test")
    assert store.revoke_rule("does-not-exist") is False


def test_expired_rule_pruned_on_list(_ctx):
    store = SettingsStore("test")
    store.add_rule("save_allergy", {"severity": "mild"})
    # Simulate the clock advancing past TTL.
    future = datetime.now(timezone.utc) + timedelta(days=8)
    rules = store.list_rules(now=future)
    assert rules == []


def test_per_tool_cap_evicts_oldest(_ctx):
    store = SettingsStore("test")
    for i in range(_RULES_PER_TOOL_CAP + 3):
        store.add_rule("save_allergy", {"k": i})
    rules = store.list_rules()
    assert len(rules) == _RULES_PER_TOOL_CAP


def test_deny_beats_allow_on_same_shape(_ctx):
    store = SettingsStore("test")
    store.add_rule("save_allergy", {"severity": "mild"}, action="allow")
    store.add_rule("save_allergy", {"severity": "mild"}, action="deny")
    hit = store.match_rule("save_allergy", {"severity": "mild"})
    assert hit is not None
    assert hit.action == "deny"


def test_corrupt_yaml_returns_empty_rules(_ctx, tmp_path, monkeypatch):
    """Total-corrupt settings.yaml ⇒ fail closed, no rules considered."""
    store = SettingsStore("test")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("not: valid: yaml: [", encoding="utf-8")
    assert store.list_rules() == []


def test_malformed_single_rule_discards_full_list(_ctx):
    """Per fail-closed contract: one bad entry ⇒ treat the list as empty."""
    import yaml

    store = SettingsStore("test")
    store.add_rule("save_allergy", {"severity": "mild"})
    raw = yaml.safe_load(store.path.read_text(encoding="utf-8"))
    raw["approvals"]["rules"].append(
        {"id": "x", "tool": "save_allergy", "extra": "bad"}
    )
    store.path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert store.list_rules() == []


def test_matches_skips_opaque_key_in_pattern(_ctx):
    """ApprovalRule.matches() continues past opaque keys like sha256."""
    now = datetime.now(timezone.utc)
    rule = ApprovalRule.model_construct(
        id="test-id",
        tool="save_record",
        action="allow",
        args_pattern={"sha256": "a" * 64, "name": "exam"},
        granted_at=now,
        ttl_hours=24,
        expires_at=now + timedelta(hours=24),
    )
    assert rule.matches("save_record", {"name": "exam"}) is True


def test_matches_returns_false_when_arg_value_differs(_ctx):
    """matches() returns False when a pattern value doesn't match the call args."""
    now = datetime.now(timezone.utc)
    rule = ApprovalRule.model_construct(
        id="test-id",
        tool="save_allergy",
        action="allow",
        args_pattern={"severity": "mild"},
        granted_at=now,
        ttl_hours=24,
        expires_at=now + timedelta(hours=24),
    )
    assert rule.matches("save_allergy", {"severity": "severe"}) is False


def test_save_raises_oserror_when_rename_fails(_ctx, monkeypatch):
    """OSError during atomic rename propagates after tmp cleanup."""
    from pathlib import Path

    store = SettingsStore("test")

    _real_replace = Path.replace

    def _bad_replace(self, target):
        if str(self).endswith(".tmp"):
            raise OSError("disk full")
        return _real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _bad_replace)
    with pytest.raises(OSError, match="disk full"):
        store.add_rule("save_allergy", {})


def test_list_rules_prune_takes_file_lock_and_rereads(_ctx, monkeypatch):
    """ADV-006 regression: ``list_rules`` prune-write-back must take the
    same ``file_lock`` ``add_rule`` uses, and re-read state inside the
    lock. Otherwise a concurrent ``add_rule`` lands between our pre-lock
    snapshot and our write, and our stale snapshot silently overwrites
    the freshly-persisted rule.

    This test asserts the two structural invariants directly (lock
    acquired on slow path; second ``_load_raw`` happens inside the
    lock). A true cross-process race would need a subprocess harness;
    this checks the in-process contract the fix guarantees.
    """
    import yaml
    from datetime import timedelta

    from claritymed.stores import settings_store as ss_mod

    store = SettingsStore("test")
    # Seed: one expired rule (backdate via direct YAML write — the
    # model rejects past expires_at via construction).
    store.add_rule("save_allergy", {"severity": "mild"})
    past = _utcnow_minus(hours=8 * 24)
    raw = yaml.safe_load(store.path.read_text(encoding="utf-8"))
    raw["approvals"]["rules"][0]["granted_at"] = past.isoformat()
    raw["approvals"]["rules"][0]["expires_at"] = (past + timedelta(hours=1)).isoformat()
    store.path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    events: list[str] = []

    real_load_raw = store._load_raw
    real_file_lock = ss_mod.file_lock

    def _spy_load_raw():
        events.append("load_raw")
        return real_load_raw()

    @contextlib.contextmanager
    def _spy_file_lock(path, timeout=10.0):
        events.append("lock_acquire")
        with real_file_lock(path, timeout=timeout):
            yield
        events.append("lock_release")

    monkeypatch.setattr(store, "_load_raw", _spy_load_raw)
    monkeypatch.setattr(ss_mod, "file_lock", _spy_file_lock)

    store.list_rules()

    # Contract: fast-path load, then lock, then re-read inside the lock,
    # then release. The locked re-read is the structural defense against
    # ADV-006 — without it the prune writes a stale snapshot.
    assert events == [
        "load_raw",
        "lock_acquire",
        "load_raw",
        "lock_release",
    ], f"slow-path lock contract violated: {events}"


def test_list_rules_fast_path_skips_lock_when_nothing_expired(_ctx, monkeypatch):
    """Cost guard for the fix in
    ``test_list_rules_prune_takes_file_lock_and_rereads``: when nothing
    needs pruning we still take zero locks. Reads are hot; an
    unconditional ``file_lock`` would serialize them needlessly."""
    from claritymed.stores import settings_store as ss_mod

    store = SettingsStore("test")
    store.add_rule("save_allergy", {"severity": "mild"})

    locks_taken = {"n": 0}
    real_file_lock = ss_mod.file_lock

    @contextlib.contextmanager
    def _counting_lock(path, timeout=10.0):
        locks_taken["n"] += 1
        with real_file_lock(path, timeout=timeout):
            yield

    monkeypatch.setattr(ss_mod, "file_lock", _counting_lock)

    store.list_rules()
    assert locks_taken["n"] == 0


def _utcnow_minus(*, hours: int):
    from datetime import timedelta

    return _utcnow() - timedelta(hours=hours)


def _utcnow():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
