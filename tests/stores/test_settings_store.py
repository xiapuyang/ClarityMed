"""Tests for ``stores.settings_store.SettingsStore``."""

from __future__ import annotations

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
    tokens = apply_context("20260611000000ABCDEF12", "alice", "en")
    yield
    reset_context(tokens)


def test_add_rule_round_trip(_ctx):
    store = SettingsStore("alice")
    rule = store.add_rule("save_allergy", {"severity": "mild"})
    assert isinstance(rule, ApprovalRule)
    assert rule.tool == "save_allergy"
    rules = store.list_rules()
    assert any(r.id == rule.id for r in rules)


def test_match_rule_returns_first_match(_ctx):
    store = SettingsStore("alice")
    store.add_rule("save_allergy", {"severity": "mild"})
    hit = store.match_rule(
        "save_allergy",
        {"substance": "peanut", "severity": "mild", "source": "self_report"},
    )
    assert hit is not None
    assert hit.tool == "save_allergy"


def test_match_rule_no_match_on_different_pattern(_ctx):
    store = SettingsStore("alice")
    store.add_rule("save_allergy", {"severity": "mild"})
    hit = store.match_rule("save_allergy", {"severity": "severe"})
    assert hit is None


def test_match_rule_no_match_on_different_tool(_ctx):
    store = SettingsStore("alice")
    store.add_rule("save_allergy", {"severity": "mild"})
    hit = store.match_rule("save_medication", {"severity": "mild"})
    assert hit is None


def test_opaque_keys_excluded_from_pattern(_ctx):
    """sha256 / record_path / attachments are stripped from the pattern."""
    store = SettingsStore("alice")
    rule = store.add_rule(
        "save_record",
        {"category": "exam-reports", "sha256": "a" * 64},
    )
    assert "sha256" not in rule.args_pattern
    assert rule.args_pattern == {"category": "exam-reports"}


def test_replace_on_duplicate_pattern(_ctx):
    """Adding a rule with the same (tool, pattern) replaces — doesn't stack."""
    store = SettingsStore("alice")
    r1 = store.add_rule("save_allergy", {"severity": "mild"})
    r2 = store.add_rule("save_allergy", {"severity": "mild"})
    rules = store.list_rules()
    assert len(rules) == 1
    assert r1.id != r2.id  # new rule replaces, gets new id
    assert rules[0].id == r2.id


def test_revoke_rule_returns_true_on_hit(_ctx):
    store = SettingsStore("alice")
    rule = store.add_rule("save_allergy", {"severity": "mild"})
    assert store.revoke_rule(rule.id) is True
    assert store.list_rules() == []


def test_revoke_rule_returns_false_on_miss(_ctx):
    store = SettingsStore("alice")
    assert store.revoke_rule("does-not-exist") is False


def test_expired_rule_pruned_on_list(_ctx):
    store = SettingsStore("alice")
    store.add_rule("save_allergy", {"severity": "mild"})
    # Simulate the clock advancing past TTL.
    future = datetime.now(timezone.utc) + timedelta(days=8)
    rules = store.list_rules(now=future)
    assert rules == []


def test_per_tool_cap_evicts_oldest(_ctx):
    store = SettingsStore("alice")
    for i in range(_RULES_PER_TOOL_CAP + 3):
        store.add_rule("save_allergy", {"k": i})
    rules = store.list_rules()
    assert len(rules) == _RULES_PER_TOOL_CAP


def test_deny_beats_allow_on_same_shape(_ctx):
    store = SettingsStore("alice")
    store.add_rule("save_allergy", {"severity": "mild"}, action="allow")
    store.add_rule("save_allergy", {"severity": "mild"}, action="deny")
    hit = store.match_rule("save_allergy", {"severity": "mild"})
    assert hit is not None
    assert hit.action == "deny"


def test_corrupt_yaml_returns_empty_rules(_ctx, tmp_path, monkeypatch):
    """Total-corrupt settings.yaml ⇒ fail closed, no rules considered."""
    store = SettingsStore("alice")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("not: valid: yaml: [", encoding="utf-8")
    assert store.list_rules() == []


def test_malformed_single_rule_discards_full_list(_ctx):
    """Per fail-closed contract: one bad entry ⇒ treat the list as empty."""
    import yaml

    store = SettingsStore("alice")
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

    store = SettingsStore("alice")

    _real_replace = Path.replace

    def _bad_replace(self, target):
        if str(self).endswith(".tmp"):
            raise OSError("disk full")
        return _real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _bad_replace)
    with pytest.raises(OSError, match="disk full"):
        store.add_rule("save_allergy", {})
