"""``build_rules`` registry behaviour."""

from __future__ import annotations

import pytest

from claritymed.core.audit import build_rules, list_rules


def test_build_rules_returns_all_by_default():
    rules = build_rules()
    names = [r.name for r in rules]
    assert "tool_announced_but_skipped" in names


def test_build_rules_subset_by_name():
    rules = build_rules(only=["tool_announced_but_skipped"])
    assert len(rules) == 1
    assert rules[0].name == "tool_announced_but_skipped"


def test_build_rules_unknown_name_raises():
    """Typos on the CLI must not silently no-op — fail loud."""
    with pytest.raises(KeyError):
        build_rules(only=["does_not_exist"])


def test_list_rules_returns_name_description_pairs():
    pairs = list_rules()
    assert pairs
    names = [name for name, _ in pairs]
    descs = [desc for _, desc in pairs]
    assert "tool_announced_but_skipped" in names
    assert all(isinstance(d, str) and d for d in descs)
