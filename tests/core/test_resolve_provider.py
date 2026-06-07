"""Tests for ``stores.models.resolve_provider`` precedence."""

from __future__ import annotations

import pytest

from claritymed.core.schemas import Account
from claritymed.errors import UnknownProviderError
from claritymed.stores.models import load_models, resolve_provider


def test_default_when_nothing_overrides():
    p = resolve_provider()
    assert p.id == load_models().default_provider


def test_account_override_beats_default():
    a = Account(user_id="alice", display_name="Alice", provider_id="claude")
    p = resolve_provider(account=a)
    assert p.id == "claude"


def test_cli_override_beats_account():
    a = Account(user_id="alice", display_name="Alice", provider_id="claude")
    p = resolve_provider(override="deepseek", account=a)
    assert p.id == "deepseek"


def test_account_provider_id_none_falls_through_to_default():
    a = Account(user_id="alice", display_name="Alice", provider_id=None)
    p = resolve_provider(account=a)
    assert p.id == load_models().default_provider


def test_unknown_override_raises():
    with pytest.raises(UnknownProviderError, match="override"):
        resolve_provider(override="not-a-real-provider")


def test_unknown_account_id_raises_not_silently_falls_back():
    """A typo in settings.yaml must surface — silently downgrading to the
    default would hide a misconfigured cloud opt-in."""
    a = Account(user_id="alice", display_name="Alice", provider_id="claud")  # typo
    with pytest.raises(UnknownProviderError, match="account"):
        resolve_provider(account=a)


def test_resolved_provider_carries_wire_format():
    """The resolver returns a full ProviderConfig — callers read api/kind from it."""
    p = resolve_provider(override="claude")
    assert p.kind == "cloud"
    assert p.api == "anthropic"
    assert p.api_key_env == "ANTHROPIC_API_KEY"
