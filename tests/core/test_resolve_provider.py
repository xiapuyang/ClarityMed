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


def test_resolved_provider_carries_full_config():
    """The resolver returns a full ProviderConfig — callers read kind/model
    from it. (`kind` is what the PHI guard branches on; `model` is the
    pydantic-ai prefix string that build_model() consumes.)"""
    p = resolve_provider(override="claude")
    assert p.kind == "cloud"
    assert p.model.startswith("anthropic:")


# ---------- pick_reachable_provider ----------


def test_pick_reachable_returns_first_reachable_provider(monkeypatch):
    """omlx and ollama both healthy + credentialed → first in probe order."""
    from claritymed.stores import models as _m

    monkeypatch.setattr(_m, "_local_service_up", lambda url, timeout_s=2.0: True)
    monkeypatch.setattr(_m, "is_provider_available", lambda p: True)

    picked = _m.pick_reachable_provider()
    assert picked is not None
    assert picked.id == "omlx"  # first in _LOCAL_HEALTH_URLS


def test_pick_reachable_skips_uncredentialed_provider(monkeypatch):
    from claritymed.stores import models as _m

    monkeypatch.setattr(_m, "_local_service_up", lambda url, timeout_s=2.0: True)
    monkeypatch.setattr(_m, "is_provider_available", lambda p: p.id != "omlx")

    picked = _m.pick_reachable_provider()
    assert picked is not None
    assert picked.id == "ollama"


def test_pick_reachable_skips_unreachable_provider(monkeypatch):
    from claritymed.stores import models as _m

    monkeypatch.setattr(_m, "is_provider_available", lambda p: True)
    monkeypatch.setattr(
        _m,
        "_local_service_up",
        lambda url, timeout_s=2.0: "omlx" not in url and "8000" not in url,
    )

    picked = _m.pick_reachable_provider()
    assert picked is not None
    assert picked.id == "ollama"


def test_pick_reachable_returns_none_when_nothing_up(monkeypatch):
    from claritymed.stores import models as _m

    monkeypatch.setattr(_m, "is_provider_available", lambda p: True)
    monkeypatch.setattr(_m, "_local_service_up", lambda url, timeout_s=2.0: False)

    assert _m.pick_reachable_provider() is None
