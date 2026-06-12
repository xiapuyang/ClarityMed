"""Tests for ``stores.models.resolve_provider`` precedence."""

from __future__ import annotations

import pytest

from claritymed.core.schemas import Account
from claritymed.errors import CloudOptInRequiredError, UnknownProviderError
from claritymed.stores.models import load_models, resolve_provider


def test_default_when_nothing_overrides():
    p = resolve_provider()
    assert p.id == load_models().default_provider


def test_account_override_beats_default_when_opted_in():
    """Account-level provider_id wins over the catalog default when set."""
    a = Account(
        user_id="alice",
        display_name="Alice",
        provider_id="claude",
        cloud_provider_opt_in=True,
    )
    p = resolve_provider(account=a)
    assert p.id == "claude"


def test_cli_override_beats_account():
    """``--provider`` (override) is the strongest opt-in signal — exempt from
    the per-account cloud opt-in check by design."""
    a = Account(user_id="alice", display_name="Alice", provider_id="claude")
    p = resolve_provider(override="deepseek-v4-flash", account=a)
    assert p.id == "deepseek-v4-flash"


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


def test_account_cloud_without_opt_in_raises():
    """Per-user cloud opt-in is the third leg of the documented three-AND
    invariant. An account pointing at a cloud provider with
    ``cloud_provider_opt_in=False`` must NOT silently resolve to that
    provider — that would let an unconsented user's PHI reach the cloud."""
    a = Account(
        user_id="alice",
        display_name="Alice",
        provider_id="claude",
        cloud_provider_opt_in=False,
    )
    with pytest.raises(CloudOptInRequiredError, match="cloud_provider_opt_in"):
        resolve_provider(account=a)


def test_account_local_provider_allowed_regardless_of_opt_in():
    """The opt-in check only applies to cloud providers; local providers
    are always permitted for any account."""
    a = Account(
        user_id="alice",
        display_name="Alice",
        provider_id="ollama",
        cloud_provider_opt_in=False,
    )
    p = resolve_provider(account=a)
    assert p.id == "ollama"
    assert p.kind == "local"


def test_override_to_cloud_exempt_from_opt_in_check():
    """Explicit ``--provider`` from the CLI is the human's strongest opt-in
    signal — it bypasses the account opt-in gate so an operator can still
    test a cloud provider one-shot without flipping the persisted flag."""
    a = Account(
        user_id="alice",
        display_name="Alice",
        cloud_provider_opt_in=False,
    )
    p = resolve_provider(override="claude", account=a)
    assert p.id == "claude"
    assert p.kind == "cloud"


def test_default_to_cloud_blocked_when_account_opts_out():
    """When ``ModelsConfig.default_provider`` is a cloud entry, an account
    without opt-in still gets blocked — the fallback path is not a safety
    bypass."""
    # We don't mutate the YAML here; instead simulate the default-cloud
    # case directly by setting account.provider_id = the default and
    # asserting the same gate fires.
    default_id = load_models().default_provider
    default = next(p for p in load_models().providers if p.id == default_id)
    if default.kind != "cloud":
        pytest.skip(
            "default provider is local; this test only applies to cloud-default"
        )
    a = Account(user_id="alice", display_name="Alice", cloud_provider_opt_in=False)
    with pytest.raises(CloudOptInRequiredError):
        resolve_provider(account=a)


def test_resolved_provider_carries_full_config():
    """The resolver returns a full ProviderConfig — callers read kind/model
    from it. (`kind` is what the PHI guard branches on; `model` is the
    pydantic-ai prefix string that build_model() consumes.)"""
    p = resolve_provider(override="claude")
    assert p.kind == "cloud"
    assert p.model.startswith("anthropic:")


# ---------- is_provider_available ----------


def test_is_provider_available_returns_false_for_unknown_prefix():
    """A typoed prefix (e.g. ``"openni:gpt-4o"``) must surface as
    unavailable so the operator can read the error message at provider
    selection time, not deep inside pydantic-ai's model dispatch.

    Regression guard for ce:review P1 #8 — the optimistic ``return
    True`` for unknown prefixes was hiding misconfiguration.
    """
    from claritymed.core.schemas import ProviderConfig
    from claritymed.stores.models import is_provider_available

    p = ProviderConfig(
        id="typo",
        kind="cloud",
        model="openni:gpt-4o",  # deliberate typo (open**ni**, not openai)
    )
    assert is_provider_available(p) is False


def test_is_provider_available_returns_true_for_known_prefix_with_env(monkeypatch):
    """Known prefix + env var present → available. Sanity check the
    happy path didn't regress alongside the unknown-prefix fix."""
    from claritymed.core.schemas import ProviderConfig
    from claritymed.stores.models import is_provider_available

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    p = ProviderConfig(id="ok", kind="cloud", model="openai:gpt-4o")
    assert is_provider_available(p) is True


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
