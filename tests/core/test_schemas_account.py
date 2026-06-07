"""Tests for ``Account`` (auth metadata, deliberately not PHI)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claritymed.core.schemas import Account


def test_happy_admin():
    a = Account(user_id="alice", display_name="Alice", role="admin", language="zh")
    assert a.cloud_provider_opt_in is False


def test_defaults_user_and_english():
    a = Account(user_id="bob", display_name="Bob")
    assert a.role == "user"
    assert a.language == "en"


def test_role_is_mutable_after_construction():
    """Demotion is a normal action; require_admin guards who may invoke it."""
    a = Account(user_id="alice", display_name="Alice", role="admin")
    a.role = "user"  # not frozen
    assert a.role == "user"


def test_unicode_display_name():
    a = Account(user_id="xiao_ming", display_name="小明")
    assert a.display_name == "小明"


def test_invalid_role_rejected():
    with pytest.raises(ValidationError):
        Account(user_id="alice", display_name="Alice", role="root")  # type: ignore[arg-type]


def test_extra_phi_field_blocked():
    with pytest.raises(ValidationError):
        Account(
            user_id="alice",
            display_name="Alice",
            medications=[],  # type: ignore[call-arg]
        )


def test_blank_display_name_rejected():
    with pytest.raises(ValidationError):
        Account(user_id="alice", display_name="")


def test_invalid_user_id_rejected():
    with pytest.raises(ValidationError):
        Account(user_id="../etc", display_name="x")


def test_model_dump_has_no_phi_keys():
    """The Account / Patient split contract: dumping an Account never reveals PHI."""
    a = Account(user_id="alice", display_name="Alice", role="admin")
    keys = set(a.model_dump().keys())
    expected = {
        "user_id",
        "display_name",
        "role",
        "language",
        "cloud_provider_opt_in",
        "created_at",
        "updated_at",
    }
    assert keys == expected
