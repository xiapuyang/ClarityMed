"""PasswordStore: bcrypt set/verify, atomic write, missing-file behaviour."""

from __future__ import annotations

import pytest
import yaml

from claritymed.errors import InvalidUserIdError
from claritymed.stores.auth import PASSWORD_FIELD, PasswordStore
from claritymed.stores.paths import user_root

TEST_USER = "test"


def test_set_then_verify_returns_true(test_user_dir):  # noqa: ARG001
    PasswordStore.set_password(TEST_USER, "correct horse battery staple")
    assert (
        PasswordStore.verify_password(TEST_USER, "correct horse battery staple") is True
    )


def test_verify_wrong_password_returns_false(test_user_dir):  # noqa: ARG001
    PasswordStore.set_password(TEST_USER, "right")
    assert PasswordStore.verify_password(TEST_USER, "wrong") is False


def test_verify_unknown_user_returns_false():
    # No file on disk; must return False without raising.
    assert PasswordStore.verify_password(TEST_USER, "anything") is False


def test_verify_corrupt_hash_returns_false(test_user_dir):
    # Write a malformed auth.yaml — verify must not raise.
    path = user_root(TEST_USER) / "auth.yaml"
    path.write_text("password_hash: not-a-bcrypt-hash\n", encoding="utf-8")
    assert PasswordStore.verify_password(TEST_USER, "anything") is False


def test_verify_yaml_not_mapping_returns_false(test_user_dir):
    path = user_root(TEST_USER) / "auth.yaml"
    path.write_text("[not, a, mapping]\n", encoding="utf-8")
    assert PasswordStore.verify_password(TEST_USER, "anything") is False


def test_set_password_writes_bcrypt_hash(test_user_dir):  # noqa: ARG001
    PasswordStore.set_password(TEST_USER, "hello-world")
    path = user_root(TEST_USER) / "auth.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert PASSWORD_FIELD in raw
    # passlib bcrypt hashes start with $2b$ (modern) or $2a$ (legacy)
    assert raw[PASSWORD_FIELD].startswith("$2")


def test_set_password_rotates_existing(test_user_dir):  # noqa: ARG001
    PasswordStore.set_password(TEST_USER, "first")
    PasswordStore.set_password(TEST_USER, "second")
    assert PasswordStore.verify_password(TEST_USER, "first") is False
    assert PasswordStore.verify_password(TEST_USER, "second") is True


@pytest.mark.parametrize(
    "bad",
    [
        "../etc/passwd",
        "with/slash",
        "this_user_id_is_far_too_long_to_be_valid_under_the_regex_xxxx",
        "",
    ],
)
def test_set_password_rejects_invalid_user_id(bad: str):
    with pytest.raises(InvalidUserIdError):
        PasswordStore.set_password(bad, "anything")


@pytest.mark.parametrize("bad", ["../etc/passwd", "with/slash", ""])
def test_verify_password_rejects_invalid_user_id(bad: str):
    with pytest.raises(InvalidUserIdError):
        PasswordStore.verify_password(bad, "anything")


@pytest.fixture
def test_user_dir(tmp_path):  # noqa: ARG001 — tmp_path picked up by isolation fixture
    """Ensure the user directory exists before PasswordStore writes."""
    from claritymed.stores.account import init_user

    init_user(TEST_USER, display_name="Test")
