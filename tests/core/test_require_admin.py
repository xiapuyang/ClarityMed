"""Tests for ``require_admin`` — the role guard."""

from __future__ import annotations

import pytest

from claritymed.context import MissingContextError
from claritymed.errors import PermissionDeniedError
from claritymed.stores.account import require_admin


def test_admin_passes(as_, trio):
    with as_(trio["admin"]):
        require_admin()  # should not raise


def test_admin_pass_emits_audit_event(as_, trio, tmp_path):
    with as_(trio["admin"]):
        require_admin()
    audit = (tmp_path / "logs" / "audit.log").read_text(encoding="utf-8")
    assert '"kind":"require_admin_pass"' in audit


def test_user_blocked(as_, trio):
    bob = trio["users"][0]
    with as_(bob):
        with pytest.raises(PermissionDeniedError):
            require_admin()


def test_user_blocked_emits_audit_event(as_, trio, tmp_path):
    bob = trio["users"][0]
    with as_(bob):
        with pytest.raises(PermissionDeniedError):
            require_admin()
    audit = (tmp_path / "logs" / "audit.log").read_text(encoding="utf-8")
    assert '"kind":"require_admin_blocked"' in audit
    assert '"actual_role":"user"' in audit


def test_missing_context_distinguished_from_blocked():
    """No ContextVar set -> MissingContextError, not PermissionDeniedError.

    Audit log gets nothing — there is no user identity to attribute the
    attempt to.
    """
    with pytest.raises(MissingContextError):
        require_admin()
