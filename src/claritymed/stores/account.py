"""Account I/O, first-user-is-admin bootstrap, and the ``require_admin`` guard.

Account state lives in ``~/.claritymed/data/users/<id>/settings.yaml`` —
plain YAML so the admin module can later expose a "view all accounts"
listing without ever touching ``profile.db`` (which holds PHI).

``current_account()`` caches by ``(user_id, mtime)`` so editing settings.yaml
hot-reloads on the next call without an explicit restart.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from claritymed import config as _cfg
from claritymed.context import MissingContextError, user_id_ctx
from claritymed.core.observability.audit import audit_event
from claritymed.core.schemas import Account, Role
from claritymed.errors import PermissionDeniedError, UserIdMismatch
from claritymed.stores.paths import (
    list_user_ids,
    user_root,
    user_settings_path,
    user_uploads_dir,
    validate_user_id,
)


class AccountStore:
    """Reads / writes one user's ``settings.yaml``.

    The store does not enforce role — it is a pure I/O layer. Callers that
    mutate state must call ``require_admin()`` themselves.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = validate_user_id(user_id)
        self.path: Path = user_settings_path(self.user_id)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> Account:
        if not self.path.exists():
            raise FileNotFoundError(f"no settings.yaml for user {self.user_id!r}")
        with self.path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return Account.model_validate(raw)

    def save(self, account: Account) -> None:
        if account.user_id != self.user_id:
            raise UserIdMismatch(
                f"account.user_id {account.user_id!r} does not match "
                f"store {self.user_id!r}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(account.model_dump(mode="json"), fh, sort_keys=False)


def init_user(user_id: str, display_name: str | None = None) -> Account:
    """Create the per-user tree and write settings.yaml. Idempotent.

    When ``list_user_ids()`` is empty (first install), the new account is
    auto-promoted to ``admin``. Subsequent calls default to ``user``.
    """
    _cfg.ensure_runtime_dirs()
    user_id = validate_user_id(user_id)
    store = AccountStore(user_id)
    if store.exists():
        return store.load()

    is_first = len(list_user_ids()) == 0
    role: Role = "admin" if is_first else "user"

    user_root(user_id).mkdir(parents=True, exist_ok=True)
    user_uploads_dir(user_id).mkdir(parents=True, exist_ok=True)
    account = Account(
        user_id=user_id,
        display_name=display_name or user_id,
        role=role,
        language=_cfg.default_lang(),  # type: ignore[arg-type]
    )
    store.save(account)

    # Audit needs user_id_ctx; init_user is often called outside a request
    # (CLI bootstrap), so only emit when context is set.
    if user_id_ctx.get():
        audit_event(
            "account_created",
            payload={"user_id": user_id, "role": role, "first_user": is_first},
        )
    return account


@lru_cache(maxsize=64)
def _cached_account(user_id: str, mtime: float) -> Account:
    return AccountStore(user_id).load()


def current_account() -> Account:
    """Return the ``Account`` for the user_id ContextVar. mtime-cached."""
    uid = user_id_ctx.get()
    if not uid:
        raise MissingContextError("user_id_ctx is not set")
    path = user_settings_path(uid)
    if not path.exists():
        raise FileNotFoundError(
            f"current user {uid!r} has no settings.yaml — "
            "did you forget to init_user()?"
        )
    return _cached_account(uid, path.stat().st_mtime)


def reset_account_cache() -> None:
    """Test / admin hook: forget all cached accounts."""
    _cached_account.cache_clear()


def require_admin() -> None:
    """Raise ``PermissionDeniedError`` if the current user is not admin.

    Always emits an audit event so blocked attempts are traceable.
    """
    account = current_account()
    if account.role != "admin":
        audit_event(
            "require_admin_blocked",
            payload={
                "user_id": account.user_id,
                "actual_role": account.role,
            },
        )
        raise PermissionDeniedError(f"user {account.user_id!r} is not admin")
    audit_event("require_admin_pass", payload={"user_id": account.user_id})
