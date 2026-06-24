"""Admin user-management endpoints.

CRUD over per-user ``settings.yaml`` plus an out-of-band password reset.
``settings.yaml`` is the project's single source of truth for accounts
(no JOINs / no FK rule), so listing iterates ``list_user_ids()`` and
each detail reads the YAML directly.

Self-protection: an admin cannot demote themselves (would lock the
admin surface entirely once they're the only one). The 400 message
explains the rule rather than a generic 403.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field

from claritymed.context import user_id_ctx
from claritymed.core.observability.audit import audit_event
from claritymed.core.schemas import Account
from claritymed.stores.account import AccountStore, reset_account_cache
from claritymed.stores.auth import PasswordStore
from claritymed.stores.models import load_models
from claritymed.stores.paths import list_user_ids, validate_user_id
from claritymed.web.deps import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["admin", "users"])


class AdminUserSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    display_name: str
    role: Literal["admin", "user"]
    language: Literal["en", "zh"]
    provider_id: str | None
    active_system_rag_collections: list[str]


class AdminUserListResponse(BaseModel):
    items: list[AdminUserSummary]
    total_count: int
    offset: int
    limit: int


class AdminUserPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=64)
    role: Literal["admin", "user"] | None = None
    language: Literal["en", "zh"] | None = None
    provider_id: str | None = Field(default=None, max_length=64)
    active_system_rag_collections: list[str] | None = None


class AdminPasswordReset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_password: str = Field(min_length=4, max_length=256)


def _to_summary(account: Account) -> AdminUserSummary:
    return AdminUserSummary(
        user_id=account.user_id,
        display_name=account.display_name,
        role=account.role,
        language=account.language,
        provider_id=account.provider_id,
        active_system_rag_collections=list(account.active_system_rag_collections),
    )


@router.get("", response_model=AdminUserListResponse)
async def list_users(
    response: Response,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> AdminUserListResponse:
    """Return all accounts visible on disk."""
    ids = list_user_ids()
    response.headers["X-Total-Count"] = str(len(ids))
    page_ids = ids[offset : offset + limit]
    items: list[AdminUserSummary] = []
    for uid in page_ids:
        try:
            items.append(_to_summary(AccountStore(uid).load()))
        except FileNotFoundError:
            continue
    return AdminUserListResponse(
        items=items, total_count=len(ids), offset=offset, limit=limit
    )


@router.get("/{user_id}", response_model=AdminUserSummary)
async def get_user(user_id: str) -> AdminUserSummary:
    validate_user_id(user_id)
    store = AccountStore(user_id)
    if not store.exists():
        raise HTTPException(status_code=404, detail=f"user {user_id!r} not found")
    return _to_summary(store.load())


@router.patch("/{user_id}", response_model=AdminUserSummary)
async def patch_user(
    user_id: str,
    patch: AdminUserPatch,
    current: Account = Depends(get_current_user),
) -> AdminUserSummary:
    validate_user_id(user_id)
    store = AccountStore(user_id)
    if not store.exists():
        raise HTTPException(status_code=404, detail=f"user {user_id!r} not found")
    account = store.load()
    self_demoting = (
        current.user_id == user_id and patch.role is not None and patch.role != "admin"
    )
    if self_demoting:
        raise HTTPException(
            status_code=400,
            detail=(
                "admins may not demote themselves — promote another admin "
                "first, then sign out and back in to confirm the role drop"
            ),
        )
    if patch.provider_id is not None and patch.provider_id != account.provider_id:
        catalog_ids = {p.id for p in load_models().providers}
        if patch.provider_id not in catalog_ids:
            raise HTTPException(
                status_code=400,
                detail=f"unknown provider id {patch.provider_id!r}",
            )
    fields_to_update: dict[str, Any] = {}
    for field_name in (
        "display_name",
        "role",
        "language",
        "provider_id",
        "active_system_rag_collections",
    ):
        value = getattr(patch, field_name)
        if value is not None:
            fields_to_update[field_name] = value
    if not fields_to_update:
        return _to_summary(account)
    data = account.model_dump(mode="python")
    data.update(fields_to_update)
    new_account = Account.model_validate(data)
    store.save(new_account)
    reset_account_cache()
    audit_event(
        "admin.user.update",
        payload={
            "target_uid": user_id,
            "fields_changed": sorted(fields_to_update.keys()),
        },
    )
    return _to_summary(new_account)


@router.post("/{user_id}/reset-password", status_code=204)
async def reset_password(
    user_id: str,
    body: AdminPasswordReset,
    _: Account = Depends(get_current_user),
) -> Response:
    validate_user_id(user_id)
    store = AccountStore(user_id)
    if not store.exists():
        raise HTTPException(status_code=404, detail=f"user {user_id!r} not found")
    PasswordStore.set_password(user_id, body.new_password)
    # Apply the target uid in the audit payload only — the request actor
    # is the admin (already in user_id_ctx).
    audit_event(
        "admin.user.reset_password",
        payload={"target_uid": user_id, "actor_uid": user_id_ctx.get()},
    )
    return Response(status_code=204)
