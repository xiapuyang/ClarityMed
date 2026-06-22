"""``/api/v1/me`` — read + partial-update the current user's Account.

``GET`` returns the Account (sans PHI / sans password hash). ``PATCH``
accepts ``display_name``, ``language``, and ``provider_id``; any other
field in the body is rejected by Pydantic ``extra="forbid"`` so a buggy
frontend can't silently mutate ``role`` via this endpoint.

When ``language`` changes, the JWT cookie is reissued with the new
``lang`` claim so the WebContextMiddleware on the next request sees a
coherent (cookie, AccountStore) pair. The frontend gates
``i18n.changeLanguage()`` on the ``useMe`` refetch landing, never on
the optimistic mutation result — see the plan's A6 deepening note.

When ``provider_id`` changes, the router validates against the live
catalog and rejects unknown values with 422. The actual provider
resolution for the next stream call goes through ``resolve_provider``
again, so a Yet-Unknown id never lands on disk only to fail later.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from claritymed.core.observability.audit import audit_event
from claritymed.core.schemas import Account
from claritymed.stores.account import AccountStore, reset_account_cache
from claritymed.stores.models import load_models
from claritymed.web.deps import get_current_user
from claritymed.web.jwt import create_token
from claritymed.web.middleware import COOKIE_ACCESS_TOKEN
from claritymed.web.schemas import AccountResponse, MePatch

router = APIRouter(prefix="/api/v1", tags=["me"])

# Same lifetime as /auth/login. Centralized as a module constant rather
# than imported from the auth router to keep the two endpoints free to
# diverge if MVP follow-ups need different cookie policies (e.g. shorter
# session-after-language-change to force re-auth, etc.).
_COOKIE_MAX_AGE_S = 7 * 24 * 60 * 60


@router.get("/me", response_model=AccountResponse)
async def get_me(account: Account = Depends(get_current_user)) -> AccountResponse:
    """Return the Account for the authenticated user."""
    return _to_response(account)


@router.patch("/me", response_model=AccountResponse)
async def patch_me(
    patch: MePatch,
    request: Request,
    response: Response,
    account: Account = Depends(get_current_user),
) -> AccountResponse:
    """Update display_name, language, and/or provider_id.

    Reissues JWT on language change. Validates provider_id against the
    catalog (422 on unknown). No-op patches return the existing Account
    unchanged.
    """
    new_display = (
        patch.display_name if patch.display_name is not None else account.display_name
    )
    new_language = patch.language if patch.language is not None else account.language
    new_provider_id = (
        patch.provider_id if patch.provider_id is not None else account.provider_id
    )

    if patch.provider_id is not None and patch.provider_id != account.provider_id:
        catalog_ids = {p.id for p in load_models().providers}
        if patch.provider_id not in catalog_ids:
            audit_event(
                "web.me.unknown_provider",
                payload={"requested": patch.provider_id},
            )
            raise HTTPException(
                status_code=422,
                detail=f"Unknown provider id {patch.provider_id!r}",
            )

    if (
        new_display == account.display_name
        and new_language == account.language
        and new_provider_id == account.provider_id
    ):
        return _to_response(account)

    # ``model_copy(update=…)`` skips validators — but display_name,
    # language, and provider_id are simple str/Literal fields, no
    # coercion needed. model_dump → mutate → model_validate is the safe
    # form if any constraint tightens later. For now, model_copy is fine.
    updated = account.model_copy(
        update={
            "display_name": new_display,
            "language": new_language,
            "provider_id": new_provider_id,
        }
    )
    AccountStore(account.user_id).save(updated)
    # Invalidate the (user_id, mtime) cache so the next ``current_account``
    # read sees the updated fields rather than the pre-save copy.
    reset_account_cache()

    if new_provider_id != account.provider_id:
        audit_event(
            "web.me.provider_changed",
            payload={
                "from": account.provider_id or "<default>",
                "to": new_provider_id or "<default>",
            },
        )

    if new_language != account.language:
        token = create_token(account.user_id, new_language)
        response.set_cookie(
            COOKIE_ACCESS_TOKEN,
            token,
            max_age=_COOKIE_MAX_AGE_S,
            httponly=True,
            samesite="lax",
            secure=_secure_cookie(request),
            path="/",
        )
        audit_event(
            "web.me.language_changed",
            payload={"from": account.language, "to": new_language},
        )

    return _to_response(updated)


def _to_response(account: Account) -> AccountResponse:
    return AccountResponse(
        user_id=account.user_id,
        display_name=account.display_name,
        role=account.role,
        language=account.language,
        provider_id=account.provider_id,
    )


def _secure_cookie(request: Request) -> bool:
    return not getattr(request.app.state, "dev", False)
