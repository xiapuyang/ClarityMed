"""``/api/v1/me`` — read + partial-update the current user's Account.

``GET`` returns the Account (sans PHI / sans password hash). ``PATCH``
accepts ``display_name`` and ``language``; any other field in the body
is rejected by Pydantic ``extra="forbid"`` so a buggy frontend can't
silently mutate ``role`` or ``provider_id`` via this endpoint.

When ``language`` changes, the JWT cookie is reissued with the new
``lang`` claim so the WebContextMiddleware on the next request sees a
coherent (cookie, AccountStore) pair. The frontend gates
``i18n.changeLanguage()`` on the ``useMe`` refetch landing, never on
the optimistic mutation result — see the plan's A6 deepening note.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from claritymed.core.observability.audit import audit_event
from claritymed.core.schemas import Account
from claritymed.stores.account import AccountStore, reset_account_cache
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
    """Update display_name and/or language. Reissues JWT on lang change."""
    new_display = (
        patch.display_name if patch.display_name is not None else account.display_name
    )
    new_language = patch.language if patch.language is not None else account.language

    if new_display == account.display_name and new_language == account.language:
        return _to_response(account)

    # ``model_copy(update=…)`` skips validators — but display_name and
    # language are simple str/Literal fields, no coercion needed.
    # model_dump → mutate → model_validate is the safe form if either
    # constraint tightens later. For now, model_copy is fine.
    updated = account.model_copy(
        update={"display_name": new_display, "language": new_language}
    )
    AccountStore(account.user_id).save(updated)
    # Invalidate the (user_id, mtime) cache so the next ``current_account``
    # read sees the updated fields rather than the pre-save copy.
    reset_account_cache()

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
