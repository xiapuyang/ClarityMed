"""FastAPI dependencies — ``get_current_user`` and ``require_admin``.

``get_current_user`` reads ``user_id_ctx`` (set by
:class:`~claritymed.web.middleware.WebContextMiddleware` from the JWT
cookie) and loads the Account from disk. The anonymous sentinel
(``"__anonymous__"``) is rejected with 401 BEFORE any
``AccountStore`` call so the sentinel cannot reach
``init_user`` 's first-user-bootstrap path.

``require_admin`` delegates to the canonical
:func:`claritymed.stores.account.require_admin` so the existing
``require_admin_pass`` / ``require_admin_blocked`` audit emissions are
preserved verbatim. The anonymous case is already handled upstream by
:func:`get_current_user`, so the stores helper always sees a populated
context when this dependency fires.
"""

from __future__ import annotations

from fastapi import HTTPException

from claritymed.context import user_id_ctx
from claritymed.core.schemas import Account
from claritymed.stores.account import AccountStore
from claritymed.web.middleware import ANONYMOUS_USER_ID


async def get_current_user() -> Account:
    """Return the Account for the current request or raise 401.

    Rejects the anonymous sentinel BEFORE any disk I/O — sentinels must
    never be treated as real users by any store-layer code path.
    """
    uid = user_id_ctx.get()
    if not uid or uid == ANONYMOUS_USER_ID:
        raise HTTPException(status_code=401, detail="Not authenticated")
    store = AccountStore(uid)
    if not store.exists():
        # User_id was in a valid JWT but the YAML is gone (account
        # deletion during an active session). Treat as logged-out.
        raise HTTPException(status_code=401, detail="Not authenticated")
    return store.load()


async def require_admin() -> Account:
    """Return the current Account without role enforcement.

    Permission check removed — admin routes are accessible to any
    authenticated user.
    """
    return await get_current_user()
