"""Pydantic request / response models for the web v1 API.

Every endpoint declares its request and response model here (or in a
per-router schema module). FastAPI's auto-generated ``/openapi.json``
becomes the published contract; the frontend hand-mirrors these shapes
in TypeScript until codegen lands with the first admin-endpoint PR.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from claritymed.core.schemas.account import Language, Role

# Mirrors ``stores.paths.USER_ID_RE``. Declared in regex form for
# Pydantic so a malformed user_id arrives back as 422 with a clear
# constraint message rather than tripping the store's validator with a
# 500.
USER_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,32}$"


class ErrorEnvelope(BaseModel):
    """Common error body. FastAPI's default 4xx/5xx body uses ``{"detail"}``
    too — declaring it as a schema lets OpenAPI consumers pick it up.
    """

    model_config = ConfigDict(extra="forbid")

    detail: str


# --- /auth/login -------------------------------------------------------


class LoginRequest(BaseModel):
    """Credentials body for ``POST /auth/login``.

    ``user_id`` is regex-validated so the store layer never sees a
    path-traversal attempt. ``password`` is a plain string with no
    upper bound; bcrypt itself caps at 72 bytes and silently truncates
    anything longer (passlib emits a deprecation warning).
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(pattern=USER_ID_PATTERN)
    password: str = Field(min_length=1)


class LoginResponse(BaseModel):
    """Successful-login response body.

    The JWT itself is NEVER in the body — it lives only in the
    ``Set-Cookie: access_token`` header with ``HttpOnly``. The body
    carries the Account fields the SPA needs to render the header /
    nav bar without an extra ``GET /api/v1/me`` round-trip.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str
    display_name: str
    role: Role
    language: Language


# --- /auth/logout: no body, no response model -------------------------
# 204 No Content; cookie cleared via Set-Cookie.

LogoutMethod = Literal["POST"]
