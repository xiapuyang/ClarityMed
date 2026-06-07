"""Per-request ContextVars and the canonical ``request_id`` generator.

Three ContextVars travel together through one request:

* ``request_id_ctx`` — 22-char id (``YYYYMMDDhhmmss`` UTC + 8 uppercase hex)
* ``user_id_ctx`` — the local account this request is acting as
* ``language_ctx`` — language the answer must be produced in (``en`` / ``zh``)

Generators and validators live here so every entry point (CLI, FastAPI
middleware, tests) builds ids the same way. Audit-trail correctness is the
reason this lives at the package root rather than under ``core/``.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Optional

REQUEST_ID_PATTERN = re.compile(r"^[0-9]{14}[0-9A-F]{8}$")
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")

request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
language_ctx: ContextVar[Optional[str]] = ContextVar("language", default=None)


class MissingContextError(RuntimeError):
    """A store / guard needed a ContextVar that was not set."""


def new_request_id() -> str:
    """Return a fresh 22-char request id.

    Format ``YYYYMMDDhhmmss<8-uppercase-hex>`` (UTC). The timestamp prefix
    keeps log and audit lines naturally sortable by event time; the random
    suffix disambiguates within a second.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:8].upper()
    return f"{ts}{suffix}"


def is_valid_request_id(value: str) -> bool:
    """Pattern + timestamp ``strptime`` double-check.

    Used by the FastAPI middleware to reject hand-forged ``X-Request-ID``
    headers (a pattern match alone would accept, e.g., month 13).
    """
    if not value or not REQUEST_ID_PATTERN.match(value):
        return False
    try:
        datetime.strptime(value[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return False
    return True


def apply_context(
    request_id: str, user_id: str, language: str
) -> tuple[Token, Token, Token]:
    """Set all three ContextVars and return their reset tokens."""
    return (
        request_id_ctx.set(request_id),
        user_id_ctx.set(user_id),
        language_ctx.set(language),
    )


def reset_context(tokens: tuple[Token, Token, Token]) -> None:
    """Reset all three ContextVars using tokens from ``apply_context``."""
    request_id_ctx.reset(tokens[0])
    user_id_ctx.reset(tokens[1])
    language_ctx.reset(tokens[2])


def get_context_or_raise() -> tuple[str, str, str]:
    """Return (request_id, user_id, language) or raise ``MissingContextError``.

    Stores call this to refuse "blind" reads that would scan across users.
    """
    rid = request_id_ctx.get()
    uid = user_id_ctx.get()
    lang = language_ctx.get()
    if not rid or not uid or not lang:
        msg = (
            "request_id / user_id / language ContextVars must all be set "
            "(use cli.entry.inject_context or the FastAPI middleware)."
        )
        raise MissingContextError(msg)
    return rid, uid, lang
