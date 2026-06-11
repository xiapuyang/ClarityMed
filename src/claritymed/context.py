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

import logging
import re
import uuid
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

REQUEST_ID_PATTERN = re.compile(r"^[0-9]{14}[0-9A-F]{8}$")
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")

request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
language_ctx: ContextVar[Optional[str]] = ContextVar("language", default=None)
# A TUI launch / API session id (uuid4). Shared by ``ChatSession`` (which writes
# ``sessions/<sid>.jsonl``) and the new ``SessionAttachments`` store (which
# writes ``session/<sid>/attachments.json``). Same id across both so
# ``/resume <id>`` rehydrates the attachments tray as well as the chat log.
session_id_ctx: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


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


BAGGAGE_REQUEST_ID = "claritymed.request_id"
BAGGAGE_USER_ID = "claritymed.user_id"
BAGGAGE_SESSION_ID = "claritymed.session_id"


def apply_context(
    request_id: str, user_id: str, language: str
) -> tuple[Token, Token, Token, object | None]:
    """Set all three ContextVars and return their reset tokens.

    The fourth tuple element is an opaque OpenTelemetry attach token (or
    ``None`` if OTel is not importable). It carries ``request_id`` and
    ``user_id`` as baggage so any span started inside this with-block can
    correlate back to the audit log line of the same request. Pass the
    same tuple back to :func:`reset_context` to detach cleanly.
    """
    rid_token = request_id_ctx.set(request_id)
    uid_token = user_id_ctx.set(user_id)
    lang_token = language_ctx.set(language)
    otel_token = _attach_baggage(request_id, user_id)
    return (rid_token, uid_token, lang_token, otel_token)


def reset_context(
    tokens: tuple[Token, Token, Token] | tuple[Token, Token, Token, object | None],
) -> None:
    """Reset all three ContextVars using tokens from :func:`apply_context`.

    Accepts both the legacy 3-tuple and the new 4-tuple (which carries the
    OTel attach token) so callers older than the tracing-correlation work
    keep functioning.
    """
    request_id_ctx.reset(tokens[0])
    user_id_ctx.reset(tokens[1])
    language_ctx.reset(tokens[2])
    if len(tokens) >= 4 and tokens[3] is not None:  # type: ignore[misc]
        _detach_baggage(tokens[3])  # type: ignore[index]


def _attach_baggage(request_id: str, user_id: str) -> object | None:
    """Attach our two baggage keys onto the current OTel context.

    Returns the attach token (opaque, passed back to ``context.detach``),
    or ``None`` when OpenTelemetry is unavailable in the environment. Any
    failure is swallowed — request handling must not depend on telemetry.
    """
    try:
        from opentelemetry import baggage
        from opentelemetry import context as otel_context

        ctx = otel_context.get_current()
        ctx = baggage.set_baggage(BAGGAGE_REQUEST_ID, request_id, context=ctx)
        ctx = baggage.set_baggage(BAGGAGE_USER_ID, user_id, context=ctx)
        return otel_context.attach(ctx)
    except ImportError:
        return None
    except Exception:  # noqa: BLE001
        logger.warning("OTel baggage attach failed", exc_info=True)
        return None


def _detach_baggage(token: object) -> None:
    try:
        from opentelemetry import context as otel_context

        otel_context.detach(token)
    except ImportError:
        pass
    except Exception:  # noqa: BLE001
        logger.warning("OTel baggage detach failed", exc_info=True)


def attach_session_baggage(session_id: str) -> object | None:
    """Attach ``claritymed.session_id`` baggage on top of the current OTel
    context. Returns a detach token (or ``None`` if OTel is unavailable)
    that must be passed back to :func:`detach_session_baggage`.

    Unlike :func:`apply_context`, this is for state the caller picks up
    *after* the request boundary — e.g. ``AskService`` learns the active
    session_id from its ``ChatSession`` and stamps it on every span the
    LLM call produces, so a Phoenix trace can be grouped by conversation
    even though the audit log already carries the same id in its payload.
    """
    try:
        from opentelemetry import baggage
        from opentelemetry import context as otel_context

        ctx = otel_context.get_current()
        ctx = baggage.set_baggage(BAGGAGE_SESSION_ID, session_id, context=ctx)
        return otel_context.attach(ctx)
    except ImportError:
        return None
    except Exception:  # noqa: BLE001
        logger.warning("OTel session baggage attach failed", exc_info=True)
        return None


def detach_session_baggage(token: object | None) -> None:
    """Detach a baggage scope previously attached by
    :func:`attach_session_baggage`. Accepting ``None`` is intentional —
    callers can blindly pass back whatever the attach returned without a
    ``None`` check on the hot path."""
    if token is None:
        return
    _detach_baggage(token)


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
