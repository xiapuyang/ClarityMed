"""CLI entry helper: one context manager wires the three ContextVars.

Concrete command functions (``claritymed ask``, ``claritymed init-user``, …)
live in a later plan; this is the helper they will all open with::

    with inject_context(user_id=args.user, language=args.lang):
        ...
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

from claritymed import config as _cfg
from claritymed.context import apply_context, new_request_id, reset_context
from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.logging import get_access_logger
from claritymed.core.observability.tracing import setup_tracing

DEFAULT_USER_ID = "default"

logger = logging.getLogger(__name__)


def _resolve_language(explicit: str | None) -> str:
    supported = _cfg.supported_langs()
    if explicit and explicit.lower() in supported:
        return explicit.lower()
    env_value = (os.environ.get("CLARITYMED_LANG") or "").strip().lower()
    if env_value in supported:
        return env_value
    if env_value:
        logger.warning(
            "CLARITYMED_LANG=%r is not supported; falling back to app.yaml",
            env_value,
        )
    fallback = _cfg.default_lang()
    return fallback if fallback in supported else _cfg.DEFAULT_LANG_FALLBACK


def _resolve_user_id(explicit: str | None) -> tuple[str, bool]:
    """Return (user_id, used_default)."""
    if explicit:
        return explicit, False
    env_value = os.environ.get("CLARITYMED_USER", "").strip()
    if env_value:
        return env_value, False
    return DEFAULT_USER_ID, True


@contextmanager
def inject_context(
    user_id: str | None = None,
    language: str | None = None,
    request_id: str | None = None,
    command: str | None = None,
    check_user_exists: bool = False,
) -> Iterator[tuple[str, str, str]]:
    """Set request / user / language ContextVars for the with-block.

    Emits ``request_start`` and ``request_end`` audit events. Warns once when
    the user_id falls through to ``"default"`` so the operator notices a
    missing ``--user`` flag.

    ``command`` is a short human-readable label written to access.log so each
    line identifies what CLI operation ran (e.g. ``"ask"``, ``"ingest.profile"``).

    ``check_user_exists=True`` raises ``UserNotFoundError`` when the resolved
    user has no ``settings.yaml`` on disk.  All production CLI commands set this
    so a typo in ``--user`` fails fast before any I/O.
    """
    rid = request_id or new_request_id()
    uid, used_default = _resolve_user_id(user_id)
    lang = _resolve_language(language)
    if check_user_exists:
        from claritymed.errors import UserNotFoundError
        from claritymed.stores.account import AccountStore

        if not AccountStore(uid).exists():
            raise UserNotFoundError(
                f"user {uid!r} not found — "
                "run 'claritymed init-user' to create a user account first"
            )
    if used_default:
        from claritymed.core.i18n import t

        logger.warning(t("ui.cli.user_required", lang=lang))

    # Idempotent — every CLI invocation calls this; only the first one
    # with tracing.enabled: true in app.yaml actually installs the provider.
    setup_tracing()

    tokens = apply_context(rid, uid, lang)
    access = get_access_logger()
    cmd_label = command or "unknown"
    try:
        audit_event("request_start", payload={"entry": "cli", "command": cmd_label})
        access.info("cli.start cmd=%s", cmd_label)
        yield rid, uid, lang
        audit_event("request_end", payload={"status": "ok"})
        access.info("cli.end cmd=%s status=ok", cmd_label)
    except BaseException:
        audit_event("request_end", payload={"status": "exception"})
        access.info("cli.end cmd=%s status=exception", cmd_label)
        raise
    finally:
        reset_context(tokens)
