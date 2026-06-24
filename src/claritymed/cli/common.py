"""Shared CLI helpers used by every subcommand module.

Lives in its own module so each ``cli/commands/*.py`` can import only what
it actually needs without dragging in unrelated subcommand wiring. The
``Console`` instance is module-level so subcommands share one Rich
console (matching the previous single-file behaviour).
"""

from __future__ import annotations

import asyncio
import enum
import logging

from rich.console import Console

from claritymed.bootstrap import (
    bootstrap_once as _shared_bootstrap_once,
    prefetch_models as _shared_prefetch_models,
)


class CLIEmergencySensitivity(str, enum.Enum):
    """CLI-accepted sensitivity values for ``--emergency-sensitivity``.

    A deliberate subset of the runtime
    :data:`~claritymed.core.emergency.schemas.SensitivityName` literal:
    ``"off"`` is **excluded**. The runtime supports four values; the CLI
    surface only three.

    Why: ``settings.yaml`` requires ``off_acknowledged_at`` alongside
    ``sensitivity: off`` (CLAUDE.md off-mode safeguard #3 — the
    "two-step opt-out" timestamp). A CLI flag completely bypasses that
    validator because the override never round-trips through the YAML
    file, so allowing ``--emergency-sensitivity off`` would silently
    void the safeguard.

    Operator escape hatches that *do* preserve observability:
    1. ``CLARITYMED_FORCE_EMERGENCY_GATE=off`` env var (master switch;
       downgrades requested ``off`` to ``lenient`` at the runtime
       layer — note this is *not* identical to ``off``).
    2. Hand-edit ``data/users/<uid>/settings.yaml`` with both
       ``sensitivity: "off"`` and ``off_acknowledged_at`` — the
       Pydantic validator catches the two-step requirement here.
    """

    strict = "strict"
    balanced = "balanced"
    lenient = "lenient"


# One Rich console for every subcommand. Module-level singleton matches
# the pre-split behaviour and keeps colour / theming settings consistent.
console = Console()

logger = logging.getLogger(__name__)


def bootstrap_once() -> None:
    """CLI-flavoured bootstrap — thin delegator to :mod:`claritymed.bootstrap`.

    Kept here so existing ``from claritymed.cli.common import
    bootstrap_once`` imports across subcommand modules keep working. New
    code should import from :mod:`claritymed.bootstrap` directly.
    """
    _shared_bootstrap_once(script_name="claritymed", console_level=logging.WARNING)


def run_async(coro):
    """Run an async coroutine on a fresh event loop. Thin asyncio.run wrapper."""
    return asyncio.run(coro)


def try_current_account():
    """Return the account for the current ContextVar user, or None.

    Used by commands that have already entered ``inject_context``.
    Returning None rather than raising lets ``resolve_provider`` fall
    through to its default path when no settings.yaml exists yet (fresh
    install). When the file does exist, the account is returned so the
    cloud-opt-in invariant gets enforced.
    """
    try:
        from claritymed.stores.account import current_account

        return current_account()
    except Exception:  # noqa: BLE001 — best-effort lookup
        return None


def try_load_account(user_id: str):
    """Load an account by user_id without opening a ContextVar scope.

    Used by ``tui`` which resolves the provider before entering
    ``inject_context``. Returns None when the user has no settings.yaml
    yet (fresh install) or any read error.
    """
    try:
        from claritymed.stores.account import AccountStore

        store = AccountStore(user_id)
        if not store.exists():
            return None
        return store.load()
    except Exception:  # noqa: BLE001
        return None


def prefetch_models() -> None:
    """Thin delegator — see :func:`claritymed.bootstrap.prefetch_models`.

    Kept so existing ``from claritymed.cli.common import prefetch_models``
    imports in subcommand modules keep working. New code should import
    from :mod:`claritymed.bootstrap` directly.
    """
    _shared_prefetch_models()
