"""Shared CLI helpers used by every subcommand module.

Lives in its own module so each ``cli/commands/*.py`` can import only what
it actually needs without dragging in unrelated subcommand wiring. The
``Console`` instance is module-level so subcommands share one Rich
console (matching the previous single-file behaviour).
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console

from claritymed import config as _cfg
from claritymed.core.observability.logging import setup_logging

# One Rich console for every subcommand. Module-level singleton matches
# the pre-split behaviour and keeps colour / theming settings consistent.
console = Console()

_BOOTSTRAPPED = False


def bootstrap_once() -> None:
    """Idempotent CLI bootstrap: load ``~/.claritymed/.env`` then init the loggers.

    Called from the root Typer callback (every subcommand) and from
    one-shot entry points like ``init-user`` that may run before the
    callback. The first call wins; subsequent calls in the same process
    are no-ops.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _cfg.load_env_file()
    setup_logging(script_name="claritymed", console_level=None)
    _BOOTSTRAPPED = True


def stderr(msg: str) -> None:
    """Write a single message to stderr without Rich formatting.

    Kept separate from ``console.print`` so error lines round-trip cleanly
    when stdout is piped (CLI agent-readiness — stderr stays plain text).
    """
    print(msg, file=sys.stderr)


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
    """Download in-process model weights to HF cache before the TUI starts.

    Fails fast if ``privacy_filter.enabled=true`` but the required
    packages are not installed — a missing package is a misconfiguration,
    not a recoverable runtime condition.
    """
    from claritymed.core.scrub.service import ScrubService

    svc = ScrubService.from_config()
    if not svc._config.privacy_filter.enabled:
        return
    try:
        svc.check_runtime_deps()
    except ImportError as exc:
        print(f"startup error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    ok = svc.ensure_downloaded()
    if not ok:
        model = svc._config.privacy_filter.model_name
        print(f"  ✗ {model} download failed", file=sys.stderr)
        raise SystemExit(1)
