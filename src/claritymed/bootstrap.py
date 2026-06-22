"""Host-agnostic one-shot process bootstrap.

Every entry point (Typer CLI subcommand, Textual TUI, FastAPI web worker)
must, exactly once per process:

* load ``~/.claritymed/.env`` so provider API keys are visible;
* configure the four named loggers (``app`` / ``access`` / ``audit`` /
  ``llm``) with rotating-file handlers + WARNING-level console echo;
* silence noisy library loggers (httpx, urllib3, …) so the operator's
  terminal doesn't drown the real warnings.

This module is the single source of truth for that sequence. CLI's
``cli.common.bootstrap_once`` and web's ``app.lifespan`` both call into
:func:`bootstrap_once` here; the function is idempotent, so duplicate
calls from nested Typer callbacks or test fixtures are safe.

Why this is not in ``core/``: ``setup_logging`` lives in
``core/observability/logging.py`` (which this module imports), but the
*orchestration* of "env + loggers, once, with a script label" is
process-startup glue that doesn't belong inside ``core``. Living at the
package root keeps it import-cheap for the CLI startup path while still
re-usable from ``web/app.py``.
"""

from __future__ import annotations

import logging

from claritymed import config as _cfg
from claritymed.core.observability.logging import setup_logging

logger = logging.getLogger(__name__)

_BOOTSTRAPPED = False


def bootstrap_once(
    script_name: str = "claritymed",
    console_level: int | None = logging.WARNING,
) -> None:
    """Load ``.env`` + initialise the four loggers. Idempotent.

    Args:
        script_name: Label written to ``app.log`` as the ``START: ...``
            banner. CLI passes ``"claritymed"``; web passes
            ``"claritymed-web"``. Distinguishes log streams when a single
            developer machine has both running.
        console_level: Console handler level for the ``app`` logger.
            CLI defaults to ``WARNING`` so operator errors surface
            without enabling debug logging. Web also uses ``WARNING``
            because uvicorn provides its own access log when
            ``CLARITYMED_DEV=1``.

    First call wins; subsequent calls in the same process are no-ops.
    The TUI is a special case — it calls bootstrap via the Typer
    callback (``cli/main.py``) *before* Textual takes over stderr, so
    the LazyStderrHandler captures the real fd and pre-TUI fatal errors
    still reach the terminal.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _cfg.load_env_file()
    setup_logging(script_name=script_name, console_level=console_level)
    _BOOTSTRAPPED = True


def prefetch_models() -> None:
    """Download in-process model weights to HF cache before serving traffic.

    Currently only the PHI-scrub model (``configs/scrub.yaml``,
    ``privacy_filter.enabled=true``). Fails fast via ``SystemExit(1)``
    when the runtime deps are missing or the download itself fails —
    cloud-bound PHI requests cannot proceed without a working scrubber,
    so half-booting and crashing on the first request is worse than a
    clean startup failure.

    Both TUI (``cli/commands/tui.py``) and the web worker
    (``web/app.py:lifespan``) call this after :func:`bootstrap_once`
    so the failure surfaces in the same console handler. A no-op when
    ``privacy_filter.enabled=false``.
    """
    from claritymed.core.scrub.service import ScrubService

    svc = ScrubService.from_config()
    if not svc._config.privacy_filter.enabled:
        return
    try:
        svc.check_runtime_deps()
    except ImportError as exc:
        logger.error("startup error: %s", exc)
        raise SystemExit(1) from exc
    ok = svc.ensure_downloaded()
    if not ok:
        model = svc._config.privacy_filter.model_name
        logger.error("model %s download failed", model)
        raise SystemExit(1)


__all__ = ["bootstrap_once", "prefetch_models"]
