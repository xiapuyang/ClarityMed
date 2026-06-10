"""Three separate loggers: app (debug), access (HTTP / CLI requests), audit.

Architecture §6.5 forbids mixing debug noise with the audit trail; this module
puts each on its own logger + handler + file with ``propagate=False``. The
formatter is a single class that reads ``request_id`` / ``user_id`` /
``language`` ContextVars on every emit so any caller using a stdlib logger
gets the right labels automatically.

LOG_DIR is resolved on each ``setup_logging()`` / ``get_*_logger()`` call by
reading ``claritymed.config`` as a module (not a name binding), so tests that
reload the config module pick up the new path.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from claritymed import config as _cfg
from claritymed.context import language_ctx, request_id_ctx, user_id_ctx

APP_LOGGER = "claritymed"
ACCESS_LOGGER = "claritymed.access"
AUDIT_LOGGER = "claritymed.audit"
LLM_LOGGER = "claritymed.llm"

APP_FMT = (
    "%(asctime)s [%(levelname)s] [%(request_id)s][%(user_id)s] "
    "%(relpath)s:%(lineno)d - %(message)s"
)
ACCESS_FMT = "%(asctime)s [%(request_id)s][%(user_id)s][%(language)s] %(message)s"
AUDIT_FMT = "%(asctime)s [%(request_id)s][%(user_id)s] [%(language)s] %(message)s"

NOISY_LOGGERS = (
    "urllib3",
    "uvicorn",
    "starlette",
    "multipart",
    "python_multipart",
    "qdrant_client",
    "lancedb",
    "httpx",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _LazyStderrHandler(logging.StreamHandler):
    """StreamHandler that reads sys.stderr at emit time rather than init time.

    logging.StreamHandler captures sys.stderr once at construction. That
    breaks Click's test runner (which redirects sys.stderr per invocation)
    and Textual (which replaces sys.stderr with an internal pipe). Reading
    at emit time matches the behaviour of plain ``print(file=sys.stderr)``.
    """

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


class ClarityMedFormatter(logging.Formatter):
    """Inject relpath + the three request ContextVars on each record."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            record.relpath = os.path.relpath(record.pathname, _PROJECT_ROOT)
        except ValueError:
            record.relpath = record.pathname
        record.request_id = request_id_ctx.get() or "-"
        record.user_id = user_id_ctx.get() or "-"
        record.language = language_ctx.get() or "-"
        return super().format(record)


def _file_handler(
    log_dir: Path, filename: str, fmt: str, max_bytes: int, backup_count: int
) -> RotatingFileHandler:
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(ClarityMedFormatter(fmt))
    return handler


def _reset_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    return logger


def _configured_file_level() -> int:
    """Resolve file log level: CLARITYMED_LOG_LEVEL env > app.yaml logging.level > DEBUG."""
    level_str = (
        os.environ.get("CLARITYMED_LOG_LEVEL")
        or _cfg.load_yaml("app.yaml").get("logging", {}).get("level")
        or "DEBUG"
    )
    return getattr(logging, str(level_str).upper(), logging.DEBUG)


def setup_logging(
    script_name: str,
    console_level: int | None = logging.INFO,
) -> logging.Logger:
    """Configure all three loggers. Idempotent — clears handlers first."""
    _cfg.ensure_runtime_dirs()
    log_dir = _cfg.LOG_DIR
    file_level = _configured_file_level()

    app = _reset_logger(APP_LOGGER)
    app.setLevel(file_level)
    app.propagate = False
    handler = _file_handler(log_dir, "app.log", APP_FMT, 5 * 1024 * 1024, 5)
    handler.setLevel(file_level)
    app.addHandler(handler)

    if console_level is not None:
        ch = _LazyStderrHandler()
        ch.setLevel(console_level)
        ch.setFormatter(
            ClarityMedFormatter(
                "%(asctime)s [%(levelname)s] [%(request_id)s][%(user_id)s] %(message)s"
            )
        )
        app.addHandler(ch)

    # access, audit, and llm loggers are configured by their getters on
    # demand, but we tear them down here so a re-call of setup_logging()
    # does not leave duplicate handlers from a previous run.
    _reset_logger(ACCESS_LOGGER)
    _reset_logger(AUDIT_LOGGER)
    _reset_logger(LLM_LOGGER)

    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    app.info("=" * 60)
    app.info("START: %s", script_name)
    app.info("-" * 60)
    return app


def get_access_logger() -> logging.Logger:
    """Lazily configure and return the access logger."""
    logger = logging.getLogger(ACCESS_LOGGER)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(
            _file_handler(_cfg.LOG_DIR, "access.log", ACCESS_FMT, 10 * 1024 * 1024, 10)
        )
    return logger


def get_audit_logger() -> logging.Logger:
    """Lazily configure and return the audit logger.

    Larger rotation budget — audit logs are retained longer for compliance.
    """
    logger = logging.getLogger(AUDIT_LOGGER)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.addHandler(
            _file_handler(_cfg.LOG_DIR, "audit.log", AUDIT_FMT, 20 * 1024 * 1024, 20)
        )
    return logger


def get_llm_logger() -> logging.Logger:
    """Lazily configure and return the llm debug logger."""
    logger = logging.getLogger(LLM_LOGGER)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(
            _file_handler(_cfg.LOG_DIR, "llm.log", "%(message)s", 20 * 1024 * 1024, 5)
        )
    return logger
