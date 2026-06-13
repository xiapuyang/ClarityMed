"""Runtime configuration: paths, YAML loader, language default.

Three independent override layers:

    CLARITYMED_HOME  (single dial)  -> ~/.claritymed
        |                               |
        +------------+------------------+------------------+
                     |                  |                  |
    CLARITYMED_DATA_DIR  CLARITYMED_SHARED_DIR  CLARITYMED_LOG_DIR
       per-user PHI       admin-managed shared    logs
        ~/.claritymed/data  ~/.claritymed/shared    ~/.claritymed/logs

Per the foundation plan, ``DATA_DIR`` / ``SHARED_DIR`` / ``LOG_DIR`` are
evaluated once at import time (constant style). Tests that need to redirect
them must set env vars *before* importing ``claritymed`` (see ``tests/conftest.py``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src" / "claritymed"
CONFIGS_DIR = PROJECT_ROOT / "configs"
I18N_DIR = CONFIGS_DIR / "i18n"
PROMPTS_STORE = SRC_ROOT / "core" / "prompts" / "store"

CLARITYMED_HOME = Path(os.environ.get("CLARITYMED_HOME", Path.home() / ".claritymed"))
DATA_DIR = Path(os.environ.get("CLARITYMED_DATA_DIR", CLARITYMED_HOME / "data"))
SHARED_DIR = Path(os.environ.get("CLARITYMED_SHARED_DIR", CLARITYMED_HOME / "shared"))
LOG_DIR = Path(os.environ.get("CLARITYMED_LOG_DIR", CLARITYMED_HOME / "logs"))

DEFAULT_LANG_FALLBACK = "en"
DEFAULT_PASTE_MAX_FILE_SIZE_MB = 20


def ensure_runtime_dirs() -> None:
    """Create ``data/``, ``shared/``, and ``logs/`` under the runtime root.

    Idempotent. Called by ``setup_logging()`` and ``init_user()`` at first use.
    Subdirectories (``data/users/<id>/``, ``shared/knowledge/``, etc.) are
    created lazily by the code that first writes to them — keeping the top
    level empty when a feature has not been exercised yet.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=32)
def load_yaml(name: str) -> dict:
    """Load ``configs/<name>`` with ``yaml.safe_load``.

    Returns an empty dict when the file is missing. Result is cached in
    process; call ``reload_configs()`` to invalidate.
    """
    path = CONFIGS_DIR / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def default_lang() -> str:
    """Return ``app.yaml`` ``i18n.default_lang`` or ``"en"``."""
    return (
        load_yaml("app.yaml").get("i18n", {}).get("default_lang")
        or DEFAULT_LANG_FALLBACK
    )


def paste_max_file_size_bytes() -> int:
    """Max bytes accepted by a single drag-drop / Ctrl+V / /upload entry.

    Read from ``app.yaml`` ``paste.max_file_size_mb`` and converted to
    bytes. Falls back to ``DEFAULT_PASTE_MAX_FILE_SIZE_MB`` when missing
    so an unconfigured install still has a sensible ceiling.
    """
    mb = load_yaml("app.yaml").get("paste", {}).get("max_file_size_mb")
    if mb is None:
        mb = DEFAULT_PASTE_MAX_FILE_SIZE_MB
    return int(float(mb) * 1024 * 1024)


def supported_langs() -> tuple[str, ...]:
    """Return ``app.yaml`` ``i18n.supported_langs`` or ``(DEFAULT_LANG_FALLBACK,)``."""
    raw = load_yaml("app.yaml").get("i18n", {}).get("supported_langs")
    if not raw:
        return (DEFAULT_LANG_FALLBACK,)
    return tuple(str(x).lower() for x in raw)


def load_router_config() -> "RouterConfig":
    """Load and validate ``configs/router.yaml``.

    Single source of truth for confidence thresholds and classification rules
    used by the hybrid mode router. Wraps :func:`load_yaml` (cached) and
    validates with Pydantic so misspelled keys fail fast at load time.

    Raises:
        FileNotFoundError: If ``configs/router.yaml`` does not exist.
        pydantic.ValidationError: If the YAML is structurally wrong.
    """
    from claritymed.core.schemas.router import RouterConfig

    raw = load_yaml("router.yaml")
    if not raw:
        raise FileNotFoundError(
            "configs/router.yaml missing or empty — required for mode dispatch."
        )
    return RouterConfig.model_validate(raw)


def load_evals_config() -> "EvalsConfig":
    """Load and validate ``configs/evals.yaml``.

    Single source of truth for which benchmark tasks the runner executes by
    default, where per-run JSONL lands, and which provider (if any) acts as
    a judge. Wraps :func:`load_yaml` (cached) and validates with Pydantic so
    a typo fails fast at load time, not in the middle of a 20-minute run.

    Raises:
        FileNotFoundError: If ``configs/evals.yaml`` does not exist.
        pydantic.ValidationError: If the YAML is structurally wrong.
    """
    from claritymed.core.schemas.evals import EvalsConfig

    raw = load_yaml("evals.yaml")
    if not raw:
        raise FileNotFoundError(
            "configs/evals.yaml missing or empty — required for `claritymed eval`."
        )
    return EvalsConfig.model_validate(raw)


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from ``CLARITYMED_HOME/.env`` into ``os.environ``.

    Per-user keys (provider API keys, language overrides, default user) live
    in the runtime root rather than the repo, so an open-source clone never
    ships secrets. Lines starting with ``#`` are ignored. A leading
    ``export `` is stripped so the same file can be sourced by a shell.
    Existing env vars are preserved — the file is a default, not an override.

    Returns the dict of keys that were applied (useful for tests).
    """
    env_path = path or CLARITYMED_HOME / ".env"
    if not env_path.exists():
        return {}
    applied: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


_DEFAULT_TOOL_APPROVAL_TTL_HOURS = 24

_DEFAULT_INGEST_TOOL_MAX_RETRIES = 3

_DEFAULT_ASK_USER_QUESTION_MAX_RETRIES = 3


def tool_approval_rule_ttl_hours() -> int:
    """Return the TTL (in hours) for always-allow tool approval rules.

    Read from ``app.yaml`` ``tool_approval.rule_ttl_hours``.
    Falls back to ``_DEFAULT_TOOL_APPROVAL_TTL_HOURS`` when missing.
    """
    val = load_yaml("app.yaml").get("tool_approval", {}).get("rule_ttl_hours")
    if val is None:
        return _DEFAULT_TOOL_APPROVAL_TTL_HOURS
    return int(val)


def ingest_tool_max_retries() -> int:
    """Return the per-tool-name retry budget for ingest tools.

    Read from ``app.yaml`` ``tools.ingest.max_retries``. Passed to
    pydantic-ai's ``Tool(max_retries=N)`` so a small / local model gets
    N chances to self-correct its arg payload after the dispatcher
    raises ``ModelRetry``. Counter is keyed by tool name, not call_id,
    so concurrent calls of the same tool share the budget.

    Defaults to ``_DEFAULT_INGEST_TOOL_MAX_RETRIES`` when missing.
    """
    val = load_yaml("app.yaml").get("tools", {}).get("ingest", {}).get("max_retries")
    if val is None:
        return _DEFAULT_INGEST_TOOL_MAX_RETRIES
    return int(val)


def ask_user_question_max_retries() -> int:
    """Return the retry budget for the ``ask_user_question`` tool.

    Read from ``app.yaml`` ``tools.ask_user_question.max_retries``.
    Passed to pydantic-ai's ``Tool(max_retries=N)`` so a small / local
    model gets N chances to repair a malformed question payload (too
    few options, missing field, header > 12 chars) before pydantic-ai
    raises ``UnexpectedModelBehavior``.

    Separate from ``ingest_tool_max_retries`` because the failure modes
    differ — ingest tools fail on stringified lists and field-name
    typos, ``ask_user_question`` more often fails on schema shape
    (1 option instead of ≥2, label length). Defaults to
    ``_DEFAULT_ASK_USER_QUESTION_MAX_RETRIES`` when missing.
    """
    val = (
        load_yaml("app.yaml")
        .get("tools", {})
        .get("ask_user_question", {})
        .get("max_retries")
    )
    if val is None:
        return _DEFAULT_ASK_USER_QUESTION_MAX_RETRIES
    return int(val)


def reload_configs() -> None:
    """Invalidate the YAML cache. Test helper / admin hot-reload entry."""
    load_yaml.cache_clear()


if False:  # pragma: no cover — TYPE_CHECKING-only forward ref
    from claritymed.core.schemas.evals import EvalsConfig  # noqa: F401
    from claritymed.core.schemas.router import RouterConfig  # noqa: F401
