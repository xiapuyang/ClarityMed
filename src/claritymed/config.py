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


def supported_langs() -> tuple[str, ...]:
    """Return ``app.yaml`` ``i18n.supported_langs`` or ``(DEFAULT_LANG_FALLBACK,)``."""
    raw = load_yaml("app.yaml").get("i18n", {}).get("supported_langs")
    if not raw:
        return (DEFAULT_LANG_FALLBACK,)
    return tuple(str(x).lower() for x in raw)


def load_modes_config() -> "ModesConfig":
    """Load and validate ``configs/modes.yaml``.

    Single source of truth for which modes exist, what tools they may call,
    whether LLM inference is allowed per-mode, and how the router classifies
    inputs. Wraps :func:`load_yaml` (cached) and validates with Pydantic so
    misspelled keys fail fast at load time, not at first runtime use.

    Raises:
        FileNotFoundError: If ``configs/modes.yaml`` does not exist.
        pydantic.ValidationError: If the YAML is structurally wrong.
    """
    from claritymed.core.schemas.modes import ModesConfig

    raw = load_yaml("modes.yaml")
    if not raw:
        raise FileNotFoundError(
            "configs/modes.yaml missing or empty — required for mode dispatch."
        )
    return ModesConfig.model_validate(raw)


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from ``CLARITYMED_HOME/.env`` into ``os.environ``.

    Per-user keys (provider API keys, language overrides, default user) live
    in the runtime root rather than the repo, so an open-source clone never
    ships secrets. Lines starting with ``#`` are ignored. Existing env vars
    are preserved — the file is a default, not an override.

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


def reload_configs() -> None:
    """Invalidate the YAML cache. Test helper / admin hot-reload entry."""
    load_yaml.cache_clear()


if False:  # pragma: no cover — TYPE_CHECKING-only forward ref
    from claritymed.core.schemas.modes import ModesConfig  # noqa: F401
