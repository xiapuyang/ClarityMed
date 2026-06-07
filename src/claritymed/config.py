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


def reload_configs() -> None:
    """Invalidate the YAML cache. Test helper / admin hot-reload entry."""
    load_yaml.cache_clear()
