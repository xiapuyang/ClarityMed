"""Session fixtures.

Critical ordering: ``CLARITYMED_*_DIR`` must be set *before* any ``claritymed``
import, because ``config.py`` evaluates ``DATA_DIR`` / ``SHARED_DIR`` /
``LOG_DIR`` once at module import. We use a ``pytest_configure`` hook so the
env vars land before test collection imports anything.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Point all runtime dirs into a per-session tmp tree."""
    root = Path(config.cache.mkdir("claritymed_runtime"))
    os.environ["CLARITYMED_HOME"] = str(root)
    os.environ["CLARITYMED_DATA_DIR"] = str(root / "data")
    os.environ["CLARITYMED_SHARED_DIR"] = str(root / "shared")
    os.environ["CLARITYMED_LOG_DIR"] = str(root / "logs")


@pytest.fixture(autouse=True)
def _isolate_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Each test gets its own runtime root (test isolation).

    Only ``CLARITYMED_HOME`` is set; ``DATA_DIR`` / ``SHARED_DIR`` / ``LOG_DIR``
    fall through to HOME children. Tests that want to exercise the specific-env
    fallback override just those vars themselves.

    ``importlib.reload(claritymed.config)`` lets ``DATA_DIR`` etc. pick up the
    monkeypatched env. Reloading the config module is cheap; the modules that
    cache paths (logging handlers, stores) re-resolve on call.
    """
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    monkeypatch.delenv("CLARITYMED_DATA_DIR", raising=False)
    monkeypatch.delenv("CLARITYMED_SHARED_DIR", raising=False)
    monkeypatch.delenv("CLARITYMED_LOG_DIR", raising=False)

    import importlib

    from claritymed import config as _config

    importlib.reload(_config)
    _config.reload_configs()
    yield
    _config.reload_configs()
