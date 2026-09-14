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
    # Route unit-test traces to a separate Phoenix project so they don't
    # pollute the production project. e2e/conftest.py overrides to "claritymed-e2e".
    os.environ.setdefault("CLARITYMED_TRACE_PROJECT", "claritymed-pytest")
    # MLflow's default tracking URI is ``./mlruns`` (file backend) relative
    # to cwd — any test path that touches mlflow without going through
    # :func:`claritymed.ingest.mlflow_utils.mlflow_run` would create
    # ``mlflow.db`` / ``mlruns/`` in the repo root. Force a session-scoped
    # tmp tracking URI so the safety net catches such bypasses. Tests that
    # exercise ``mlflow_run`` still get per-test isolation via the
    # ``CLARITYMED_HOME=tmp_path`` monkeypatch in ``_isolate_runtime``.
    mlflow_db = root / "mlflow.db"
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{mlflow_db}"


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

    # Per-user engines and account cache are process-global; reset between
    # tests so one test's "alice" cannot leak into another's tmp tree.
    from claritymed.stores import profile as _profile

    _profile._ENGINES.clear()
    from claritymed.stores.account import reset_account_cache

    reset_account_cache()

    # Reset the three loggers between tests so audit / access handlers from a
    # previous test do not still write into the previous tmp_path's logs dir.
    # Lazy getters reconfigure on next call against the current LOG_DIR.
    import logging as _logging

    for name in ("claritymed", "claritymed.access", "claritymed.audit"):
        lg = _logging.getLogger(name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        # setup_logging sets propagate=False; reset so caplog can see records
        # in tests that did not call setup_logging themselves.
        lg.propagate = True

    yield
    _config.reload_configs()


@pytest.fixture
def alice():
    """First user — auto-promoted to admin by ``init_user``."""
    from claritymed.stores.account import init_user

    return init_user("alice", display_name="Alice")


@pytest.fixture
def bob(alice):
    """Second user — default ``user`` role. Depends on alice for ordering."""
    from claritymed.stores.account import init_user

    return init_user("bob", display_name="Bob")


@pytest.fixture
def carol(alice):
    """Third user — default ``user`` role. Depends on alice for ordering."""
    from claritymed.stores.account import init_user

    return init_user("carol", display_name="Carol")


@pytest.fixture
def trio(alice, bob, carol):
    """admin + two users, the canonical role-isolation fixture."""
    return {"admin": alice, "users": [bob, carol]}


@pytest.fixture
def as_():
    """Context manager that sets the three ContextVars for a given account.

    Usage::

        with as_(bob):
            with pytest.raises(PermissionDeniedError):
                require_admin()
    """
    from contextlib import contextmanager

    from claritymed.context import apply_context, new_request_id, reset_context

    @contextmanager
    def _switch(account_or_uid):
        uid = (
            account_or_uid.user_id
            if hasattr(account_or_uid, "user_id")
            else account_or_uid
        )
        tokens = apply_context(new_request_id(), uid, "en")
        try:
            yield
        finally:
            reset_context(tokens)

    return _switch
