"""E2E test fixtures — requires live local services.

Run with:
    uv run pytest tests/e2e -v --no-cov

Services required:
    claritymed-embedder  (port 8082)
    claritymed-reranker  (port 8083)
    Qdrant               (url from configs/retrieval.yaml)
    LLM server           (omlx on :8000 or ollama on :11434)

Data directory:
    Unlike unit tests (which run under pytest's per-test ``tmp_path``),
    e2e tests redirect ``CLARITYMED_HOME`` to a fixed location under
    ``<repo>/data/.e2e_root/`` so a maintainer can inspect
    ``profile.db``, record manifests, audit log, and the chat session
    JSONL after the run. The directory is wiped once at session start
    so re-runs are deterministic, but kept between tests inside one run
    so accumulated state (allergies + medications + …) lands in the
    same place. Per-test isolation of in-memory caches (SQLAlchemy
    engines, account cache) still happens so SQLite handles don't go
    stale across tests.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import httpx
import pytest

# Load provider API keys and overrides from ~/.claritymed/.env before any
# fixture or test runs — mirrors what the CLI does in _bootstrap_once().
from claritymed import config as _cfg
from claritymed.config import load_env_file

load_env_file()

# Snapshot the session-level log root before _isolate_runtime redirects LOG_DIR
# to a per-test tmp_path.  All e2e tests write their logs here so a developer
# can inspect them after a run without hunting through temp directories.
_E2E_LOG_DIR = _cfg.LOG_DIR / "e2e"

# E2E writes into the developer's standard CLARITYMED_HOME
# (``~/.claritymed/`` by default) so the on-disk layout is identical to a
# real CLI invocation — the maintainer can ``cat
# ~/.claritymed/data/users/e2e/profile.db`` after a run finishes. Only
# the ``users/e2e/`` subtree is wiped at session start; sibling user
# dirs (the developer's personal CLI user) are left alone.
_E2E_HOME = Path.home() / ".claritymed"
_E2E_USER_DIR = _E2E_HOME / "data" / "users" / "e2e"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "local: tests that require live local services; excluded from CI",
    )


@pytest.fixture(scope="session", autouse=True)
def _wipe_e2e_user_dir() -> None:
    """Clear ``~/.claritymed/data/users/e2e/`` once at session start.

    Surgical: only the ``e2e`` user subtree is removed. Sibling user
    dirs (the developer's personal CLI user, prior test users) and
    shared resources (``shared/``, ``logs/``, prompt store) are left
    untouched. Re-runs are deterministic: a previous run's profile.db
    / manifests / chat sessions do not bleed into this run's verify
    helpers, but accumulated state within one run lines up so a
    successful trigger lands where the next test will look.
    """
    if _E2E_USER_DIR.exists():
        shutil.rmtree(_E2E_USER_DIR)
    yield


@pytest.fixture(autouse=True)
def _persistent_e2e_home(
    _isolate_runtime,
    _wipe_e2e_user_dir,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Override the project-level ``_isolate_runtime`` tmp_path redirect.

    ``_isolate_runtime`` (in ``tests/conftest.py``) sets
    ``CLARITYMED_HOME`` to a per-test tmp_path so unit tests can't
    pollute each other. For e2e we want the opposite — a fixed home
    matching the real CLI layout — so we drop the per-test override
    here and reload config. Listed as a parameter on
    ``_isolate_runtime`` ensures we run after its setup body completes.

    The wipe-once-at-session-start fixture took care of the user
    subtree; this fixture only re-resolves the config paths so stores
    target ``~/.claritymed/`` instead of the tmp path.
    """
    monkeypatch.delenv("CLARITYMED_HOME", raising=False)
    monkeypatch.delenv("CLARITYMED_DATA_DIR", raising=False)
    monkeypatch.delenv("CLARITYMED_SHARED_DIR", raising=False)
    monkeypatch.delenv("CLARITYMED_LOG_DIR", raising=False)

    import importlib

    from claritymed import config as _config

    importlib.reload(_config)
    _config.reload_configs()

    # _isolate_runtime already cleared the per-user engine + account caches
    # for the tmp_path it installed; redo for the fresh CLARITYMED_HOME so
    # the next store call resolves paths against ``~/.claritymed/``.
    from claritymed.stores import profile as _profile

    _profile._ENGINES.clear()
    from claritymed.stores.account import reset_account_cache

    reset_account_cache()
    yield


@pytest.fixture(autouse=True)
def _install_e2e_log_handlers() -> None:
    """Write e2e test logs to logs/e2e/ for post-run inspection.

    Runs after _isolate_runtime clears all handlers.  propagate=True keeps
    pytest's caplog fixture working so tests can still assert via caplog while
    the log files accumulate in _E2E_LOG_DIR for manual inspection.
    """
    from claritymed.core.observability.logging import install_test_file_handlers

    install_test_file_handlers(_E2E_LOG_DIR)


def _service_up(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _seed_terminology() -> None:
    """Write the seed concepts.jsonl into the per-test SHARED_DIR.

    ``configs/retrieval.yaml`` has ``term_service.active: umls_cmekg_local``;
    without a seeded file, ``build_hybrid_retriever()`` raises
    ``FileNotFoundError`` before any live service is reached. The seed
    script's built-in dataset is identical to what an operator would
    write with ``init_terminology.py --seed``, so e2e tests exercise the
    real production code path.
    """
    import sys
    from pathlib import Path

    from claritymed.stores.paths import shared_terminology_jsonl

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        from init_terminology import cmd_seed  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    target = shared_terminology_jsonl()
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd_seed(target, force=True, dry_run=False)


@pytest.fixture(scope="session", autouse=True)
def require_local_services() -> None:
    """Skip all e2e tests when required local services aren't reachable."""
    checks = {
        "claritymed-embedder": "http://localhost:8082/health",
        "claritymed-reranker": "http://localhost:8083/health",
    }
    down = [name for name, url in checks.items() if not _service_up(url)]
    if down:
        pytest.skip(
            f"Required local services not running: {', '.join(down)}\n"
            "  Start them with:\n"
            "    uv run --extra rag-server claritymed-embedder &\n"
            "    uv run --extra rag-server claritymed-reranker &"
        )


def _resolve_e2e_provider_ids() -> list[str]:
    """Pick the set of providers e2e should run against.

    Order:
    * ``CLARITYMED_E2E_PROVIDERS=omlx,deepseek`` (comma-separated catalog
      ids) — explicit opt-in for a multi-provider matrix run.
    * Otherwise: the single local provider ``pick_reachable_provider``
      finds (the historical default). Returns ``[]`` when nothing is
      reachable so the fixture skips cleanly.

    The list is computed once at conftest import time so pytest's
    parametrize machinery can register all params at collection time.
    """
    import os

    raw = os.environ.get("CLARITYMED_E2E_PROVIDERS", "").strip()
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    from claritymed.stores.models import pick_reachable_provider

    p = pick_reachable_provider()
    return [p.id] if p is not None else []


_E2E_PROVIDER_IDS: list[str] = _resolve_e2e_provider_ids() or ["__none__"]


@pytest.fixture(scope="session", params=_E2E_PROVIDER_IDS)
def e2e_provider_id(request: pytest.FixtureRequest) -> str:
    """Return one provider id per parametrize iteration.

    Skips when no provider is reachable / when credentials are missing
    for the requested id; this keeps a partially-configured matrix
    (``omlx`` up + ``deepseek`` key unset) from failing the whole run.
    """
    pid = request.param
    if pid == "__none__":
        pytest.skip(
            "No LLM provider reachable with valid credentials.\n"
            "  Start a server (omlx :8000 or ollama :11434) "
            "and set any required API key env vars, "
            "or set CLARITYMED_E2E_PROVIDERS=<id>[,<id>...]."
        )

    from claritymed.stores.models import is_provider_available, load_models

    catalog = {p.id: p for p in load_models().providers}
    provider = catalog.get(pid)
    if provider is None:
        pytest.skip(f"Provider {pid!r} not in configs/models.yaml catalog.")
    if not is_provider_available(provider):
        pytest.skip(f"Provider {pid!r} catalog-listed but credentials missing in env.")
    return pid
