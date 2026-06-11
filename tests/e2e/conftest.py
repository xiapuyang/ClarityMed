"""E2E test fixtures — requires live local services.

Run with:
    uv run pytest tests/e2e -v --no-cov

Services required:
    claritymed-embedder  (port 8082)
    claritymed-reranker  (port 8083)
    Qdrant               (url from configs/retrieval.yaml)
    LLM server           (omlx on :8000 or ollama on :11434)
"""

from __future__ import annotations

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


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "local: tests that require live local services; excluded from CI",
    )


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


@pytest.fixture(scope="session")
def e2e_provider_id() -> str:
    """Return the first usable LLM provider id; skip the test if none found.

    Delegates to ``stores.models.pick_reachable_provider`` so production
    code (e.g. ``claritymed eval``) and this fixture agree on what
    "reachable" means — without one drifting from the other.
    """
    from claritymed.stores.models import pick_reachable_provider

    provider = pick_reachable_provider()
    if provider is not None:
        return provider.id

    pytest.skip(
        "No LLM provider reachable with valid credentials.\n"
        "  Start a server (omlx :8000 or ollama :11434) "
        "and set any required API key env vars."
    )
