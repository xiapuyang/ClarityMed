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
from claritymed.config import load_env_file

load_env_file()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "local: tests that require live local services; excluded from CI",
    )


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

    Checks both server reachability and credential availability.
    Local providers are checked by their health/version endpoints; credentials
    are validated via the project's is_provider_available() helper.
    """
    from claritymed.stores.models import is_provider_available, load_models

    # Endpoint that returns 200 without auth — one per local provider id.
    health_urls: dict[str, str] = {
        "omlx": "http://127.0.0.1:8000/health",
        "ollama": "http://127.0.0.1:11434/api/version",
    }

    catalog = {p.id: p for p in load_models().providers}
    for pid, url in health_urls.items():
        provider = catalog.get(pid)
        if provider is None:
            continue
        if not is_provider_available(provider):
            continue  # missing credentials
        if _service_up(url):
            return pid

    pytest.skip(
        "No LLM provider reachable with valid credentials.\n"
        "  Checked: " + ", ".join(health_urls) + "\n"
        "  Start a server and set any required API key env vars."
    )
