"""E2E test fixtures — requires live local services.

Run with:
    uv run pytest tests/e2e -v --no-cov

Services required:
    claritymed-embedder  (port 8082)
    claritymed-reranker  (port 8083)
    Qdrant               (url from configs/retrieval.yaml)
"""

from __future__ import annotations

import httpx
import pytest


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
