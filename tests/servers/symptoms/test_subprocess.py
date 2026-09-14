"""End-to-end smoke test that ``claritymed-symptoms-server`` boots over TCP.

The unit tests in ``test_app.py`` run the FastAPI app via the in-process ASGI
transport — fast, but they bypass uvicorn entirely, so a broken script
entry point, port-binding bug, or import-time crash would never surface.
This test spawns the real CLI script in a subprocess, waits for /health to
respond, asserts the response shape, then terminates the process.

``CLARITYMED_SYMPTOMS_SKIP_LOAD=1`` keeps the lifespan from trying to load
real model weights — the boot path itself is what we care about. Anything
weights-related is covered by ``tests/ingest/symptoms`` and the ASGI-level
tests.

The test is opt-in via ``CLARITYMED_RUN_SUBPROCESS_E2E=1``. CI environments
without the ``symptoms-server`` extra installed (or behind sandbox network
restrictions that block ``127.0.0.1`` sockets) skip it cleanly.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time

import httpx
import pytest

HEALTH_POLL_INTERVAL_S = 0.1
HEALTH_POLL_TIMEOUT_S = 20.0
SHUTDOWN_GRACE_S = 5.0


def _free_port() -> int:
    """Bind, read the OS-assigned port, release. Standard cooperative pattern.

    There is a small race between releasing the socket here and uvicorn
    re-binding it — fine for a single-test environment, would matter for
    parallel test runs. xdist would need a per-worker port range instead.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(url: str, deadline: float) -> httpx.Response:
    """Poll ``url`` until 200 or deadline. Last attempt's response/error is reported."""
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return r
            last_exc = AssertionError(f"unexpected status {r.status_code}: {r.text!r}")
        except httpx.HTTPError as exc:
            last_exc = exc
        time.sleep(HEALTH_POLL_INTERVAL_S)
    raise AssertionError(
        f"symptoms-server /health did not respond within "
        f"{HEALTH_POLL_TIMEOUT_S}s: {last_exc!r}"
    )


@pytest.mark.skipif(
    os.environ.get("CLARITYMED_RUN_SUBPROCESS_E2E") != "1",
    reason=(
        "subprocess smoke test is opt-in via CLARITYMED_RUN_SUBPROCESS_E2E=1 "
        "to keep the unit-test sweep socket-free"
    ),
)
def test_symptoms_server_boots_and_serves_health_over_tcp() -> None:
    """Spawn ``claritymed-symptoms-server`` and confirm /health speaks JSON.

    Skips when the script isn't on PATH — the ``symptoms-server`` extra is
    optional, and CI runs without it shouldn't fail.
    """
    script = shutil.which("claritymed-symptoms-server")
    if script is None:
        pytest.skip(
            "claritymed-symptoms-server not on PATH; "
            "install with `uv sync --extra symptoms-server`"
        )

    port = _free_port()
    env = {
        **os.environ,
        "CLARITYMED_SYMPTOMS_PORT": str(port),
        "CLARITYMED_SYMPTOMS_SKIP_LOAD": "1",
    }
    proc = subprocess.Popen(
        [script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        response = _wait_for_health(
            f"http://127.0.0.1:{port}/health",
            deadline=time.monotonic() + HEALTH_POLL_TIMEOUT_S,
        )
        body = response.json()
        # Skip-load mode → no datasets/models, status reports "loading".
        assert body["datasets_loaded"] == []
        assert body["models_loaded"] == []
        assert body["status"] in {"ok", "loading"}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=SHUTDOWN_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=SHUTDOWN_GRACE_S)
