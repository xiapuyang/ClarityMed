"""Server-side request/response logging middleware.

Covers :func:`claritymed.servers._devices.add_logging_middleware`:

* both arrival (``→``) and departure (``←``) lines fire for non-health
  POST requests,
* ``log_body=True`` (the default) includes the request body and the
  response body in the log line,
* the request body re-attach lets the downstream handler still read
  ``request.json()`` after the middleware drained it,
* the response body re-emit lets the client still see the original
  payload after the middleware drained it,
* bodies that exceed ``max_body_chars`` are truncated with a count
  suffix (a base64 image won't dump in full),
* non-UTF-8 bytes are summarised as ``<binary N bytes>``,
* ``log_body=False`` falls back to the lean (path + status + elapsed)
  log lines.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from claritymed.servers._devices import add_logging_middleware


def _build_app(logger: logging.Logger, **kwargs: Any) -> FastAPI:
    """Construct a FastAPI app with the middleware + a body-echo route.

    The ``/echo`` route reads the request body and returns it inside a
    JSON envelope — used to verify that the middleware's body
    re-attach didn't break the handler's body read.
    """
    app = FastAPI()
    add_logging_middleware(app, server_logger=logger, **kwargs)

    @app.post("/echo")
    async def echo(req: Request) -> dict[str, Any]:
        payload = await req.json()
        return {"got": payload}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


@pytest.fixture
def captured(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return a live list that fills with INFO-level messages on this test's logger."""
    caplog.set_level(logging.INFO, logger="tests.middleware")
    return [r.message for r in caplog.records if r.name == "tests.middleware"]


def _drain(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.message for r in caplog.records if r.name == "tests.middleware"]


# --- core behaviour --------------------------------------------------------


def test_logs_arrival_and_departure_with_body(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="tests.middleware")
    logger = logging.getLogger("tests.middleware")
    app = _build_app(logger)
    client = TestClient(app)

    resp = client.post("/echo", json={"hello": "world"})
    assert resp.status_code == 200
    assert resp.json() == {"got": {"hello": "world"}}

    msgs = _drain(caplog)
    assert any("→ POST /echo" in m and 'body={"hello":"world"}' in m for m in msgs), (
        msgs
    )
    assert any(
        "← POST /echo" in m
        and "status=200" in m
        and 'body={"got":{"hello":"world"}}' in m
        for m in msgs
    ), msgs


def test_health_lines_are_suppressed(caplog: pytest.LogCaptureFixture) -> None:
    """``GET /health`` is the polling channel — we don't spam it into logs."""
    caplog.set_level(logging.INFO, logger="tests.middleware")
    logger = logging.getLogger("tests.middleware")
    app = _build_app(logger)
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    msgs = _drain(caplog)
    assert not any("/health" in m for m in msgs), msgs


def test_body_truncation_for_large_payloads(caplog: pytest.LogCaptureFixture) -> None:
    """Bodies bigger than ``max_body_chars`` are clipped with a count suffix.

    Probes the same path that protects vision-server / medical-clip
    logs from dumping a full base64 image in one line.
    """
    caplog.set_level(logging.INFO, logger="tests.middleware")
    logger = logging.getLogger("tests.middleware")
    app = _build_app(logger, max_body_chars=80)
    client = TestClient(app)

    big_value = "X" * 4000
    resp = client.post("/echo", json={"data": big_value})
    assert resp.status_code == 200
    # Echo handler returns the same payload — both request and response
    # should hit the truncation path.
    msgs = _drain(caplog)
    truncated_lines = [m for m in msgs if "more chars" in m]
    assert len(truncated_lines) == 2, msgs
    for line in truncated_lines:
        # Suffix format: ``…<+N more chars>``.
        assert "…<+" in line
        assert "more chars>" in line


def test_binary_fields_are_redacted_before_length_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``data_b64`` (and the rest of ``DEFAULT_REDACT_FIELDS``) shrink to
    a head + count regardless of value length, so the body cap covers
    the meaningful JSON fields instead of a useless base64 prefix.

    Probes the path the vision-server and medical-clip both hit on every
    ``/v1/detect`` / ``/v1/embed_image`` call. Pinning the head length
    keeps the log eyeball-friendly: a PNG or JPEG header in the first
    few chars is enough to spot bad payload shape.
    """
    caplog.set_level(logging.INFO, logger="tests.middleware")
    logger = logging.getLogger("tests.middleware")
    app = _build_app(logger)
    client = TestClient(app)

    big_b64 = "iVBORw0KGgoAAAA" * 1000  # ~15KB pretend-PNG base64
    resp = client.post(
        "/echo",
        json={
            "request_id": "20260616145546AB9C71BB",
            "image": {"sha256": "1a08", "data_b64": big_b64},
            "disease_id": "lung_cancer_chest_ct",
        },
    )
    assert resp.status_code == 200
    msgs = _drain(caplog)
    arrival = [m for m in msgs if "→ POST /echo" in m]
    assert arrival, msgs
    line = arrival[0]
    # data_b64 shrank to a 16-char head + ``…<+N more chars>`` counter.
    assert '"data_b64":"iVBORw0KGgoAAAAi…<+' in line
    assert "more chars>" in line
    # Other JSON keys survive in full — that's the whole point.
    assert "request_id" in line and "20260616145546AB9C71BB" in line
    assert "disease_id" in line and "lung_cancer_chest_ct" in line
    # And the line is well under the 2000-char default cap (no outer truncation).
    assert not line.endswith("more chars>}'") and "more chars>'" not in line[-40:]


def test_request_id_propagates_into_response_header(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A canonical ``X-Request-ID`` from the client is adopted and echoed back.

    The middleware accepts only ids matching
    :func:`~claritymed.context.is_valid_request_id` — the 22-char
    ``YYYYMMDDhhmmss<8 uppercase hex>`` shape the orchestrator uses.
    """
    caplog.set_level(logging.INFO, logger="tests.middleware")
    logger = logging.getLogger("tests.middleware")
    app = _build_app(logger)
    client = TestClient(app)

    rid = "20260615120000ABCDEF01"  # canonical 22-char shape
    resp = client.post("/echo", json={}, headers={"X-Request-ID": rid})
    assert resp.headers["X-Request-ID"] == rid
    msgs = _drain(caplog)
    assert any(f"req_id={rid}" in m for m in msgs), msgs


def test_malformed_request_id_is_replaced_with_a_fresh_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A forged or malformed ``X-Request-ID`` is rejected and replaced.

    Mirrors the orchestrator's API middleware policy — a header that
    fails ``is_valid_request_id`` cannot be allowed to poison the
    audit trail. The middleware generates a fresh 22-char id matching
    the standard shape so all downstream correlation works.
    """
    import re

    caplog.set_level(logging.INFO, logger="tests.middleware")
    logger = logging.getLogger("tests.middleware")
    app = _build_app(logger)
    client = TestClient(app)

    resp = client.post("/echo", json={}, headers={"X-Request-ID": "not-a-valid-id"})
    echoed = resp.headers["X-Request-ID"]
    assert echoed != "not-a-valid-id"
    # The replacement matches the canonical 22-char shape.
    assert re.match(r"^[0-9]{14}[0-9A-F]{8}$", echoed), echoed


def test_request_id_generated_when_header_absent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No ``X-Request-ID`` header → middleware generates a 22-char id.

    Same shape as the orchestrator's :func:`new_request_id` so the
    fallback id is indistinguishable from a header-supplied one in
    downstream logs.
    """
    import re

    caplog.set_level(logging.INFO, logger="tests.middleware")
    logger = logging.getLogger("tests.middleware")
    app = _build_app(logger)
    client = TestClient(app)

    resp = client.post("/echo", json={})
    echoed = resp.headers["X-Request-ID"]
    assert re.match(r"^[0-9]{14}[0-9A-F]{8}$", echoed), echoed
    msgs = _drain(caplog)
    assert any(f"req_id={echoed}" in m for m in msgs), msgs


def test_binary_body_summarised_as_byte_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-UTF-8 bodies don't get rendered raw — we log the byte count.

    This is the bytes-shape vision-server's ``image/png`` uploads
    would hit if they ever stopped being JSON-wrapped.
    """
    caplog.set_level(logging.INFO, logger="tests.middleware")
    logger = logging.getLogger("tests.middleware")
    app = FastAPI()
    add_logging_middleware(app, server_logger=logger)

    @app.post("/raw")
    async def raw(req: Request) -> dict[str, int]:
        body = await req.body()
        return {"got_bytes": len(body)}

    client = TestClient(app)
    # 0xff80 is invalid UTF-8 → triggers the binary-summary branch.
    payload = b"\xff\x80\x00binary-image-bytes-here\xfe"
    resp = client.post("/raw", content=payload)
    assert resp.status_code == 200
    msgs = _drain(caplog)
    arrival_lines = [m for m in msgs if "→ POST /raw" in m]
    assert arrival_lines, msgs
    assert "<binary " in arrival_lines[0]
    assert "bytes>" in arrival_lines[0]


# --- log_body=False fallback ----------------------------------------------


def test_log_body_false_falls_back_to_lean_lines(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When body logging is off, the original log shape (no body=) is used."""
    caplog.set_level(logging.INFO, logger="tests.middleware")
    logger = logging.getLogger("tests.middleware")
    app = _build_app(logger, log_body=False)
    client = TestClient(app)

    resp = client.post("/echo", json={"hello": "world"})
    assert resp.status_code == 200

    msgs = _drain(caplog)
    arrival_lines = [m for m in msgs if "→ POST /echo" in m]
    departure_lines = [m for m in msgs if "← POST /echo" in m]
    assert arrival_lines and "body=" not in arrival_lines[0], arrival_lines
    assert departure_lines and "body=" not in departure_lines[0], departure_lines


# --- isolation: handler can still read the body ---------------------------


def test_handler_can_still_read_body_after_middleware_drain(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression guard: middleware's body re-attach must replay the bytes.

    BaseHTTPMiddleware drains the receive() channel; if we forget to
    rewire it, downstream ``await request.json()`` blocks forever or
    yields empty bytes. The ``/echo`` route reads ``await req.json()``
    and returns the payload verbatim — non-trivial round-trip proves
    the re-attach worked.
    """
    caplog.set_level(logging.INFO, logger="tests.middleware")
    logger = logging.getLogger("tests.middleware")
    app = _build_app(logger)
    client = TestClient(app)

    payload = {"a": 1, "nested": [1, 2, 3], "flag": True}
    resp = client.post("/echo", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"got": payload}
