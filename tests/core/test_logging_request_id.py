"""Tests for ``claritymed.core.observability.logging``."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

from claritymed.context import (
    apply_context,
    language_ctx,
    request_id_ctx,
    reset_context,
    user_id_ctx,
)
from claritymed.core.observability.logging import (
    APP_LOGGER,
    get_access_logger,
    get_audit_logger,
    setup_logging,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_app_logger_writes_with_relpath(tmp_path):
    setup_logging("test", console_level=None)
    logger = logging.getLogger(APP_LOGGER)
    logger.info("hello world")
    log_text = _read(tmp_path / "logs" / "app.log")
    assert "hello world" in log_text


def test_request_id_in_app_log(tmp_path):
    setup_logging("test", console_level=None)
    tokens = apply_context("20260606222522A1B2C3D4", "alice", "zh")
    try:
        logging.getLogger(APP_LOGGER).info("with context")
    finally:
        reset_context(tokens)
    text = _read(tmp_path / "logs" / "app.log")
    assert "20260606222522A1B2C3D4" in text
    assert "alice" in text
    # language only shows up in audit format, not app — check just the labels.
    assert "[20260606222522A1B2C3D4][alice]" in text


def test_missing_context_renders_dashes(tmp_path):
    setup_logging("test", console_level=None)
    logging.getLogger(APP_LOGGER).info("no ctx")
    text = _read(tmp_path / "logs" / "app.log")
    assert "[-][-]" in text


def test_setup_logging_is_idempotent(tmp_path):
    setup_logging("first", console_level=None)
    setup_logging("second", console_level=None)
    logging.getLogger(APP_LOGGER).info("once")
    text = _read(tmp_path / "logs" / "app.log")
    assert text.count("once") == 1


def test_three_loggers_do_not_propagate(tmp_path):
    from claritymed.core.observability.logging import install_test_file_handlers

    setup_logging("test", console_level=None)
    # Simulate production isolation: propagate=False so audit/access records
    # stay in their own files and do not bleed into app.log.
    install_test_file_handlers(tmp_path / "logs", propagate=False)
    audit = get_audit_logger()
    access = get_access_logger()
    audit.info("audit-only")
    access.info("access-only")
    app_text = _read(tmp_path / "logs" / "app.log")
    audit_text = _read(tmp_path / "logs" / "audit.log")
    access_text = _read(tmp_path / "logs" / "access.log")
    assert "audit-only" in audit_text
    assert "access-only" in access_text
    assert "audit-only" not in app_text
    assert "access-only" not in app_text
    assert "audit-only" not in access_text


def test_audit_logger_includes_language(tmp_path):
    from claritymed.core.observability.logging import install_test_file_handlers

    # Use file-based assertion: ClarityMedFormatter injects [zh] and request_id
    # at format time; caplog stores raw records and never calls the formatter.
    install_test_file_handlers(tmp_path / "logs", propagate=False)
    tokens = apply_context("20260606222522DEADBEEF", "alice", "zh")
    try:
        get_audit_logger().info("trace")
    finally:
        reset_context(tokens)
    text = _read(tmp_path / "logs" / "audit.log")
    assert "[zh]" in text
    assert "20260606222522DEADBEEF" in text


def test_context_vars_isolated_across_concurrent_tasks(tmp_path):
    """Async tasks must each see their own request_id (ContextVar copy semantics)."""
    setup_logging("test", console_level=None)
    logger = logging.getLogger(APP_LOGGER)

    async def emit(rid: str, uid: str) -> None:
        request_id_ctx.set(rid)
        user_id_ctx.set(uid)
        language_ctx.set("en")
        await asyncio.sleep(0.01)
        logger.info("from-%s", uid)

    async def main() -> None:
        await asyncio.gather(
            *(emit(f"2026060622252200000{i:03X}", f"user{i}") for i in range(5))
        )

    asyncio.run(main())
    text = _read(tmp_path / "logs" / "app.log")
    for i in range(5):
        marker = f"[2026060622252200000{i:03X}][user{i}]"
        assert marker in text, f"missing isolation marker for user{i}"


async def test_middleware_assigns_request_id_and_resets_context(caplog):
    """Integration: starlette middleware + httpx AsyncClient pair."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from claritymed.core.observability.middleware import ContextMiddleware

    # No setup_logging() — it sets claritymed.propagate=False, breaking caplog.
    # _isolate_runtime already set propagate=True on all claritymed.* loggers.

    async def endpoint(request):
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/", endpoint)])
    app.add_middleware(ContextMiddleware)

    transport = httpx.ASGITransport(app=app)
    with caplog.at_level(logging.INFO, logger="claritymed.audit"):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as client:
            # Forged id is rejected and replaced.
            r1 = await client.get("/", headers={"X-Request-ID": "deadbeef"})
            assert r1.status_code == 200
            replaced = r1.headers["X-Request-ID"]
            assert len(replaced) == 22

            # Valid id is echoed.
            r2 = await client.get(
                "/", headers={"X-Request-ID": "20260606222522DEADBEEF"}
            )
            assert r2.headers["X-Request-ID"] == "20260606222522DEADBEEF"

    audit_msgs = " ".join(
        r.getMessage() for r in caplog.records if r.name == "claritymed.audit"
    )
    assert '"x_request_id_rejected":true' in audit_msgs
    # ContextVar reset after the request.
    assert request_id_ctx.get() is None


def test_new_request_id_format_and_uniqueness():
    """The format buys time-ordered sortability *across seconds*; within a
    second the suffix is random by design. Verify uniqueness + that timestamp
    prefixes are non-decreasing, which is what audit-log search relies on."""
    from claritymed.context import new_request_id

    ids = [new_request_id() for _ in range(1000)]
    assert len(set(ids)) == 1000  # uniqueness
    prefixes = [rid[:14] for rid in ids]
    assert prefixes == sorted(prefixes)  # monotonic timestamps
