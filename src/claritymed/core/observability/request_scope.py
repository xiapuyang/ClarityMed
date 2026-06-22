"""``request_scope`` — paired ``request_start`` / ``request_end`` emit.

Every entry point — Typer CLI subcommand, FastAPI HTTP request, future
ag-ui websocket frame — must bracket its work with two audit events so
``audit.log`` can be reassembled into per-request traces alongside the
OTel spans. Before this helper existed, the CLI did it inline in
:func:`claritymed.cli.entry.inject_context` and web didn't do it at all.

The function is a synchronous ``@contextmanager`` because:

* ``audit_event`` and the access logger are sync; making the helper
  ``async`` would push every call site through ``asyncio.run`` for no
  gain.
* Sync ``with`` blocks can wrap ``await`` calls in async functions
  (standard Python pattern), so a Starlette middleware can do::

      with request_scope("web", method=req.method, path=req.url.path):
          response = await call_next(req)

  and still get paired emit-on-success / emit-on-exception semantics.

Payload schema (free-form per ``entry``):

* ``entry="cli"`` →  ``{"command": "ask"}``
* ``entry="web"`` →  ``{"method": "POST", "path": "/api/v1/sessions/.../stream"}``

Consumers of ``audit.log`` should treat the payload as opaque and key
off ``entry`` for routing.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.logging import get_access_logger


@contextmanager
def request_scope(entry: str, **payload: Any) -> Iterator[None]:
    """Emit paired ``request_start`` + ``request_end`` audit & access lines.

    Args:
        entry: Host identifier — ``"cli"`` / ``"web"`` / future hosts.
            Becomes the ``entry`` key in both audit events and the
            first segment of the access.log line (``cli.start`` /
            ``web.start``).
        **payload: Arbitrary metadata attached to ``request_start``.
            ``request_end`` only carries ``{"status": "ok"|"exception"}``
            so consumers can pair the two events by request_id (set by
            the caller's context wiring).

    Exceptions propagate after emitting ``request_end`` with
    ``status="exception"`` — never swallow.
    """
    access = get_access_logger()
    summary = " ".join(f"{k}={v}" for k, v in payload.items()) or "-"
    audit_event("request_start", payload={"entry": entry, **payload})
    access.info("%s.start %s", entry, summary)
    try:
        yield
    except BaseException:
        audit_event("request_end", payload={"entry": entry, "status": "exception"})
        access.info("%s.end %s status=exception", entry, summary)
        raise
    else:
        audit_event("request_end", payload={"entry": entry, "status": "ok"})
        access.info("%s.end %s status=ok", entry, summary)


__all__ = ["request_scope"]
