"""Admin audit-log viewer — ``GET /api/v1/admin/audit``.

Reads ``LOG_DIR/audit.log`` (one JSON event per line), filters in
memory, and returns an offset-paginated page. The implementation is
naive on purpose — at admin frequency and current log volume this
is well below 100ms per request. We revisit if the file grows past
~100MB or if we need streaming results.

Filter knobs (all optional, all combinable):

* ``kind`` — comma-separated list of ``AuditKind`` values.
* ``actor`` — exact match on ``user_id``.
* ``since`` / ``until`` — ISO-8601; either side may be missing.

Response shape: ``{items, total_count, offset, limit}`` plus an
``X-Total-Count`` header so the SPA can render page controls without
parsing the body twice.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Response

from claritymed import config as _cfg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["admin", "audit"])

AUDIT_LOG_FILENAME = "audit.log"
DEFAULT_LIMIT = 50
MAX_LIMIT = 500


def _audit_log_path():
    return _cfg.LOG_DIR / AUDIT_LOG_FILENAME


def _parse_iso(value: str | None, field: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"invalid ISO-8601 for {field}: {value!r}"
        ) from exc


def _load_events() -> list[dict[str, Any]]:
    """Return parsed events in chronological order. Empty when log missing.

    Each line carries the stdlib ``logging`` formatter preamble before
    the JSON payload (``AUDIT_FMT`` in
    ``core/observability/logging.py`` puts ``%(message)s`` last). The
    JSON event always starts with the first ``{`` on the line — slice
    from there and parse the rest. A line without ``{`` is preamble
    only (e.g. a partially-rotated rollover marker) and is skipped.
    """
    path = _audit_log_path()
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        brace = line.find("{")
        if brace < 0:
            continue
        try:
            out.append(json.loads(line[brace:]))
        except json.JSONDecodeError:
            # Partial write mid-rotation can leave a half-line. Skip
            # silently — surfacing the malformed event would be more
            # confusing than dropping one byte.
            continue
    return out


def _match(event: dict[str, Any], filters: dict[str, Any]) -> bool:
    if filters["kinds"]:
        if event.get("kind") not in filters["kinds"]:
            return False
    if filters["actor"] and event.get("user_id") != filters["actor"]:
        return False
    if filters["request_id"] and event.get("request_id") != filters["request_id"]:
        return False
    if filters["since"] or filters["until"]:
        raw_ts = event.get("created_at")
        if not raw_ts:
            return False
        try:
            ts = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        except ValueError:
            return False
        if filters["since"] and ts < filters["since"]:
            return False
        if filters["until"] and ts > filters["until"]:
            return False
    return True


@router.get("")
async def list_audit(
    response: Response,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    kind: str | None = Query(default=None, description="comma-separated kinds"),
    actor: str | None = Query(default=None),
    request_id: str | None = Query(default=None, description="exact request_id"),
    since: str | None = Query(default=None, description="ISO-8601 timestamp"),
    until: str | None = Query(default=None, description="ISO-8601 timestamp"),
) -> dict[str, Any]:
    """Return a paginated slice of the audit log, newest-first.

    Filtering is applied before pagination so ``total_count`` reflects
    the post-filter set (matching the SPA's expectation that the
    pagination controls are sized by what the user is filtering on).

    ``distinct_actors`` lists every ``user_id`` seen across the whole
    log (not just the current page) so the SPA can render a stable
    dropdown without paging back through history.
    """
    kinds = {k.strip() for k in kind.split(",") if k.strip()} if kind else set()
    filters = {
        "kinds": kinds,
        "actor": actor,
        "request_id": request_id,
        "since": _parse_iso(since, "since"),
        "until": _parse_iso(until, "until"),
    }
    events = _load_events()
    actors_seen = {e["user_id"] for e in events if isinstance(e.get("user_id"), str)}
    matched = [e for e in events if _match(e, filters)]
    # Newest first.
    matched.reverse()
    total = len(matched)
    page = matched[offset : offset + limit]
    response.headers["X-Total-Count"] = str(total)
    return {
        "items": page,
        "total_count": total,
        "offset": offset,
        "limit": limit,
        "distinct_actors": sorted(actors_seen),
    }
