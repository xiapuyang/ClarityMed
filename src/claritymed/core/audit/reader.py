"""Iterate audit.log* JSONL events with optional filters.

Lifted from the ``audit grep`` CLI body so rules and ad-hoc tools share
the same parser. Walks files in modification-time order to keep output
roughly chronological across rotations.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from claritymed import config as _cfg

logger = logging.getLogger(__name__)


def read_audit_events(
    *,
    since: str | None = None,
    until: str | None = None,
    user_id: str | None = None,
    log_dir: Path | None = None,
) -> Iterator[dict]:
    """Yield parsed audit events from rotated ``audit.log*`` files.

    Filters are evaluated as ISO-8601 prefix string compare (``since`` /
    ``until``) and exact match (``user_id``). Missing log dir raises
    ``FileNotFoundError`` so a misconfigured ``CLARITYMED_LOG_DIR`` fails
    loud rather than silently returning zero matches.
    """
    base = Path(log_dir) if log_dir is not None else Path(_cfg.LOG_DIR)
    if not base.exists():
        raise FileNotFoundError(f"audit log dir does not exist: {base}")
    files = sorted(base.glob("audit.log*"), key=lambda p: p.stat().st_mtime)
    for fp in files:
        try:
            fh = fp.open(encoding="utf-8")
        except OSError as exc:
            logger.warning("cannot open audit log %s: %s", fp, exc)
            continue
        with fh:
            for raw in fh:
                line = raw.rstrip("\n")
                idx = line.find("{")
                if idx < 0:
                    continue
                try:
                    event = json.loads(line[idx:])
                except json.JSONDecodeError:
                    continue
                if user_id and event.get("user_id") != user_id:
                    continue
                created = event.get("created_at", "")
                if since and created < since:
                    continue
                if until and created > until:
                    continue
                yield event
