"""Owner-only side-channel for PHI text that audit.log can't carry.

``audit.log`` is the structured one-line-per-event ledger; it keeps
non-PHI fields (``record_path``, ``sha256``, ``tool_name``, ``decision``)
so grep + tail are safe to run by an operator. The PHI-bearing fields
of a tool call (``title``, ``notes``, ``extracted_labs.value``) need
audit too — but they cannot share a file with the structured rows
without violating the no-raw-PHI contract.

This module writes each request's PHI payload to
``data/users/<id>/audit_payloads/<request_id>.json`` with mode ``0o600``
(owner read/write only). The two ledgers correlate by ``request_id``,
which both records carry.

Atomicity + permissions are enforced via ``os.open`` with
``O_CREAT | O_EXCL`` and mode ``0o600`` — closes the
write-then-chmod race where another process could read between
``open`` and ``chmod``.
"""

from __future__ import annotations

import json
import logging
import os

from claritymed.stores.paths import (
    user_audit_payload_path,
    user_audit_payloads_dir,
)

logger = logging.getLogger(__name__)


def write_payload(user_id: str, request_id: str, payload: dict) -> None:
    """Write ``payload`` JSON to the user's audit_payloads dir.

    Idempotent: if ``<request_id>.json`` already exists (because a retry
    or a concurrent write got there first), the function returns
    silently. The contract is "one PHI payload per request_id"; the
    caller picks the right request_id.
    """
    payloads_dir = user_audit_payloads_dir(user_id)
    payloads_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Best-effort tighten existing dir; mkdir's ``mode`` is umask-masked.
    try:
        os.chmod(payloads_dir, 0o700)
    except OSError:  # pragma: no cover - filesystem-dependent
        logger.warning("could not chmod audit_payloads dir", exc_info=True)

    path = user_audit_payload_path(user_id, request_id)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(str(path), flags, 0o600)
    except FileExistsError:
        # The first writer's payload stands; we don't overwrite history.
        logger.info(
            "audit payload already exists for request_id=%s; skipping",
            request_id,
        )
        return
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception:
        # Best effort: remove the partial file so a follow-up retry can
        # produce a clean payload.
        try:
            os.remove(path)
        except OSError:  # pragma: no cover
            pass
        raise
