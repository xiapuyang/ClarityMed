"""Bootstrap runner — ensures every system collection in
``configs/retrieval.yaml`` exists as an empty Qdrant collection.

This mirrors ``scripts/init_system_rag.py`` only insofar as both touch
:class:`RagCollectionStore.ensure_collection`. The CLI script also
ingests caller-supplied documents; this admin job leaves the seed-docs
choice to the operator (the SPA's ingest flow handles that).

Idempotent — running twice is a no-op. The runner reports progress per
collection so the Jobs UI shows which name is being processed at any
moment.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from claritymed import config as _cfg

if TYPE_CHECKING:
    from claritymed.web.admin.jobs import JobRegistry, JobSpec

logger = logging.getLogger(__name__)


async def run(spec: "JobSpec", registry: "JobRegistry") -> None:
    """Walk ``system_rag.collections`` and ensure each exists.

    ``skip_existing`` (default True) leaves a populated collection
    alone. False would imply re-creating it from scratch, which would
    destroy the chunks the operator may have already loaded — we
    refuse that path for safety in v1.
    """
    skip_existing = bool(spec.params.get("skip_existing", True))
    if not skip_existing:
        raise RuntimeError(
            "skip_existing=False is not supported — re-creating a "
            "populated system_rag collection would drop existing chunks. "
            "Delete the collection explicitly via DELETE /admin/rag/"
            "collections/{name} first if that's what you want."
        )
    raw = _cfg.load_yaml("retrieval.yaml")
    entries = (raw.get("system_rag") or {}).get("collections") or []
    if not entries:
        registry.update(spec.id, progress="no system_rag.collections configured")
        return

    try:
        from claritymed.stores.knowledge import RagCollectionStore
    except ImportError:
        # Defensive — if the store moves we'd rather see this than a
        # crash mid-bootstrap.
        registry.update(
            spec.id,
            progress="RagCollectionStore unavailable; skipping",
        )
        return

    total = len(entries)
    for i, entry in enumerate(entries):
        name = entry.get("name") if isinstance(entry, dict) else None
        if not name:
            continue
        registry.update(spec.id, progress=f"{i + 1}/{total}: {name}")
        try:
            store = RagCollectionStore(name)
            store.ensure_collection()
            registry.append_stdout(spec.id, f"ensured: {name}")
        except Exception as exc:  # noqa: BLE001
            registry.append_stdout(spec.id, f"failed to ensure {name}: {exc!r}")
            logger.warning("rag bootstrap: %s failed: %s", name, exc)

    registry.update(spec.id, progress=f"done ({total} collections)")
