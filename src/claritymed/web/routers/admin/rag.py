"""Admin RAG-corpus endpoints.

Manages system-level Qdrant collections (the ones declared in
``retrieval.yaml`` under ``system_rag.collections``). Per-user RAG
stays under the user surface.

Five endpoints:

* ``GET /admin/rag/collections`` — list system corpora + chunk counts.
* ``GET /admin/rag/collections/{name}`` — metadata for one corpus.
* ``DELETE /admin/rag/collections/{name}`` — drop a Qdrant collection
  (and emit audit). Chunks are gone after this.
* ``POST /admin/rag/ingest`` — kick off an ingest Job for one file.
* ``POST /admin/rag/bootstrap`` — kick off a bootstrap Job (ensures
  every declared collection exists).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from claritymed import config as _cfg
from claritymed.core.observability.audit import audit_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["admin", "rag"])


class RagIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_name: str
    file_path: str
    topic: str | None = None
    language: str = "en"
    authority_tier: int = 3
    doc_id: str | None = None


class RagBootstrapRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skip_existing: bool = True


def _load_system_entries() -> list[dict[str, Any]]:
    raw = _cfg.load_yaml("retrieval.yaml")
    return list((raw.get("system_rag") or {}).get("collections") or [])


def _chunk_count(name: str) -> int | None:
    """Best-effort chunk count via ``RagCollectionStore``. ``None`` on error."""
    try:
        from claritymed.stores.knowledge import RagCollectionStore

        store = RagCollectionStore(name)
        return store.count_chunks()
    except Exception:  # noqa: BLE001
        return None


@router.get("/collections")
async def list_collections() -> dict[str, Any]:
    entries = _load_system_entries()
    items: list[dict[str, Any]] = []
    for entry in entries:
        name = entry.get("name")
        if not name:
            continue
        items.append(
            {
                "name": name,
                "language": entry.get("language", "en"),
                "authority_tier": int(entry.get("authority_tier", 3)),
                "topics": list(entry.get("topics") or []),
                "license": entry.get("license"),
                "chunk_count": _chunk_count(name),
            }
        )
    return {"items": items, "total_count": len(items)}


@router.get("/collections/{name}")
async def inspect_collection(name: str) -> dict[str, Any]:
    entry = next(
        (e for e in _load_system_entries() if e.get("name") == name),
        None,
    )
    if entry is None:
        raise HTTPException(status_code=404, detail=f"collection {name!r} not declared")
    audit_event("admin.rag.collection.read", payload={"name": name})
    return {
        "name": name,
        "metadata": entry,
        "chunk_count": _chunk_count(name),
    }


@router.delete("/collections/{name}", status_code=204)
async def delete_collection(name: str) -> None:
    entry = next(
        (e for e in _load_system_entries() if e.get("name") == name),
        None,
    )
    if entry is None:
        raise HTTPException(status_code=404, detail=f"collection {name!r} not declared")
    try:
        from claritymed.stores.knowledge import RagCollectionStore

        store = RagCollectionStore(name)
        store.delete_collection()
    except AttributeError as exc:
        # Store doesn't expose delete; fail clearly rather than swallow.
        raise HTTPException(
            status_code=501,
            detail="RagCollectionStore.delete_collection not implemented",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"delete failed: {exc!r}") from exc
    audit_event("admin.rag.collection.delete", payload={"name": name})


@router.post("/ingest")
async def trigger_ingest(req: RagIngestRequest, request: Request) -> dict[str, Any]:
    registry = request.app.state.jobs
    if not registry.has_runner("rag_ingest"):
        raise HTTPException(status_code=503, detail="rag_ingest runner not registered")
    spec = await registry.submit("rag_ingest", req.model_dump(mode="python"))
    return spec.to_json()


@router.post("/bootstrap")
async def trigger_bootstrap(
    req: RagBootstrapRequest, request: Request
) -> dict[str, Any]:
    registry = request.app.state.jobs
    if not registry.has_runner("rag_bootstrap"):
        raise HTTPException(
            status_code=503, detail="rag_bootstrap runner not registered"
        )
    spec = await registry.submit("rag_bootstrap", req.model_dump(mode="python"))
    audit_event("admin.rag.bootstrap", payload={"job_id": spec.id})
    return spec.to_json()
