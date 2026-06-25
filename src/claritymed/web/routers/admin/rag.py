"""Admin RAG-corpus endpoints.

Manages system-level Qdrant collections (the ones declared in
``retrieval.yaml`` under ``system_rag.collections``). Per-user RAG
stays under the user surface.

Endpoints:

* ``GET /admin/rag/collections`` — list system corpora + chunk counts.
* ``GET /admin/rag/collections/{name}`` — metadata for one corpus.
* ``DELETE /admin/rag/collections/{name}`` — drop a Qdrant collection
  (and emit audit). Chunks are gone after this.
* ``POST /admin/rag/collections/upsert`` — multipart upload that
  creates a new system corpus (or appends files to an existing one),
  routed through the same OCR/chunk/embed/dedup pipeline as the
  ``scripts/init_system_rag.py`` CLI.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict

from qdrant_client import AsyncQdrantClient

from claritymed import config as _cfg
from claritymed.core.observability.audit import audit_event
from claritymed.core.rag.qdrant_store import build_qdrant_client
from claritymed.stores.paths import jobs_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["admin", "rag"])

# Upload caps. Match the per-user attachment endpoint for individual file
# size (25 MiB) but allow more files per request since corpus ingest
# legitimately uploads dozens of guideline PDFs at once.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_MAX_UPLOAD_FILES = 100
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class RagUpsertMetadata(BaseModel):
    """JSON metadata field that accompanies the multipart upload.

    Optional fields (``language``, ``authority_tier``, ``topics``,
    ``license``, ``cross_lingual``) are inherited from ``retrieval.yaml``
    when ``name`` matches an existing collection. New collections get
    historical defaults (``en`` / tier 2 / empty topics).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    topics: list[str] = []
    language: str | None = None
    cross_lingual: bool = False
    authority_tier: int | None = None
    license: str | None = None
    dedupe_cosine_threshold: float = 0.0


def _load_system_entries() -> list[dict[str, Any]]:
    raw = _cfg.load_yaml("retrieval.yaml")
    return list((raw.get("system_rag") or {}).get("collections") or [])


def _open_qdrant_aclient() -> AsyncQdrantClient | None:
    """Open an async Qdrant client from raw yaml ``qdrant.*`` fields.

    Bypasses :func:`load_retrieval_config` so admin endpoints stay
    operable even when unrelated retrieval sub-sections are malformed
    (e.g. a stripped-down test fixture). Honors the
    ``CLARITYMED_QDRANT_URL`` env override that
    :func:`load_retrieval_config` would otherwise apply. Returns
    ``None`` when ``qdrant.url`` is unconfigured so the caller can
    still render metadata with chunk_count=None instead of 500-ing.
    """
    import os

    raw = _cfg.load_yaml("retrieval.yaml")
    qcfg = raw.get("qdrant") or {}
    url = os.environ.get("CLARITYMED_QDRANT_URL") or qcfg.get("url")
    if not url:
        return None
    return build_qdrant_client(url=url, api_key_env=qcfg.get("api_key_env"))


async def _chunk_count(aclient: AsyncQdrantClient, name: str) -> int | None:
    """Live chunk count for one collection. ``None`` when probe fails.

    Cheap path: skip the count call entirely when the collection isn't
    declared in Qdrant yet (fresh install pre-bootstrap), so the SPA can
    distinguish "not created" (``—``) from "created but empty" (``0``).
    """
    try:
        if not await aclient.collection_exists(name):
            return None
        info = await aclient.count(name, exact=True)
        return int(info.count)
    except Exception:  # noqa: BLE001 — single-collection failures shouldn't blank the list
        return None


@router.get("/collections")
async def list_collections() -> dict[str, Any]:
    entries = _load_system_entries()
    aclient = _open_qdrant_aclient()
    try:
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
                    "chunk_count": (
                        await _chunk_count(aclient, name) if aclient else None
                    ),
                }
            )
    finally:
        if aclient is not None:
            await aclient.close()
    return {"items": items, "total_count": len(items)}


@router.get("/collections/{name}")
async def inspect_collection(name: str) -> dict[str, Any]:
    entry = next(
        (e for e in _load_system_entries() if e.get("name") == name),
        None,
    )
    if entry is None:
        raise HTTPException(status_code=404, detail=f"collection {name!r} not declared")
    aclient = _open_qdrant_aclient()
    try:
        count = await _chunk_count(aclient, name) if aclient else None
    finally:
        if aclient is not None:
            await aclient.close()
    audit_event("admin.rag.collection.read", payload={"name": name})
    return {
        "name": name,
        "metadata": entry,
        "chunk_count": count,
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


def _parse_metadata(raw: str) -> RagUpsertMetadata:
    """Validate the JSON metadata blob from the multipart form.

    Wrapping in two error shapes lets the SPA distinguish a missing
    field (422) from invalid JSON (also 422) without inspecting the
    detail string.
    """
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"metadata is not valid JSON: {exc.msg}",
        ) from exc
    try:
        return RagUpsertMetadata.model_validate(body)
    except Exception as exc:  # noqa: BLE001 — pydantic raises a few flavours
        raise HTTPException(status_code=422, detail=f"invalid metadata: {exc}") from exc


def _save_uploads(files: list[UploadFile], target_dir: Path) -> list[Path]:
    """Persist each upload under ``target_dir`` and return their paths.

    Bytes are buffered in memory because FastAPI's UploadFile is a
    SpooledTemporaryFile under the hood — by the time the runner picks
    up the job, the request has already returned, so we must commit to
    a stable on-disk path before then. Per-file cap enforces an upper
    bound on that buffer.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for upload in files:
        if not upload.filename:
            raise HTTPException(status_code=422, detail="file part missing filename")
        data = upload.file.read()
        if not data:
            raise HTTPException(
                status_code=422, detail=f"empty file: {upload.filename!r}"
            )
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"{upload.filename!r} is {len(data)} bytes; cap is "
                    f"{_MAX_UPLOAD_BYTES} bytes"
                ),
            )
        # Strip directory components from the upload-supplied name —
        # multipart filenames are attacker-controlled.
        safe_name = Path(upload.filename).name or "upload.bin"
        dest = target_dir / safe_name
        # Disambiguate collisions within one upload batch.
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            i = 2
            while (target_dir / f"{stem}-{i}{suffix}").exists():
                i += 1
            dest = target_dir / f"{stem}-{i}{suffix}"
        dest.write_bytes(data)
        out.append(dest)
    return out


@router.post("/collections/upsert")
async def upsert_collection(
    request: Request,
    metadata: str = Form(
        ...,
        description=(
            "JSON-encoded RagUpsertMetadata: "
            "{name, topics?, language?, cross_lingual?, authority_tier?, "
            "license?, dedupe_cosine_threshold?}"
        ),
    ),
    files: list[UploadFile] = File(..., description="One or more documents to ingest"),
) -> dict[str, Any]:
    """Create or append to a system RAG collection from uploaded files.

    The same logical pipeline runs whether ``name`` is new or already
    declared in ``retrieval.yaml``: the resolver inherits defaults on
    append and uses historical defaults (en / tier 2) for new collections.
    On a successful new-collection ingest, the runner auto-appends the
    rendered YAML snippet to ``configs/retrieval.yaml`` so the router
    sees the collection on the next reload — no manual git edit needed.

    The request returns the dispatched ``JobSpec`` immediately; the
    operator polls ``/admin/jobs/{id}`` for progress and the final
    summary (chunks written, dedup count, yaml-append outcome).
    """
    meta = _parse_metadata(metadata)
    if not _NAME_RE.match(meta.name):
        raise HTTPException(
            status_code=422,
            detail=f"name must match {_NAME_RE.pattern}",
        )
    if not files:
        raise HTTPException(status_code=422, detail="no files provided")
    if len(files) > _MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=422,
            detail=f"at most {_MAX_UPLOAD_FILES} files per request",
        )

    registry = request.app.state.jobs
    if not registry.has_runner("rag_ingest"):
        raise HTTPException(status_code=503, detail="rag_ingest runner not registered")

    # Save uploads under data/jobs/_uploads/<uuid>/. Using a UUID
    # directory (rather than the job id) means we can commit the files
    # to disk BEFORE the registry generates the job id — no rename
    # race between submit() dispatching the task and the runner reading
    # file paths. The runner takes the directory via ``upload_dir`` and
    # is responsible for cleaning it up on exit.
    import uuid as _uuid

    upload_dir = jobs_dir() / "_uploads" / _uuid.uuid4().hex
    saved_paths = _save_uploads(files, upload_dir)

    params: dict[str, Any] = {
        "name": meta.name,
        "file_paths": [str(p) for p in saved_paths],
        "upload_dir": str(upload_dir),
        "topics": list(meta.topics),
        "language": meta.language,
        "cross_lingual": meta.cross_lingual,
        "authority_tier": meta.authority_tier,
        "license": meta.license,
        "dedupe_cosine_threshold": meta.dedupe_cosine_threshold,
    }
    spec = await registry.submit("rag_ingest", params)

    audit_event(
        "admin.rag.upsert",
        payload={
            "job_id": spec.id,
            "name": meta.name,
            "file_count": len(saved_paths),
        },
    )
    return spec.to_json()
