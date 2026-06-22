"""``POST /api/v1/sessions/{id}/attachments`` — multipart upload endpoint.

Mirrors the TUI's paste path (``_ingest_clipboard_bytes``) in spirit but
without the chrome: for each uploaded file the endpoint runs the
content-addressable store, registers the row in the session attachments
tray, and either runs the text fast-path or enqueues an OCR job. The
endpoint returns the list of registered attachments so the SPA can
render preview chips immediately.

OCR worker is built lazily on first upload and cached on
``app.state.ocr_worker``. A worker build failure (missing provider
config, broken vision plugin) does not block the upload — the
attachment is registered with ``ocr_status="pending"`` and the next
stream call's ``_await_pending_ocr`` will time out gracefully rather
than dead-end the upload.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import quote

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)

from claritymed.core.observability.audit import audit_event
from claritymed.core.schemas import Account
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.paths import list_user_ids, user_sessions_dir
from claritymed.stores.session_attachments import (
    SessionAttachment,
    SessionAttachments,
)
from claritymed.web.deps import get_current_user
from claritymed.web.schemas import (
    AttachmentListResponse,
    AttachmentMetaItem,
    AttachmentMetaResponse,
    AttachmentResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["attachments"])

# Per-file cap. 25 MiB matches the practical upper bound for medical
# scans (PDF reports, hi-res photos of paper) without inviting abuse.
# A user with a larger image should compress or split it.
_MAX_BYTES = 25 * 1024 * 1024
# Per-request file cap. Mirrors StreamRequest.attachment_ids max_length=8.
_MAX_FILES = 8

# Whitelist of accepted extensions. Source of truth: OcrConfig.text_extensions
# (loaded once at module init) plus the document/image extensions that the
# routing OCR provider knows how to handle. Anything outside this set is
# rejected at the upload boundary rather than landing in the blob pool
# and failing later during OCR.
_IMAGE_EXTS = frozenset({"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "heic"})
_DOC_EXTS = frozenset({"pdf"})

# Accept either a full 64-char sha or an 8+ char hex prefix on the
# GET-by-sha route. The TUI / web persist references as `[File sha:XXXXXXXX]`
# with an 8-char prefix by default, so the read path must resolve them.
_SHA_LOOKUP_RE = re.compile(r"^[a-f0-9]{8,64}$")

# Batch meta-lookup ceiling. A long chat history could carry many
# attachment markers; 50 covers practical chat scrollback in one round
# trip without inviting unbounded fan-out. Bump after measuring.
_META_BATCH_LIMIT = 50


@router.post(
    "/sessions/{session_id}/attachments",
    response_model=AttachmentListResponse,
)
async def upload_attachments(
    session_id: str,
    request: Request,
    files: list[UploadFile],
    account: Account = Depends(get_current_user),
) -> AttachmentListResponse:
    """Accept multipart ``files=…`` parts. Returns one entry per stored blob.

    Validation:
    * Session must exist and belong to this user (404/403 otherwise).
    * 1–``_MAX_FILES`` parts (422 otherwise).
    * Each file's size <= ``_MAX_BYTES`` (413 otherwise).
    * Extension must be in the image / document / text whitelist
      (415 otherwise).
    """
    _validate_session_ownership(account.user_id, session_id)
    if not files:
        raise HTTPException(status_code=422, detail="No files provided")
    if len(files) > _MAX_FILES:
        raise HTTPException(
            status_code=422,
            detail=f"At most {_MAX_FILES} files per request",
        )

    text_exts = _load_text_extensions()
    accepted: list[AttachmentResponse] = []

    for upload in files:
        data = await upload.read()
        if not data:
            raise HTTPException(
                status_code=422, detail=f"Empty file: {upload.filename!r}"
            )
        if len(data) > _MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"{upload.filename!r} is {len(data)} bytes; cap is "
                    f"{_MAX_BYTES} bytes"
                ),
            )
        ext = _resolve_extension(upload.filename or "", text_exts)
        if ext is None:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type for {upload.filename!r}; "
                    "allowed: images (png/jpg/webp/…), PDF, common text."
                ),
            )
        mime = upload.content_type or _guess_mime(ext)
        row, kind = _store_one(
            user_id=account.user_id,
            session_id=session_id,
            data=data,
            ext=ext,
            mime=mime,
            filename=upload.filename or f"upload.{ext}",
            text_exts=text_exts,
            app_state=request.app.state,
        )
        accepted.append(
            AttachmentResponse(
                id=row.sha256,
                filename=row.filename,
                mime_type=row.mime,
                size_bytes=row.size,
                kind=kind,
                ocr_status=row.ocr_status,
            )
        )
        audit_event(
            "ocr.extract",
            payload={
                "user_id": account.user_id,
                "session_id": session_id,
                "sha256": row.sha256,
                "source": "web_upload",
                "kind": kind,
                "size_bytes": row.size,
            },
        )

    return AttachmentListResponse(attachments=accepted)


@router.get(
    "/sessions/{session_id}/attachments/meta",
    response_model=AttachmentMetaResponse,
)
async def get_attachments_meta(
    session_id: str,
    sha: list[str] = Query(default_factory=list),  # noqa: B008
    account: Account = Depends(get_current_user),
) -> AttachmentMetaResponse:
    """Batch resolve sha prefixes / full shas to attachment metadata.

    Required when the chat transcript carries multiple ``[File sha:…]`` /
    ``[Image sha:…]`` markers and the SPA wants to render chips for all
    of them in one round trip. Each ``sha`` query param is resolved
    independently; per-entry errors do not fail the whole call so the
    frontend can still render the hits and flag the misses.
    """
    _validate_session_ownership(account.user_id, session_id)

    if not sha:
        raise HTTPException(
            status_code=422, detail="At least one sha query param is required"
        )
    if len(sha) > _META_BATCH_LIMIT:
        raise HTTPException(
            status_code=422,
            detail=f"At most {_META_BATCH_LIMIT} sha values per request",
        )

    rows = SessionAttachments(account.user_id, session_id).list()
    items: list[AttachmentMetaItem] = []
    for raw in sha:
        items.append(_resolve_one_meta(raw, rows))
    return AttachmentMetaResponse(items=items)


@router.get("/sessions/{session_id}/attachments/{sha}")
async def get_attachment(
    session_id: str,
    sha: str,
    account: Account = Depends(get_current_user),
) -> Response:
    """Return raw image bytes inline. Accepts full or 8+ char prefix sha.

    Only ``image/*`` rows are served from this endpoint — non-image
    attachments (PDF, text) get a 415 because the SPA renders those as
    filename chips, not inline content, and the bytes carry no value to
    the browser. Use ``/meta`` to resolve their metadata.

    Status codes:
    * 400 — sha is not 8–64 hex chars.
    * 403 — session belongs to another user.
    * 404 — no session, no matching attachment, or blob bytes missing.
    * 409 — prefix matches more than one attachment; caller must refetch
      with more characters. Frontend stores 8-char prefixes by default so
      collisions on real sha256 are vanishingly rare, but fail-loud is
      better than serving a guess.
    * 415 — the matched attachment is not an image.
    """
    _validate_session_ownership(account.user_id, session_id)

    sha = sha.lower()
    if not _SHA_LOOKUP_RE.match(sha):
        raise HTTPException(
            status_code=400, detail="Invalid sha; expected 8–64 hex chars"
        )

    rows = SessionAttachments(account.user_id, session_id).list()
    matches = [r for r in rows if r.sha256.startswith(sha)]
    if not matches:
        raise HTTPException(status_code=404, detail="Attachment not found")
    if len(matches) > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Sha prefix matches {len(matches)} attachments; use more characters"
            ),
        )
    row = matches[0]

    if not row.mime.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Only image/* attachments are served as bytes; use /meta",
        )

    blob_dir = BlobStore(account.user_id).dir(row.sha256)
    content_path: Path | None = None
    if blob_dir.exists():
        content_path = next(
            (
                p
                for p in blob_dir.iterdir()
                if p.name.startswith("content.") and not p.name.endswith(".tmp")
            ),
            None,
        )
    if content_path is None:
        raise HTTPException(status_code=404, detail="Blob bytes missing")

    data = content_path.read_bytes()
    # RFC 5987 filename encoding — handles unicode and quote characters
    # without breaking the header. Browsers fall back to the bare token
    # if they can't parse `filename*`, which is fine.
    encoded = quote(row.filename, safe="")
    return Response(
        content=data,
        media_type=row.mime,
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{encoded}"},
    )


# --- helpers -----------------------------------------------------------


def _resolve_one_meta(raw: str, rows: list[SessionAttachment]) -> AttachmentMetaItem:
    """Resolve a single sha query value against the session's rows.

    Mirrors the per-sha endpoint's lookup semantics (lowercase, 8–64 hex
    chars, ``startswith`` match) but returns a structured per-entry
    result instead of raising — the batch endpoint stays partially-
    successful even when one item collides or is missing.
    """
    sha = raw.lower()
    if not _SHA_LOOKUP_RE.match(sha):
        return AttachmentMetaItem(requested=raw, error="invalid")
    matches = [r for r in rows if r.sha256.startswith(sha)]
    if not matches:
        return AttachmentMetaItem(requested=raw, error="not_found")
    if len(matches) > 1:
        return AttachmentMetaItem(requested=raw, error="ambiguous")
    row = matches[0]
    return AttachmentMetaItem(
        requested=raw,
        resolved=AttachmentResponse(
            id=row.sha256,
            filename=row.filename,
            mime_type=row.mime,
            size_bytes=row.size,
            kind=_kind_from_mime(row.mime),
            ocr_status=row.ocr_status,
        ),
    )


def _kind_from_mime(mime: str) -> str:
    """Map mime → coarse ``kind`` label used by ``AttachmentResponse``."""
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("text/"):
        return "text"
    return "other"


def _store_one(
    *,
    user_id: str,
    session_id: str,
    data: bytes,
    ext: str,
    mime: str,
    filename: str,
    text_exts: frozenset[str],
    app_state,
) -> tuple[SessionAttachment, str]:
    """Store the blob, register the row, run text fast-path or enqueue OCR.

    Returns the persisted ``SessionAttachment`` and a coarse ``kind``
    label (``image`` / ``text`` / ``other``) so the response chip can
    pick the right icon.
    """
    blob_store = BlobStore(user_id)
    sha = blob_store.store(data, ext)
    row = SessionAttachments(user_id, session_id).add(
        sha256=sha,
        filename=filename,
        mime=mime,
        size=len(data),
        source="upload",
    )

    dotted = f".{ext.lower()}"
    if dotted in text_exts:
        _run_text_fast_path(
            blob_store=blob_store,
            user_id=user_id,
            session_id=session_id,
            sha=sha,
            ext=ext,
            data=data,
            filename=filename,
        )
        # Re-read the row to pick up the ``done``/``empty`` status the
        # fast-path just wrote.
        refreshed = SessionAttachments(user_id, session_id).get(sha) or row
        return refreshed, "text"

    kind = "image" if mime.startswith("image/") else "other"
    worker = _ensure_ocr_worker(app_state)
    if worker is None:
        logger.warning("ocr worker unavailable; attachment %s left pending", sha[:8])
        return row, kind

    _enqueue_ocr(
        worker=worker,
        user_id=user_id,
        session_id=session_id,
        sha=sha,
        ext=ext,
        filename=filename,
        blob_store=blob_store,
    )
    return row, kind


def _run_text_fast_path(
    *,
    blob_store: BlobStore,
    user_id: str,
    session_id: str,
    sha: str,
    ext: str,
    data: bytes,
    filename: str,
) -> None:
    """Decode utf-8 with replacement, write sentinel, mark attachment done."""
    text = data.decode("utf-8", errors="replace")
    status = "done" if text.strip() else "empty"
    blob_store.write_ocr_result(
        sha,
        status=status,
        kind="text",
        ext=ext,
        provider="text",
        chain_tried=["text"],
        reason=None,
        text=text,
        original_filename=filename,
    )
    SessionAttachments(user_id, session_id).mark_ocr_status(
        sha, status, provider="text", reason=None
    )


def _enqueue_ocr(
    *,
    worker,
    user_id: str,
    session_id: str,
    sha: str,
    ext: str,
    filename: str,
    blob_store: BlobStore,
) -> None:
    """Submit one OCR job; safe to call from a request handler."""
    from claritymed.context import apply_context, language_ctx, new_request_id
    from claritymed.orchestrator.services.ocr_worker import OcrJob

    blob_dir = blob_store.dir(sha)
    content_path = next(
        (
            p
            for p in blob_dir.iterdir()
            if p.name.startswith("content.") and not p.name.endswith(".tmp")
        ),
        None,
    )
    if content_path is None:
        logger.error("blob dir missing content file for sha=%s", sha[:8])
        return

    # OcrWorker captures ContextVars at enqueue time so its audit rows
    # attribute back to this request. The request already has rid/user/lang
    # set by WebContextMiddleware, but extracting them via apply_context
    # is a no-op when they're already populated.
    tokens = apply_context(new_request_id(), user_id, language_ctx.get() or "en")
    try:
        worker.enqueue(
            OcrJob(
                user_id=user_id,
                session_id=session_id,
                sha256=sha,
                blob_path=content_path,
                is_phi=True,
                original_filename=filename,
            )
        )
    finally:
        # Restore — apply_context returns Tokens compatible with reset_context.
        from claritymed.context import reset_context

        reset_context(tokens)


def _ensure_ocr_worker(app_state):
    """Lazy-build the per-app OcrWorker on first upload.

    Returns ``None`` on construction failure; the caller then leaves the
    attachment pending and surfaces a warning in the API logs. We do NOT
    raise — a misconfigured OCR provider should degrade gracefully so
    uploads still register and the user sees their files in the chip row.
    """
    existing = getattr(app_state, "ocr_worker", None)
    if existing is not None:
        return existing
    from claritymed.orchestrator.services.ocr_worker_factory import (
        build_ocr_worker,
    )

    worker = build_ocr_worker(listener=None)
    app_state.ocr_worker = worker
    return worker


def _load_text_extensions() -> frozenset[str]:
    """Cached load of OcrConfig.text_extensions (lowercased, dotted)."""
    global _TEXT_EXT_CACHE
    cached = _TEXT_EXT_CACHE
    if cached is not None:
        return cached
    try:
        from claritymed.core.schemas.ocr import load_ocr_config

        cfg = load_ocr_config()
        cached = frozenset(cfg.text_extensions)
    except Exception:  # noqa: BLE001
        logger.exception("ocr config load failed; text fast-path disabled")
        cached = frozenset()
    _TEXT_EXT_CACHE = cached
    return cached


_TEXT_EXT_CACHE: frozenset[str] | None = None


def _resolve_extension(filename: str, text_exts: frozenset[str]) -> str | None:
    """Return the lowercased extension if it's in any allowlist; else None."""
    raw = Path(filename).suffix.lower().lstrip(".")
    if not raw:
        return None
    if raw in _IMAGE_EXTS or raw in _DOC_EXTS:
        return raw
    if f".{raw}" in text_exts:
        return raw
    return None


def _guess_mime(ext: str) -> str:
    """Coarse mime mapping for the response only."""
    if ext in _IMAGE_EXTS:
        if ext in {"jpg", "jpeg"}:
            return "image/jpeg"
        return f"image/{ext}"
    if ext == "pdf":
        return "application/pdf"
    return "text/plain"


def _validate_session_ownership(user_id: str, session_id: str) -> None:
    """Same shape as the chat router's helper; 404 missing, 403 cross-user."""
    own = user_sessions_dir(user_id) / f"{session_id}.jsonl"
    if own.exists():
        return
    for other in list_user_ids():
        if other == user_id:
            continue
        if (user_sessions_dir(other) / f"{session_id}.jsonl").exists():
            raise HTTPException(status_code=403, detail="Forbidden")
    raise HTTPException(status_code=404, detail="Session not found")
