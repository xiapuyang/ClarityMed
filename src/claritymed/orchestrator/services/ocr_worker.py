"""Background OCR worker — extracts text from blobs after they land.

Pasted images and uploaded PDFs are saved to the CAS blob pool
synchronously (Unit 1's BlobStore.store). OCR is async and can take
seconds, so it runs on a long-lived background task per AskService
instance: each enqueue captures ``contextvars.copy_context()`` so the
worker's audit events stay attributed to the originating request id,
the worker pops items off an asyncio.Queue, runs the chain, and writes
``ocr.md`` + the ``ocr.json`` sentinel via ``BlobStore``.

The completion sentinel is the last step — ``ocr.json`` lands only
after ``ocr.md`` has been atomically renamed in. ``BlobStore.ocr_done``
checks this single file's existence so a crash mid-OCR cannot leave a
half-written extraction looking complete.

When OCR finishes, the worker emits an ``OcrCompleted`` event into a
broadcast channel; ``AskService`` (Unit 8 wiring) subscribes and either
auto-resumes the user's last turn or just rolls the new status into the
next turn's envelope.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from claritymed.core.ocr.base import OcrError, OcrProvider
from claritymed.core.schemas.records import OcrStatus
from claritymed.orchestrator.services.session_attachments import SessionAttachments
from claritymed.stores.blob_store import BlobStore

logger = logging.getLogger(__name__)


@dataclass
class OcrJob:
    """One queued extraction request."""

    user_id: str
    session_id: str
    sha256: str
    blob_path: Path
    is_phi: bool = True


@dataclass(frozen=True)
class OcrCompleted:
    """Emitted by the worker when a job finishes (any status)."""

    user_id: str
    session_id: str
    sha256: str
    status: OcrStatus
    provider: str | None = None
    reason: str | None = None


CompletionListener = Callable[[OcrCompleted], Awaitable[None] | None]


class OcrWorker:
    """One background task per AskService instance. Single concurrent OCR."""

    def __init__(
        self,
        provider: OcrProvider,
        *,
        listener: CompletionListener | None = None,
    ) -> None:
        self._provider = provider
        self._listener = listener
        self._queue: asyncio.Queue[tuple[contextvars.Context, OcrJob]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    # --- lifecycle ---------------------------------------------------

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="ocr-worker")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    # --- enqueue -----------------------------------------------------

    def enqueue(self, job: OcrJob) -> None:
        """Snapshot the caller's ContextVars so audit attribution holds.

        ``copy_context`` captures the running call's request_id /
        user_id / language. The worker pops the (ctx, job) pair and
        invokes ``ctx.run(...)`` so ``audit_event`` calls inside the
        OCR provider observe the originator's identity even though the
        request has already returned.
        """
        ctx = contextvars.copy_context()
        self._queue.put_nowait((ctx, job))

    # --- internals ---------------------------------------------------

    async def _loop(self) -> None:
        while True:
            ctx, job = await self._queue.get()
            try:
                # asyncio.Task(context=ctx) propagates the captured
                # request_id / user_id / language into every audit_event
                # the OCR provider emits. Without it the call fires
                # seconds after the request returned and audit_event
                # silently dies on MissingContextError.
                task = asyncio.get_running_loop().create_task(
                    self._extract(job), context=ctx
                )
                completion = await task
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("ocr worker: unhandled error for %s", job.sha256[:8])
                completion = OcrCompleted(
                    user_id=job.user_id,
                    session_id=job.session_id,
                    sha256=job.sha256,
                    status="failed",
                    reason="worker error",
                )
            await self._emit(completion)
            self._queue.task_done()

    async def _extract(self, job: OcrJob) -> OcrCompleted:
        # Sentinel on disk = a prior worker run already produced a result
        # for this blob. Read it and decide:
        #   * status="done"/"empty" → short-circuit with provider="cache"
        #     so the UI can show a cache hit instead of re-extracting.
        #   * status="failed" → treat as miss and re-run. Otherwise a
        #     single bad run (e.g. CLARITYMED_ALLOW_MINERU not set when
        #     the worker started) sticks forever even after the cause is
        #     fixed, blocking every retry with the same sha.
        blob_store = BlobStore(job.user_id)
        cached = self._read_cached_sentinel(blob_store, job.sha256)
        if cached is not None and cached.get("status") != "failed":
            return OcrCompleted(
                user_id=job.user_id,
                session_id=job.session_id,
                sha256=job.sha256,
                status=cached.get("status", "done"),
                provider="cache",
                reason=cached.get("reason"),
            )
        try:
            text = await self._provider.extract_text(job.blob_path)
        except OcrError as exc:
            logger.warning(
                "ocr provider error for %s (%s): %s",
                job.sha256[:8],
                job.blob_path.name,
                exc,
            )
            self._write_sentinel(
                blob_store,
                job.sha256,
                status="failed",
                provider=None,
                reason=str(exc),
                text="",
            )
            return OcrCompleted(
                user_id=job.user_id,
                session_id=job.session_id,
                sha256=job.sha256,
                status="failed",
                reason=str(exc),
            )
        status: OcrStatus = "done" if text.strip() else "empty"
        provider_label = type(self._provider).__name__
        self._write_sentinel(
            blob_store,
            job.sha256,
            status=status,
            provider=provider_label,
            reason=None,
            text=text,
        )
        return OcrCompleted(
            user_id=job.user_id,
            session_id=job.session_id,
            sha256=job.sha256,
            status=status,
            provider=provider_label,
        )

    def _read_cached_sentinel(self, blob_store: BlobStore, sha256: str) -> dict | None:
        """Return the parsed ocr.json contents, or None if no sentinel."""
        if not blob_store.ocr_done(sha256):
            return None
        try:
            return json.loads(
                blob_store.ocr_meta_path(sha256).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt sentinel is no more useful than no sentinel —
            # log and re-run extraction rather than crash the worker.
            logger.warning(
                "ocr worker: corrupt sentinel for %s, re-extracting (%s)",
                sha256[:8],
                exc,
            )
            return None

    def _write_sentinel(
        self,
        blob_store: BlobStore,
        sha256: str,
        *,
        status: OcrStatus,
        provider: str | None,
        reason: str | None,
        text: str,
    ) -> None:
        """Write ocr.md.tmp → rename, then ocr.json.tmp → rename (sentinel)."""
        ocr_md = blob_store.ocr_path(sha256)
        ocr_md.parent.mkdir(parents=True, exist_ok=True)
        tmp_md = ocr_md.with_suffix(".md.tmp")
        tmp_md.write_text(text or "", encoding="utf-8")
        tmp_md.replace(ocr_md)

        ocr_json = blob_store.ocr_meta_path(sha256)
        tmp_json = ocr_json.with_suffix(".json.tmp")
        tmp_json.write_text(
            json.dumps(
                {
                    "status": status,
                    "provider": provider,
                    "reason": reason,
                    "chars": len(text or ""),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_json.replace(ocr_json)

    async def _emit(self, completion: OcrCompleted) -> None:
        # Update the session-attachments tray (idempotent).
        attachments = SessionAttachments(completion.user_id, completion.session_id)
        attachments.mark_ocr_status(
            completion.sha256,
            completion.status,
            provider=completion.provider,
            reason=completion.reason,
        )
        if self._listener is None:
            return
        result = self._listener(completion)
        if asyncio.iscoroutine(result):
            await result
