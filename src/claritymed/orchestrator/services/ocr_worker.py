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
    original_filename: str | None = None
    """User-facing name at paste/upload time (e.g. ``化验单.png``).
    Threaded into ``ocr.json`` for disaster-recovery and into the
    ``ocr.extract`` audit event so log consumers see the real name
    instead of the on-disk ``content.<ext>`` placeholder. ``None``
    when the caller doesn't have a meaningful name (rare)."""


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
            except Exception as exc:  # noqa: BLE001
                logger.exception("ocr worker: unhandled error for %s", job.sha256[:8])
                # Without writing the failure sentinel, BlobStore.ocr_done
                # stays False forever and the blob is stuck in "pending"
                # across restarts even though the worker already gave up.
                # Mirror the OcrError branch in _extract: persist a failed
                # ocr.json so a subsequent worker run either re-tries (the
                # _read_cached_sentinel branch treats status="failed" as
                # a miss) or short-circuits if the failure is permanent.
                try:
                    BlobStore(job.user_id).write_ocr_result(
                        job.sha256,
                        status="failed",
                        kind="ocr",
                        ext=job.blob_path.suffix.lstrip("."),
                        provider=None,
                        chain_tried=[],
                        reason=f"worker error: {exc!r}",
                        text="",
                        original_filename=job.original_filename,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "ocr worker: failed to persist failure sentinel for %s",
                        job.sha256[:8],
                    )
                completion = OcrCompleted(
                    user_id=job.user_id,
                    session_id=job.session_id,
                    sha256=job.sha256,
                    status="failed",
                    reason="worker error",
                )
            # ``_emit`` updates SessionAttachments + calls the listener; a
            # listener exception must NOT kill _loop or every subsequent
            # job sits in the queue forever (silent worker death).
            try:
                await self._emit(completion)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "ocr worker: emit failed for %s; continuing", job.sha256[:8]
                )
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
        # Stash the user-facing filename so the routing-layer audit
        # event can record it without growing the OcrProvider signature.
        from claritymed.core.ocr.routing_provider import (
            reset_original_filename,
            set_original_filename,
        )

        filename_token = set_original_filename(job.original_filename)
        try:
            try:
                result = await self._provider.extract_text(job.blob_path)
            finally:
                reset_original_filename(filename_token)
        except OcrError as exc:
            logger.warning(
                "ocr provider error for %s (%s): %s",
                job.sha256[:8],
                job.blob_path.name,
                exc,
            )
            blob_store.write_ocr_result(
                job.sha256,
                status="failed",
                kind="ocr",
                ext=job.blob_path.suffix.lstrip("."),
                provider=None,
                chain_tried=[],
                reason=str(exc),
                text="",
                original_filename=job.original_filename,
            )
            return OcrCompleted(
                user_id=job.user_id,
                session_id=job.session_id,
                sha256=job.sha256,
                status="failed",
                reason=str(exc),
            )
        status: OcrStatus = "done" if result.text.strip() else "empty"
        blob_store.write_ocr_result(
            job.sha256,
            status=status,
            kind="ocr",
            ext=job.blob_path.suffix.lstrip("."),
            provider=result.provider_used,
            chain_tried=list(result.chain_tried),
            reason=None,
            text=result.text,
            original_filename=job.original_filename,
        )
        return OcrCompleted(
            user_id=job.user_id,
            session_id=job.session_id,
            sha256=job.sha256,
            status=status,
            provider=result.provider_used,
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
