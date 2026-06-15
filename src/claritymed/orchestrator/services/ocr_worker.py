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

from claritymed.core.medical_clip.client import MedicalClipClient
from claritymed.core.ocr.base import OcrError, OcrProvider
from claritymed.core.schemas.records import OcrStatus
from claritymed.core.vision.ocr_report_detector import (
    DEFAULT_MIN_CHARS,
    has_structured_report,
)
from claritymed.errors import MedicalClipUnreachableError
from claritymed.stores.session_attachments import SessionAttachments
from claritymed.stores.blob_store import BlobStore

logger = logging.getLogger(__name__)

# Image extensions for which we run modality classification + the
# OCR-report heuristic. PDFs / DOC / TXT skip both: BiomedCLIP only
# accepts raster images, and a PDF that IS a clinician's report is
# handled by the text path on the LLM side without needing the
# `ocr_has_report` flag (the textual content speaks for itself).
_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif", ".heic"}
)


def _is_image(blob_path: Path) -> bool:
    return blob_path.suffix.lower() in _IMAGE_EXTS


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
    """One background task per AskService instance. Single concurrent OCR.

    When ``medical_clip_client`` and ``ocr_report_config`` are supplied,
    each image-blob job is additionally tagged with the BiomedCLIP
    modality (``modality`` / ``modality_confidence`` / ``is_medical``)
    and the structured-report heuristic (``ocr_has_report``). Both are
    optional so non-image OCR (PDF, plain text) and environments without
    a running medical-clip server still work — the worker just omits the
    extra fields from the sentinel and the vision plugin treats absence
    as "unknown" downstream.
    """

    def __init__(
        self,
        provider: OcrProvider,
        *,
        listener: CompletionListener | None = None,
        medical_clip_client: MedicalClipClient | None = None,
        ocr_report_config: dict | None = None,
    ) -> None:
        self._provider = provider
        self._listener = listener
        self._medical_clip = medical_clip_client
        # ``ocr_report_config`` shape: ``{"min_chars": int, "markers": dict[str, list[str]]}``
        # (the return value of :func:`load_ocr_report_config`). ``None``
        # disables the report heuristic entirely — every blob gets
        # ``ocr_has_report`` omitted, which the downstream renderer
        # treats the same as ``ocr_has_report=false`` (vision tool may
        # still run; it's the conservative default).
        cfg = ocr_report_config or {}
        self._ocr_report_min_chars = int(cfg.get("min_chars", DEFAULT_MIN_CHARS))
        self._ocr_report_markers: dict[str, list[str]] = cfg.get("markers") or {}
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
        # Single-writer sequence (plan KTD-V3 + KTD-V6): after OCR text
        # is in hand, the same task runs modality classification + the
        # report-override heuristic, then writes ocr.json once with
        # every field populated. Two parallel writers to ocr.json would
        # race; one writer with two sub-steps does not.
        vision_tags = await self._compute_vision_tags(job, result.text)
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
            **vision_tags,
        )
        return OcrCompleted(
            user_id=job.user_id,
            session_id=job.session_id,
            sha256=job.sha256,
            status=status,
            provider=result.provider_used,
        )

    async def _compute_vision_tags(self, job: OcrJob, ocr_text: str) -> dict:
        """Build the modality / is_medical / ocr_has_report kwargs.

        Returns the subset of kwargs that should be threaded into
        :meth:`BlobStore.write_ocr_result`. Skips silently for non-image
        blobs (PDFs, plain text) so the sentinel stays free of
        unmeaningful fields.
        """
        if not _is_image(job.blob_path):
            return {}
        tags: dict = {}
        warnings: list[str] = []
        if self._medical_clip is not None:
            try:
                image_bytes = job.blob_path.read_bytes()
                response = await self._medical_clip.classify_modality(
                    image_bytes,
                    request_id=f"ocr_{job.sha256[:16]}",
                    sha256=job.sha256,
                )
                tags["modality"] = response.modality
                tags["modality_confidence"] = float(response.confidence)
                tags["is_medical"] = bool(response.is_medical)
            except MedicalClipUnreachableError as exc:
                # KTD-V8 graceful: server down → keep OCR moving. The
                # vision plugin treats missing/unknown modality the
                # same as classifier-low-confidence (asks the user).
                # ``is_medical`` deliberately omitted — write_ocr_result
                # drops None-valued kwargs from the payload, and field
                # absence is the LLM-side signal "no opinion" (a stale
                # False would falsely claim "the classifier saw this
                # and decided it isn't medical").
                logger.warning(
                    "medical-clip unreachable for %s: %s; tagging modality=unknown",
                    job.sha256[:8],
                    exc,
                )
                tags["modality"] = "unknown"
                warnings.append(f"medical_clip_unreachable: {exc!s}")
            except Exception as exc:  # noqa: BLE001
                # 4xx (image_decode_failed, image_hash_mismatch) and any
                # other unexpected shape land here. Same posture as the
                # unreachable branch — OCR ingest is the priority.
                logger.warning(
                    "modality classification failed for %s: %s",
                    job.sha256[:8],
                    exc,
                )
                tags["modality"] = "unknown"
                warnings.append(f"modality_classification_failed: {exc!s}")
        # Report-override heuristic. Cheap; always run when configured,
        # regardless of whether modality classification succeeded —
        # ``ocr_has_report`` is independent of modality and the LLM-side
        # branch in Unit 7 reads it before reading modality.
        if self._ocr_report_markers:
            try:
                tags["ocr_has_report"] = has_structured_report(
                    ocr_text,
                    language=None,
                    min_chars=self._ocr_report_min_chars,
                    markers=self._ocr_report_markers,
                )
            except Exception as exc:  # noqa: BLE001 — pure function, but be paranoid
                logger.warning(
                    "ocr_has_report heuristic failed for %s: %s",
                    job.sha256[:8],
                    exc,
                )
                warnings.append(f"ocr_report_heuristic_failed: {exc!s}")
        if warnings:
            tags["vision_warnings"] = warnings
        return tags

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
