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
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from claritymed.core.medical_clip.client import MedicalClipClient
from claritymed.core.ocr.base import ExtractResult, OcrEmpty, OcrError, OcrProvider
from claritymed.core.ocr.pdf_image_peek import (
    has_vision_sidecar,
    maybe_rasterize_single_image_pdf,
    vision_sidecar_path,
    write_vision_sidecar,
)
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

# Per-job wall-clock cap. Without this, one hung provider (LLM-OCR call
# that never returns, frozen subprocess, network stall to MineRU) holds
# the single asyncio.Queue indefinitely and every subsequent paste sits
# in ``pending`` forever. 5 minutes is generous for the slowest
# realistic chain (LLM-OCR with a large multi-page PDF) but short enough
# that a stuck queue self-heals within one TUI session.
OCR_JOB_TIMEOUT_S = 300.0

# medical-clip's non-medical buckets — when ``classify_modality`` lands on
# one of these, its ``is_medical`` is forced false by the server's gating
# layer regardless of confidence (see ``servers/medical_clip/app.py``).
# The LLM-OCR fallback treats these as "soft" decisions: a vision LLM
# that read both the pixels and the surrounding text may legitimately
# upgrade a histopath slide that BiomedCLIP bucketed as ``document``.
_CLIP_NON_MEDICAL_BUCKETS: frozenset[str] = frozenset({"unknown", "photo", "document"})

# Concrete medical imaging modalities. The LLM-OCR override only flips
# medical-clip's non-medical bucket when the LLM names one of these AND
# claims ``is_medical=True`` — refusing to swap one non-medical label for
# another keeps the heuristic monotone (overrides only add medical
# signal, never erase one).
_LLM_MEDICAL_MODALITIES: frozenset[str] = frozenset(
    {"ultrasound", "ct", "xray", "dermoscopy", "histopathology"}
)


def _has_vision_payload(blob_dir: Path, blob_path: Path) -> bool:
    """True iff this blob has bytes the vision pipeline can decode.

    The ``vision.png`` sidecar (written by the 1-page image-PDF
    rasterizer) takes precedence over the extension check: a PDF whose
    single page was a CT scan gets ``content.pdf`` AND ``vision.png``
    on disk, and the latter is what every downstream consumer
    (medical-clip, vision server) should see.
    """
    return has_vision_sidecar(blob_dir) or blob_path.suffix.lower() in _IMAGE_EXTS


def _resolve_vision_payload(
    blob_dir: Path, blob_path: Path, content_sha: str
) -> tuple[bytes, str] | None:
    """Return ``(bytes, sha256)`` for the image source vision should see.

    Prefers the ``vision.png`` sidecar (PDF-rasterized) over the
    original ``content.<ext>``. ``content_sha`` is the pre-computed
    sha of the original blob and is reused when no sidecar is present
    so we don't re-hash the same bytes on every modality call. The
    sidecar branch computes its own sha because the rasterized PNG's
    hash differs from the source PDF's hash and the medical-clip
    server cross-checks the digest on the wire.

    Returns ``None`` when the blob has no vision-decodable payload
    (non-image extension and no sidecar).
    """
    sidecar = vision_sidecar_path(blob_dir)
    if sidecar.exists():
        png_bytes = sidecar.read_bytes()
        return png_bytes, hashlib.sha256(png_bytes).hexdigest()
    if blob_path.suffix.lower() in _IMAGE_EXTS:
        return blob_path.read_bytes(), content_sha
    return None


def _is_stale_sentinel(cached: dict) -> bool:
    """Return True when *cached* should be treated as a cache miss.

    Two flavours qualify:

    * ``status="failed"`` — already in scope (operator could have fixed
      the upstream cause; retry instead of locking the blob in failure).
    * ``status="empty"`` with ``chain_tried`` empty — a legacy pre-fix
      empty sentinel. Old worker code hardcoded ``chain_tried=[]`` AND
      skipped modality classification on the empty path, so the rendered
      ``<image>`` tag came out bare and the LLM-side routing rules in
      ``detect_disease_from_image_tool`` had nothing to fire on. The
      post-fix writer always records the chain it walked, so an empty
      ``chain_tried`` is the unambiguous "wrote this before the fix"
      probe. Re-extracting once promotes the sentinel into the new shape
      and unsticks the blob for every future turn.
    """
    if cached.get("status") == "failed":
        return True
    if cached.get("status") == "empty" and not cached.get("chain_tried"):
        return True
    return False


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
                # ``wait_for`` cancels ``task`` on timeout and re-raises
                # ``TimeoutError``. Without this cap a single hung OCR
                # call (e.g. an LLM provider that never returns) would
                # block the queue forever and every subsequent paste
                # would sit in ``pending`` indefinitely.
                completion = await asyncio.wait_for(task, timeout=OCR_JOB_TIMEOUT_S)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                logger.error(
                    "ocr worker: job timed out after %.0fs for %s",
                    OCR_JOB_TIMEOUT_S,
                    job.sha256[:8],
                )
                completion = self._failure_completion(
                    job,
                    reason=f"timed out after {OCR_JOB_TIMEOUT_S:.0f}s",
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("ocr worker: unhandled error for %s", job.sha256[:8])
                completion = self._failure_completion(
                    job,
                    reason=f"worker error: {exc!r}",
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

    def _failure_completion(self, job: "OcrJob", *, reason: str) -> "OcrCompleted":
        """Persist a failure sentinel and build the matching event.

        Used by both the timeout path and the unhandled-exception path
        in ``_loop``. Without writing the sentinel, ``BlobStore.ocr_done``
        stays False forever and the blob is stuck in ``pending`` across
        restarts even though the worker already gave up. Mirrors the
        OcrError branch in ``_extract``: persist a failed ``ocr.json``
        so a subsequent worker run either re-tries (the
        ``_read_cached_sentinel`` branch treats ``status="failed"`` as
        a miss) or short-circuits if the failure is permanent.
        """
        try:
            BlobStore(job.user_id).write_ocr_result(
                job.sha256,
                status="failed",
                kind="ocr",
                ext=job.blob_path.suffix.lstrip("."),
                provider=None,
                chain_tried=[],
                reason=reason,
                text="",
                original_filename=job.original_filename,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "ocr worker: failed to persist failure sentinel for %s",
                job.sha256[:8],
            )
        return OcrCompleted(
            user_id=job.user_id,
            session_id=job.session_id,
            sha256=job.sha256,
            status="failed",
            reason=reason,
        )

    async def _extract(self, job: OcrJob) -> OcrCompleted:
        # Sentinel on disk = a prior worker run already produced a result
        # for this blob. Read it and decide:
        #   * status="done"/"empty" → short-circuit with provider="cache"
        #     so the UI can show a cache hit instead of re-extracting.
        #   * status="failed" → treat as miss and re-run. Otherwise a
        #     single bad run (e.g. CLARITYMED_ALLOW_MINERU not set when
        #     the worker started) sticks forever even after the cause is
        #     fixed, blocking every retry with the same sha.
        #   * status="empty" AND chain_tried is empty (legacy pre-fix
        #     sentinel) → also treat as miss. Pre-fix code hardcoded
        #     chain_tried=[] on the empty path and skipped modality
        #     classification entirely, so those sentinels render as bare
        #     <image> tags and the LLM-side routing rules can't fire. The
        #     new code always writes the chain it walked, so an empty
        #     chain_tried is the unambiguous "this sentinel pre-dates the
        #     fix" probe. Re-extracting is cheap relative to "every empty
        #     image is permanently dead across user sessions".
        blob_store = BlobStore(job.user_id)
        cached = self._read_cached_sentinel(blob_store, job.sha256)
        if cached is not None and not _is_stale_sentinel(cached):
            return OcrCompleted(
                user_id=job.user_id,
                session_id=job.session_id,
                sha256=job.sha256,
                status=cached.get("status", "done"),
                provider="cache",
                reason=cached.get("reason"),
            )
        # Stage-1 PDF→image short-circuit: 1-page image-only PDFs (e.g.
        # a CT slice exported as a PDF wrapper) are routed through the
        # vision pipeline rather than MineRU. ``maybe_rasterize_…``
        # returns ``None`` for every PDF that should keep flowing
        # through the text path (multi-page reports, text-bearing
        # forms, corrupt files). On a hit we write ``vision.png`` next
        # to ``content.pdf`` and skip the OCR provider entirely — the
        # page has no extractable text by definition, so MineRU would
        # spend a 5-30s cloud round-trip returning the empty string.
        if job.blob_path.suffix.lower() == ".pdf":
            short_circuit = await self._handle_single_image_pdf(blob_store, job)
            if short_circuit is not None:
                return short_circuit
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
        except OcrEmpty as exc:
            logger.debug(
                "ocr: no text in %s (%s)",
                job.sha256[:8],
                job.blob_path.name,
            )
            # Even on the empty path we still want modality / is_medical
            # in the sentinel — a breast US that reads as visually blank
            # to every text-OCR provider should still render downstream
            # as <image modality="ultrasound" is_medical="true" ...> so
            # the LLM-side routing rules in
            # ``detect_disease_from_image_tool`` can fire instead of
            # bailing on the bare tag. The routing provider attaches the
            # full chain_tried + any vision-LLM hint to ``exc.extraction``.
            empty_result = exc.extraction or ExtractResult(
                text="", provider_used="", chain_tried=[]
            )
            vision_tags = await self._compute_vision_tags(job, empty_result)
            blob_store.write_ocr_result(
                job.sha256,
                status="empty",
                kind="ocr",
                ext=job.blob_path.suffix.lstrip("."),
                provider=None,
                chain_tried=list(empty_result.chain_tried),
                reason=str(exc),
                text="",
                original_filename=job.original_filename,
                **vision_tags,
            )
            return OcrCompleted(
                user_id=job.user_id,
                session_id=job.session_id,
                sha256=job.sha256,
                status="empty",
                reason=str(exc),
            )
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
        vision_tags = await self._compute_vision_tags(job, result)
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

    async def _handle_single_image_pdf(
        self, blob_store: BlobStore, job: OcrJob
    ) -> OcrCompleted | None:
        """Try the 1-page image-PDF fast path; return completion or None.

        ``None`` means "this PDF is not a single embedded scan; fall
        through to the normal OCR provider chain". A completion means
        "we wrote ``vision.png`` + computed modality tags + persisted
        the sentinel — caller should return this directly".

        The sentinel goes out with ``status="empty"`` because there is
        no text payload by construction; the rendering layer treats
        empty + ``is_medical=True`` + ``modality=<scan kind>`` as an
        image attachment (the LLM-side routing in
        ``detect_disease_from_image_tool`` keys off those tags, not
        text presence).
        """
        png_bytes = maybe_rasterize_single_image_pdf(job.blob_path)
        if png_bytes is None:
            return None
        blob_dir = blob_store.dir(job.sha256)
        try:
            write_vision_sidecar(blob_dir, png_bytes)
        except OSError as exc:
            # Disk-full / permission errors — log and fall through so
            # MineRU still gets a chance. We don't want a transient FS
            # failure to permanently mark this blob as failed.
            logger.warning(
                "vision.png sidecar write failed for %s: %s; "
                "falling back to OCR provider",
                job.sha256[:8],
                exc,
            )
            return None
        empty_result = ExtractResult(text="", provider_used="", chain_tried=[])
        vision_tags = await self._compute_vision_tags(job, empty_result)
        blob_store.write_ocr_result(
            job.sha256,
            status="empty",
            kind="ocr",
            ext=job.blob_path.suffix.lstrip("."),
            provider="pdf_image_peek",
            chain_tried=["pdf_image_peek"],
            reason="single-image PDF rasterized to vision.png",
            text="",
            original_filename=job.original_filename,
            **vision_tags,
        )
        logger.info(
            "ocr: PDF %s rasterized to vision.png (modality=%s, is_medical=%s)",
            job.sha256[:8],
            vision_tags.get("modality"),
            vision_tags.get("is_medical"),
        )
        return OcrCompleted(
            user_id=job.user_id,
            session_id=job.session_id,
            sha256=job.sha256,
            status="empty",
            provider="pdf_image_peek",
        )

    async def _compute_vision_tags(self, job: OcrJob, result: ExtractResult) -> dict:
        """Build the modality / is_medical / ocr_has_report kwargs.

        Returns the subset of kwargs that should be threaded into
        :meth:`BlobStore.write_ocr_result`. Skips silently for blobs
        without a vision-decodable payload (plain text, multi-page
        PDFs, etc.) so the sentinel stays free of unmeaningful fields.

        The bytes fed to medical-clip come from
        :func:`_resolve_vision_payload`, which prefers a ``vision.png``
        sidecar (PDF-rasterized) over the original ``content.<ext>``.
        That keeps the modality classifier fed even for the 1-page
        image-PDF case that ``content.pdf`` alone could not satisfy.

        When medical-clip returns ``modality='unknown'`` or omits
        ``is_medical``, values supplied by the LLM OCR provider (via
        ``result.modality`` / ``result.is_medical``) are used as a
        fallback — the vision LLM reads both image and text, giving it
        better coverage than the CLIP classifier alone.
        """
        blob_dir = BlobStore(job.user_id).dir(job.sha256)
        payload = _resolve_vision_payload(blob_dir, job.blob_path, job.sha256)
        if payload is None:
            return {}
        image_bytes, payload_sha = payload
        tags: dict = {}
        warnings: list[str] = []
        if self._medical_clip is not None:
            try:
                response = await self._medical_clip.classify_modality(
                    image_bytes,
                    request_id=f"ocr_{job.sha256[:16]}",
                    sha256=payload_sha,
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

        # LLM-OCR override: the vision LLM saw both the pixels and the
        # surrounding text, so it can recover signal medical-clip lost.
        # Two override paths, both monotone (only add medical signal):
        #
        # 1. medical-clip is uncertain (``unknown``/absent) — accept any
        #    non-unknown LLM label, including non-medical (``photo``,
        #    ``document``). Refining "we don't know" to "it's a receipt"
        #    is still useful provenance.
        # 2. medical-clip landed on a non-medical bucket (``photo`` /
        #    ``document``) but the LLM identified a concrete medical
        #    modality AND flags ``is_medical=True``. This is the histopath
        #    failure mode: BiomedCLIP buckets H&E slides as ``document``
        #    even after prompt tuning catches >95% of cases, so the LLM
        #    is the safety net for the long tail. Require the LLM to name
        #    a real modality (not just unknown) so we never trade a
        #    confident non-medical label for a vaguer one.
        llm_modality = result.modality
        llm_is_medical = result.is_medical
        clip_modality = tags.get("modality")
        should_override_modality = False
        if (
            clip_modality in (None, "unknown")
            and llm_modality
            and llm_modality != "unknown"
        ):
            should_override_modality = True
        elif (
            clip_modality in _CLIP_NON_MEDICAL_BUCKETS
            and llm_modality in _LLM_MEDICAL_MODALITIES
            and llm_is_medical is True
        ):
            should_override_modality = True
        if should_override_modality:
            tags["modality"] = llm_modality
            warnings.append("modality_from_llm_ocr")
        if not tags.get("is_medical") and llm_is_medical is True:
            tags["is_medical"] = True
            warnings.append("is_medical_from_llm_ocr")

        # Report-override heuristic. Cheap; always run when configured,
        # regardless of whether modality classification succeeded —
        # ``ocr_has_report`` is independent of modality and the LLM-side
        # branch in Unit 7 reads it before reading modality.
        if self._ocr_report_markers:
            try:
                tags["ocr_has_report"] = has_structured_report(
                    result.text,
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
