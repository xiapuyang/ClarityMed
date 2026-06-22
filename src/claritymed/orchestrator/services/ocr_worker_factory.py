"""Shared :class:`OcrWorker` constructor used by TUI and web hosts.

Both hosts need the same provider chain (OCR provider + medical-clip
client + ocr_report_config) for OCR background processing. The only
difference is the per-host completion listener — TUI hooks into Textual
toast/UI updates; web hosts a Server-Sent-Events bridge (currently
``None``, deferred to a later web feature).

The factory returns ``None`` on construction failure (matching both
call sites' graceful-degradation contract): a missing medical-clip
server or OCR provider misconfig should leave the upload path working
with attachments stuck in ``pending`` state — never a 500 / crash.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claritymed.orchestrator.services.ocr_worker import (
        CompletionListener,
        OcrWorker,
    )

logger = logging.getLogger(__name__)


def build_ocr_worker(
    *,
    listener: "CompletionListener | None" = None,
    start: bool = True,
) -> "OcrWorker | None":
    """Build (and optionally start) an :class:`OcrWorker`.

    Args:
        listener: Per-host completion callback. ``None`` means
            fire-and-forget — the worker still completes the job and
            writes the cached OCR result, but no UI sees the
            ``OcrCompleted`` event. TUI passes its Textual listener;
            web currently passes ``None`` and reads results via the
            attachments router cache.
        start: Whether to call ``worker.start()`` before returning.
            Default ``True`` matches both existing call sites.

    Returns ``None`` if provider construction raises (medical-clip
    config malformed, OCR factory unhappy, etc.). The caller decides
    how to surface that — TUI toasts, web responds 200 with the
    attachment in ``pending`` state.
    """
    try:
        from claritymed.config import CONFIGS_DIR, load_yaml
        from claritymed.core.medical_clip.client import (
            DEFAULT_BASE_URL as MEDICAL_CLIP_DEFAULT_BASE_URL,
            MedicalClipClient,
        )
        from claritymed.core.ocr.factory import make_ocr_provider
        from claritymed.core.vision.ocr_report_detector import (
            load_ocr_report_config,
        )
        from claritymed.orchestrator.services.ocr_worker import OcrWorker

        provider = make_ocr_provider()
        # Medical-clip client construction is unconditional; the server
        # may not be running, but ``classify_modality`` raises
        # ``MedicalClipUnreachableError`` which the worker catches and
        # tags ``modality=unknown``. Building the client up-front (vs
        # lazily) keeps ``OcrWorker`` free of base_url plumbing.
        medical_clip_base_url = (
            load_yaml("app.yaml")
            .get("medical_clip", {})
            .get("base_url", MEDICAL_CLIP_DEFAULT_BASE_URL)
        )
        medical_clip_client = MedicalClipClient(base_url=medical_clip_base_url)
        ocr_report_config = load_ocr_report_config(CONFIGS_DIR / "vision.yaml")
        worker = OcrWorker(
            provider,
            listener=listener,
            medical_clip_client=medical_clip_client,
            ocr_report_config=ocr_report_config,
        )
        if start:
            worker.start()
        return worker
    except Exception:  # noqa: BLE001
        logger.exception("ocr worker build failed; attachments will pend")
        return None


__all__ = ["build_ocr_worker"]
