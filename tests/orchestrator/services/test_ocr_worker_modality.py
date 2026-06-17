"""OCR worker — modality + ocr_has_report tagging (Unit 3).

The worker writes the BiomedCLIP result and the structured-report
heuristic into the same ``ocr.json`` sentinel as the OCR completion
fields. These tests cover the happy path, the unreachable-server
graceful path, the failed-classification path, the non-image skip, and
the legacy path with no medical-clip client wired.

Network is mocked via a tiny stub class with the same async surface as
``MedicalClipClient`` — no httpx, no port. The MedicalClipResponse
shape mirrors ``core.medical_clip.schemas.ModalityResponse``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.medical_clip.schemas import ModalityResponse
from claritymed.core.ocr.base import ExtractResult, OcrEmpty, OcrProvider
from claritymed.errors import MedicalClipUnreachableError
from claritymed.orchestrator.services.ocr_worker import OcrJob, OcrWorker
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.session_attachments import SessionAttachments


# --- helpers / stubs -----------------------------------------------------


class _StubOcrProvider(OcrProvider):
    is_local = True
    label = "stub"

    def __init__(self, text: str = "extracted text") -> None:
        self._text = text

    async def extract_text(self, path: Path) -> ExtractResult:
        return ExtractResult(
            text=self._text, provider_used=self.label, chain_tried=[self.label]
        )


class _StubMedicalClip:
    """Async-compatible MedicalClipClient stand-in.

    The worker calls ``classify_modality`` with positional image bytes
    plus keyword ``request_id`` and ``sha256``. We assert the shape and
    return whatever the test wants — including raising
    ``MedicalClipUnreachableError`` to exercise the graceful fallback.
    """

    def __init__(
        self,
        response: ModalityResponse | None = None,
        *,
        raise_with: Exception | None = None,
    ) -> None:
        self._response = response
        self._raise = raise_with
        self.calls: list[dict] = []

    async def classify_modality(
        self, image_bytes: bytes, *, request_id: str, sha256: str | None = None
    ) -> ModalityResponse:
        self.calls.append(
            {"len": len(image_bytes), "request_id": request_id, "sha256": sha256}
        )
        if self._raise is not None:
            raise self._raise
        assert self._response is not None  # programmer error in test setup
        return self._response


def _ultrasound_response(request_id: str = "ocr_test") -> ModalityResponse:
    return ModalityResponse(
        request_id=request_id,
        modality="ultrasound",
        confidence=0.93,
        is_medical=True,
        scores=[
            {"label": "ultrasound", "score": 0.93},
            {"label": "ct", "score": 0.04},
            {"label": "xray", "score": 0.02},
            {"label": "dermoscopy", "score": 0.005},
            {"label": "photo", "score": 0.003},
            {"label": "document", "score": 0.002},
        ],
        elapsed_ms=48,
    )


# Minimal stand-in for ``configs/vision.yaml::ocr_report``. Real EN/ZH
# marker lists are exercised in test_ocr_report_detector; here we just
# need *something* the worker can pass into has_structured_report.
_REPORT_CFG = {
    "min_chars": 50,
    "markers": {
        "en": ["findings", "impression"],
        "zh": ["所见", "印象"],
    },
}


@pytest.fixture
def _ctx():
    tokens = apply_context("20260615000000ABCDEF12", "test", "en")
    yield
    reset_context(tokens)


async def _enqueue_and_drain(worker: OcrWorker, job: OcrJob) -> None:
    """Run a single job through the worker and wait for it to drain."""
    worker.start()
    worker.enqueue(job)
    await worker._queue.join()
    await worker.stop()


def _seed_image_blob(text_extension: str = "png") -> tuple[BlobStore, str]:
    """Create a tiny image blob the worker can read off disk.

    Returns the BlobStore + sha. The bytes are not a real image — the
    medical-clip stub never decodes them — but the on-disk path has the
    right extension so the worker's image-vs-pdf branch picks the
    classification path.
    """
    bs = BlobStore("test")
    sha = bs.store(b"\x89PNG\r\n\x1a\nfake image bytes", text_extension)
    sa = SessionAttachments("test", "sess-1")
    sa.add(sha256=sha, filename=f"scan.{text_extension}", mime="image/png", size=24)
    return bs, sha


# --- happy path -----------------------------------------------------------


async def test_image_blob_tags_modality_and_is_medical_in_sentinel(_ctx) -> None:
    bs, sha = _seed_image_blob()
    medical_clip = _StubMedicalClip(response=_ultrasound_response())
    worker = OcrWorker(
        _StubOcrProvider("hello world. " * 30),  # > min_chars
        medical_clip_client=medical_clip,
        ocr_report_config=_REPORT_CFG,
    )
    await _enqueue_and_drain(
        worker,
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "png"),
        ),
    )

    # Classifier got called exactly once with the blob bytes + sha threaded.
    assert len(medical_clip.calls) == 1
    call = medical_clip.calls[0]
    assert call["sha256"] == sha
    assert call["len"] > 0

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["modality"] == "ultrasound"
    assert sentinel["modality_confidence"] == pytest.approx(0.93)
    assert sentinel["is_medical"] is True
    # OCR text "hello world. " repeated 30× has no marker keyword, so
    # ocr_has_report stays false even though we cleared the length floor.
    assert sentinel["ocr_has_report"] is False


async def test_report_image_sets_ocr_has_report_true(_ctx) -> None:
    bs, sha = _seed_image_blob()
    medical_clip = _StubMedicalClip(response=_ultrasound_response())
    # OCR text contains FINDINGS + IMPRESSION and clears the 50-char floor.
    report_text = (
        "FINDINGS: lesion in upper-outer quadrant. "
        "IMPRESSION: probable benign cyst. Follow up in 6 months."
    )
    worker = OcrWorker(
        _StubOcrProvider(report_text),
        medical_clip_client=medical_clip,
        ocr_report_config=_REPORT_CFG,
    )
    await _enqueue_and_drain(
        worker,
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "png"),
        ),
    )

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["ocr_has_report"] is True


# --- graceful fallback paths ---------------------------------------------


async def test_unreachable_medical_clip_tags_unknown_and_warns(_ctx) -> None:
    bs, sha = _seed_image_blob()
    medical_clip = _StubMedicalClip(
        raise_with=MedicalClipUnreachableError("connection refused"),
    )
    worker = OcrWorker(
        _StubOcrProvider("OCR text here, long enough to write the sentinel."),
        medical_clip_client=medical_clip,
        ocr_report_config=_REPORT_CFG,
    )
    await _enqueue_and_drain(
        worker,
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "png"),
        ),
    )

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    # OCR still completes — the priority is "extraction must not fail
    # because the side-channel tagger is down."
    assert sentinel["status"] == "done"
    assert sentinel["modality"] == "unknown"
    # is_medical is intentionally absent (not null) — the renderer omits
    # the attribute so the LLM doesn't read a default the classifier
    # never produced.
    assert "is_medical" not in sentinel
    assert any(
        "medical_clip_unreachable" in w for w in sentinel.get("vision_warnings", [])
    )


async def test_classification_unexpected_error_tags_unknown(_ctx) -> None:
    """A 4xx or any other surprise gets the same posture as unreachable."""
    bs, sha = _seed_image_blob()
    medical_clip = _StubMedicalClip(raise_with=RuntimeError("decode oops"))
    worker = OcrWorker(
        _StubOcrProvider("OCR text here, long enough to write the sentinel."),
        medical_clip_client=medical_clip,
        ocr_report_config=_REPORT_CFG,
    )
    await _enqueue_and_drain(
        worker,
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "png"),
        ),
    )

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["modality"] == "unknown"
    assert "is_medical" not in sentinel
    assert any(
        "modality_classification_failed" in w
        for w in sentinel.get("vision_warnings", [])
    )


# --- skipped paths -------------------------------------------------------


async def test_non_image_blob_skips_modality_classification(_ctx) -> None:
    bs = BlobStore("test")
    sha = bs.store(b"%PDF-1.4 fake", "pdf")
    sa = SessionAttachments("test", "sess-1")
    sa.add(sha256=sha, filename="r.pdf", mime="application/pdf", size=12)

    medical_clip = _StubMedicalClip(response=_ultrasound_response())
    worker = OcrWorker(
        _StubOcrProvider("pdf text"),
        medical_clip_client=medical_clip,
        ocr_report_config=_REPORT_CFG,
    )
    await _enqueue_and_drain(
        worker,
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "pdf"),
        ),
    )

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    # Worker never called the classifier (PDFs aren't images), and the
    # sentinel carries none of the vision-tag fields. Field absence is
    # the LLM-side signal "this blob has no modality opinion."
    assert medical_clip.calls == []
    for key in ("modality", "modality_confidence", "is_medical", "ocr_has_report"):
        assert key not in sentinel


class _EmptyOcrProvider(OcrProvider):
    """Raises OcrEmpty, optionally with an LLM-leaf hint attached.

    Mirrors what ``RoutingOcrProvider`` hands the worker when every leaf
    returned empty — chain_tried is populated and any leaf's modality /
    is_medical signal rides along on ``OcrEmpty.extraction``.
    """

    is_local = True
    label = "stub-empty"

    def __init__(self, *, extraction: ExtractResult | None = None) -> None:
        self._extraction = extraction

    async def extract_text(self, path: Path) -> ExtractResult:
        raise OcrEmpty("all providers returned no text", extraction=self._extraction)


async def test_empty_branch_still_classifies_modality(_ctx) -> None:
    """OcrEmpty image: medical-clip must still run + tag the sentinel.

    Regression for the breast-US dead-image case: an image with no
    extractable text used to skip ``_compute_vision_tags`` entirely, so
    the rendered ``<image>`` tag was bare and the LLM-side routing rules
    had nothing to fire on. After the fix, modality / is_medical land in
    ``ocr.json`` regardless of OCR text presence.
    """
    bs, sha = _seed_image_blob()
    medical_clip = _StubMedicalClip(response=_ultrasound_response())
    routing_hint = ExtractResult(
        text="", provider_used="rapidocr", chain_tried=["llm", "rapidocr"]
    )
    worker = OcrWorker(
        _EmptyOcrProvider(extraction=routing_hint),
        medical_clip_client=medical_clip,
        ocr_report_config=_REPORT_CFG,
    )
    await _enqueue_and_drain(
        worker,
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "png"),
        ),
    )

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "empty"
    # Classifier ran exactly once on the empty branch — the whole point.
    assert len(medical_clip.calls) == 1
    assert sentinel["modality"] == "ultrasound"
    assert sentinel["is_medical"] is True
    # chain_tried comes from the routing hint, not a hardcoded "[]".
    assert sentinel["chain_tried"] == ["llm", "rapidocr"]


async def test_empty_branch_carries_llm_hint_when_clip_unreachable(_ctx) -> None:
    """When medical-clip is down, the LLM-leaf hint still flows into ocr.json.

    Defense in depth: the LLM OCR provider had read the pixels and tagged
    modality="ultrasound" / is_medical=True before reporting "no text".
    Even with medical-clip 503, that signal must survive so the downstream
    ``<image>`` tag carries real attrs instead of degrading to bare.
    """
    bs, sha = _seed_image_blob()
    medical_clip = _StubMedicalClip(
        raise_with=MedicalClipUnreachableError("connection refused"),
    )
    llm_hint = ExtractResult(
        text="",
        provider_used="llm",
        chain_tried=["llm"],
        modality="ultrasound",
        is_medical=True,
    )
    worker = OcrWorker(
        _EmptyOcrProvider(extraction=llm_hint),
        medical_clip_client=medical_clip,
        ocr_report_config=_REPORT_CFG,
    )
    await _enqueue_and_drain(
        worker,
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "png"),
        ),
    )

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "empty"
    # The unreachable branch sets modality="unknown", then the LLM override
    # promotes it to "ultrasound" because the LLM tagged a concrete medical
    # modality and is_medical=True.
    assert sentinel["modality"] == "ultrasound"
    assert sentinel["is_medical"] is True
    assert sentinel["chain_tried"] == ["llm"]


async def test_no_medical_clip_client_leaves_legacy_sentinel(_ctx) -> None:
    """Back-compat: worker constructed without the new kwargs keeps working."""
    bs, sha = _seed_image_blob()
    worker = OcrWorker(_StubOcrProvider("legacy text long enough to write."))
    await _enqueue_and_drain(
        worker,
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "png"),
        ),
    )

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "done"
    # None of the new fields landed in the sentinel.
    for key in ("modality", "modality_confidence", "is_medical", "ocr_has_report"):
        assert key not in sentinel
