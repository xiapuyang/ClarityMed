"""E2E: 1-page image-only PDF → vision pipeline (stage 1).

What this covers
----------------

When the OCR worker sees a PDF whose single page is just one embedded
raster image (e.g. a CT slice exported as a PDF wrapper), it must:

1. Rasterize the page to PNG via PyMuPDF (no MineRU round-trip).
2. Write ``vision.png`` next to ``content.pdf`` in the blob dir.
3. Run medical-clip on the PNG (NOT the PDF bytes — that would 4xx).
4. Persist an ``ocr.json`` sentinel with ``status="empty"``,
   ``provider="pdf_image_peek"``, and the modality/is_medical tags
   medical-clip returned.
5. Make :func:`vision_plugin._read_blob_bytes` return the PNG bytes +
   a sha that matches them, so the vision server's hash check passes.

The test exercises the **production wiring** end-to-end — same
``OcrWorker`` class, same :class:`MedicalClipClient` over loopback HTTP
— so regressions in any of the four files involved (``pdf_image_peek``,
``ocr_worker``, ``vision_plugin._read_blob_bytes``, ``BlobStore`` blob
layout) are caught before they ship.

Skip semantics
--------------

* The fixture PDF lives at ``~/Downloads/000108 (3).pdf`` on the
  maintainer's box. Test skips cleanly if it isn't there — open-source
  CI doesn't have this file by design (it may be PHI).
* medical-clip-server must be reachable on the production port. If
  not, the "tags populated" assertions skip and only the
  rasterization + sidecar assertions run.

Run with
--------

::

    uv run pytest tests/e2e/test_pdf_image_peek_e2e.py -v --no-cov
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

import httpx
import pytest

from claritymed.core.medical_clip.client import MedicalClipClient
from claritymed.core.ocr.base import ExtractResult, OcrEmpty, OcrProvider
from claritymed.core.ocr.pdf_image_peek import maybe_rasterize_single_image_pdf
from claritymed.orchestrator.features.vision_plugin import _read_blob_bytes
from claritymed.orchestrator.services.ocr_worker import OcrJob, OcrWorker
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.session_attachments import SessionAttachments

logger = logging.getLogger(__name__)

# Fixture PDF — a single CT slice exported as a 1-page PDF wrapper.
# Lives outside the repo because it may carry PHI; tests skip cleanly
# when it isn't present (open-source CI, fresh checkouts).
_FIXTURE_PDF = Path.home() / "Downloads" / "000108 (3).pdf"

# Production-port medical-clip; same constant as
# ``test_medical_clip_modality_e2e`` so a contributor doesn't have to
# learn two ports.
_MEDICAL_CLIP_BASE_URL = "http://127.0.0.1:8086"

# Fixed e2e user id matches the project convention (see CLAUDE.md
# "Test user_id convention"). The autouse session-scope wipe in
# ``tests/e2e/conftest.py`` clears ``data/users/e2e/`` once at start,
# so seeded blobs accumulate within a run but never across runs.
_USER_ID = "e2e"
_SESSION_ID = "pdf-image-peek"


@pytest.fixture(scope="module")
def _fixture_pdf_bytes() -> bytes:
    if not _FIXTURE_PDF.exists():
        pytest.skip(
            f"fixture PDF not found at {_FIXTURE_PDF}; this test depends on "
            "a 1-page image-only PDF outside the repo. Drop the file at "
            "that path to run the test."
        )
    return _FIXTURE_PDF.read_bytes()


def _medical_clip_reachable() -> bool:
    try:
        return (
            httpx.get(f"{_MEDICAL_CLIP_BASE_URL}/health", timeout=2.0).status_code
            == 200
        )
    except httpx.HTTPError:
        return False


class _NoOcrProvider(OcrProvider):
    """OCR provider that always raises ``OcrEmpty``.

    The PDF-peek path short-circuits before reaching the provider, so
    the test never actually invokes ``extract_text``. We still need a
    concrete provider to construct the worker, and a raising stub makes
    a regression (peek didn't fire → MineRU dispatched) loud rather
    than silent-with-empty-text.
    """

    label = "noop-for-peek-test"
    supported_extensions = frozenset({".pdf", ".png"})

    async def extract_text(self, path: Path) -> ExtractResult:
        raise OcrEmpty(
            f"_NoOcrProvider called for {path.name} — the PDF peek "
            "fast-path should have short-circuited"
        )


# ---------------------------------------------------------------------
# 1. The pure helper: rasterize bytes into PNG
# ---------------------------------------------------------------------


def test_maybe_rasterize_returns_png_for_single_image_pdf(
    _fixture_pdf_bytes: bytes,
) -> None:
    """``maybe_rasterize_single_image_pdf`` returns a valid PNG."""
    png_bytes = maybe_rasterize_single_image_pdf(_FIXTURE_PDF)
    assert png_bytes is not None, (
        "fixture PDF was expected to qualify (1 page / 0 text / ≥1 image); "
        "got None — was the file replaced with something else?"
    )
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "missing PNG signature"
    # 200 DPI rasterization of a ~381×282 source page lands in the
    # tens-of-KB range; both ends are sanity-checked so a zero-byte
    # write or a runaway 10 MB rasterization both fail loudly.
    assert 1_000 < len(png_bytes) < 5_000_000, (
        f"rasterized PNG size suspicious: {len(png_bytes)} bytes"
    )


# ---------------------------------------------------------------------
# 2. Multi-page / text PDF returns None (the negative path)
# ---------------------------------------------------------------------


def test_maybe_rasterize_returns_none_for_text_pdf(tmp_path: Path) -> None:
    """Text-bearing PDFs fall through to the existing OCR path."""
    reportlab = pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas  # noqa: PLC0415

    p = tmp_path / "report.pdf"
    c = canvas.Canvas(str(p))
    c.drawString(100, 750, "FINDINGS: chest CT shows no acute abnormality.")
    c.drawString(100, 730, "IMPRESSION: unremarkable study.")
    c.save()

    assert maybe_rasterize_single_image_pdf(p) is None, (
        "text-bearing PDF should not trigger the image fast-path; got bytes"
    )
    # Keep reportlab reference used so the importorskip isn't dead.
    _ = reportlab


# ---------------------------------------------------------------------
# 3. Full worker integration — sentinel + vision tags
# ---------------------------------------------------------------------


async def _drive_one_job(worker: OcrWorker, job: OcrJob) -> None:
    """Enqueue a job and wait for the listener to fire."""
    done = asyncio.Event()

    def _on_done(_completion) -> None:
        done.set()

    worker._listener = _on_done  # type: ignore[assignment]
    worker.start()
    try:
        worker.enqueue(job)
        await asyncio.wait_for(done.wait(), timeout=30.0)
    finally:
        await worker.stop()


async def test_ocr_worker_routes_single_image_pdf_through_vision(
    _fixture_pdf_bytes: bytes,
) -> None:
    """End-to-end: enqueue the PDF, observe sidecar + tagged sentinel."""
    blob_store = BlobStore(_USER_ID)
    sha = blob_store.store(_fixture_pdf_bytes, "pdf")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha,
        filename=_FIXTURE_PDF.name,
        mime="application/pdf",
        size=len(_fixture_pdf_bytes),
    )

    medical_clip_client = (
        MedicalClipClient(base_url=_MEDICAL_CLIP_BASE_URL)
        if _medical_clip_reachable()
        else None
    )
    worker = OcrWorker(
        _NoOcrProvider(),
        medical_clip_client=medical_clip_client,
    )
    try:
        await _drive_one_job(
            worker,
            OcrJob(
                user_id=_USER_ID,
                session_id=_SESSION_ID,
                sha256=sha,
                blob_path=blob_store.path(sha, "pdf"),
                original_filename=_FIXTURE_PDF.name,
            ),
        )
    finally:
        if medical_clip_client is not None:
            await medical_clip_client.aclose()

    # --- sidecar landed on disk ---------------------------------------
    blob_dir = blob_store.dir(sha)
    sidecar = blob_dir / "vision.png"
    assert sidecar.exists(), (
        f"expected vision.png sidecar in {blob_dir}; got "
        f"{sorted(p.name for p in blob_dir.iterdir())}"
    )
    assert sidecar.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # --- sentinel reflects the short-circuit --------------------------
    meta = blob_store.read_ocr_metadata(sha)
    assert meta is not None, "ocr.json sentinel missing after worker run"
    assert meta["status"] == "empty"
    assert meta["provider"] == "pdf_image_peek", (
        f"expected provider=pdf_image_peek (peek fast-path); got {meta['provider']!r}"
    )
    assert meta["chain_tried"] == ["pdf_image_peek"]

    if medical_clip_client is not None:
        # medical-clip ran on the PNG bytes — tags must be present.
        assert "modality" in meta, f"vision tags missing from sentinel: {sorted(meta)}"
        # We don't pin the specific modality (BiomedCLIP's exact bucket
        # for an arbitrary CT slice is data-dependent), but it must NOT
        # be the wire-error fallback that means "we tried to classify
        # the PDF bytes" — those crash medical-clip with 4xx and the
        # worker tags ``unknown`` + emits ``medical_clip_unreachable``
        # / ``modality_classification_failed`` warnings.
        assert "medical_clip_unreachable" not in " ".join(
            meta.get("vision_warnings") or []
        )
        assert "modality_classification_failed" not in " ".join(
            meta.get("vision_warnings") or []
        )
        logger.info(
            "PDF→vision e2e tags: modality=%s is_medical=%s confidence=%s",
            meta.get("modality"),
            meta.get("is_medical"),
            meta.get("modality_confidence"),
        )

    # --- vision_plugin sees the PNG, with a sha that matches ----------
    image_bytes, wire_sha = _read_blob_bytes(_USER_ID, sha)
    assert image_bytes[:8] == b"\x89PNG\r\n\x1a\n", (
        "_read_blob_bytes returned non-PNG bytes; vision server would 400"
    )
    assert wire_sha == hashlib.sha256(image_bytes).hexdigest()
    # And critically: the wire sha must NOT equal the PDF's sha, or
    # the vision server would reject ``image_hash_mismatch``.
    assert wire_sha != sha
