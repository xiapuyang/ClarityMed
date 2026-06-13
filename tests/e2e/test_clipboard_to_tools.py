"""End-to-end: clipboard paste → OCR → inline placeholder expansion →
seven ingest tools writing through to the store.

The TUI's Ctrl+V handler, the OcrWorker pipeline, the AttachmentsFeature
plugin, and each of the seven LLM-callable ingest tools each have their
own focused tests. This file stitches them together: one paste, one OCR
pass, one placeholder expansion, then a deterministic walk through every
tool in INGEST_TOOLS.

**OCR is REAL, not mocked.** The PDF leg uses ``PyMuPDFOcrProvider`` —
a core dep (``pymupdf>=1.24``) that ships with every install — against
a PDF built on the fly with ``fitz``. No MinerU, no marker-pdf (PyTorch),
no cloud LLM. This means the test actually proves the local OCR chain
works end-to-end, not just that our wiring around a stub function is
correct. The bare-image-bytes path keeps a stub provider because no
local image-OCR (tesseract / paddle) is in the dependency set today —
adding one would balloon CI install time for marginal extra coverage.

The LLM that would normally invoke the 7 tools is bypassed; we hit each
tool directly via the dispatcher. The user explicitly accepted mocking
the tool-triggering side, since the AskService tool-loop is covered by
the agent suite.

What this test does *not* do: drive Textual key events (covered by the
TUI smoke suite). The paste path is exercised by calling the OCR worker
and SessionAttachments directly, keeping the focus on the data pipeline.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.attachments_feature import AttachmentsFeature
from claritymed.core.features.factory import build_features
from claritymed.core.ocr.base import OcrProvider
from claritymed.core.ocr.pymupdf_provider import PyMuPDFOcrProvider
from claritymed.core.schemas import Allergy, Condition, Medication
from claritymed.orchestrator.features.ingest_tools_plugin import (
    INGEST_TOOLS,
    delete_record,
    save_allergy,
    save_condition,
    save_medication,
    save_record,
    save_to_library,
    update_profile_field,
)
from claritymed.orchestrator.services.ocr_worker import OcrJob, OcrWorker
from claritymed.orchestrator.services.session_attachments import SessionAttachments
from claritymed.orchestrator.services.tool_dispatcher import ToolDispatcher
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.manifest_store import ManifestStore
from claritymed.stores.profile import ProfileStore


_REQUEST_ID = "20260611000000ABCDEF12"
_USER_ID = "e2e"
_SESSION_ID = "sess-e2e-paste"
_OCR_TEXT = (
    "Lab Report — 2024-05-10\n"
    "Patient is on metformin 500mg BID since 2023-01.\n"
    "Allergy: penicillin (severe).\n"
    "Condition: hypertension, onset 2020-03-15."
)


class _StubOcrProvider(OcrProvider):
    """Deterministic OCR — every blob extracts the same canned report.

    Real providers are gated on tesseract / mineru / a cloud LLM; an E2E
    that depends on any of those would flake on CI and on a fresh dev
    box. The stub keeps the data shape (multi-line markdown) so the
    placeholder-expansion path is exercised the same way it would be
    against the production text path."""

    is_local = True
    label = "stub-e2e"

    async def extract_text(self, path: Path):
        from claritymed.core.ocr.base import ExtractResult

        return ExtractResult(
            text=_OCR_TEXT, provider_used=self.label, chain_tried=[self.label]
        )


@pytest.fixture
def _ctx():
    tokens = apply_context(_REQUEST_ID, _USER_ID, "en")
    yield
    reset_context(tokens)


@pytest.fixture
def dispatcher() -> ToolDispatcher:
    return ToolDispatcher()


# ----- paste → blob → OCR → attachments feature ----------------------------


def _make_pdf_with_text(path: Path, text: str) -> None:
    """Write a minimal one-page PDF containing ``text`` as a text layer.

    PyMuPDF reads its own writes verbatim, so this gives the real-OCR
    test a deterministic input without depending on any sample asset
    in the repo (sample assets rot; this builder doesn't).
    """
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), text, fontsize=12)
    doc.save(str(path))
    doc.close()


async def test_pdf_paste_flows_through_real_pymupdf_ocr(_ctx, tmp_path):
    """Build a real PDF, run PyMuPDFOcrProvider against it, verify the
    AttachmentsFeature inlines the actual extracted text into the
    placeholder position.

    This is the load-bearing test for "OCR works": no stub provider,
    no canned response — if PyMuPDF stops shipping or its API breaks,
    this test fails loudly instead of the production paste path
    silently returning empty extracts."""
    pdf_path = tmp_path / "report.pdf"
    pdf_text = (
        "Patient: hypertension since 2020. Currently taking lisinopril 10mg daily."
    )
    _make_pdf_with_text(pdf_path, pdf_text)

    blob_store = BlobStore(_USER_ID)
    sha = blob_store.store(pdf_path.read_bytes(), "pdf")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha,
        filename="report.pdf",
        mime="application/pdf",
        size=pdf_path.stat().st_size,
        source="paste",
    )

    # Real provider. PyMuPDF is a core dep, so this runs in every env
    # the project supports — no skip-on-missing-binary dance.
    worker = OcrWorker(PyMuPDFOcrProvider())
    worker.start()
    try:
        worker.enqueue(
            OcrJob(
                user_id=_USER_ID,
                session_id=_SESSION_ID,
                sha256=sha,
                blob_path=blob_store.path(sha, "pdf"),
            )
        )
        await worker._queue.join()
    finally:
        await worker.stop()

    # Real text round-trips end to end into the inline <file> tag.
    assert blob_store.ocr_done(sha)
    row = SessionAttachments(_USER_ID, _SESSION_ID).get(sha)
    assert row.ocr_status == "done"

    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)

    class _Deps:
        user_id = _USER_ID
        language = "en"

    class _Ctx:
        deps = _Deps()
        scrubbed = ""

    out = await feature.expand_placeholders(f"[File sha:{sha[:8]}]", _Ctx())
    assert f'<file sha="{sha}">' in out
    # The exact text we wrote to the PDF is what the LLM would see in
    # the prompt — strict equality on a substring proves the real OCR
    # path is wired correctly, not just that we got some text back.
    assert "hypertension since 2020" in out
    assert "lisinopril" in out


async def test_paste_image_flows_through_ocr_into_attachments_feature(_ctx):
    """The whole left half of the pipeline for the image-bytes path.

    Uses a stub provider because no local *image*-OCR tool (tesseract,
    paddle, easyocr) is in the dep set; adding one would balloon CI for
    one extra test. The real-OCR coverage is the PDF test above —
    PyMuPDFOcrProvider exercises every stage downstream of the provider
    against a real text extraction. This test focuses on the worker /
    session-attachments / feature wiring rather than OCR fidelity.
    """
    blob_store = BlobStore(_USER_ID)
    sha = blob_store.store(b"fake-image-bytes", "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha,
        filename="clipboard.png",
        mime="image/png",
        size=16,
        source="paste",
    )

    worker = OcrWorker(_StubOcrProvider())
    worker.start()
    try:
        worker.enqueue(
            OcrJob(
                user_id=_USER_ID,
                session_id=_SESSION_ID,
                sha256=sha,
                blob_path=blob_store.path(sha, "png"),
            )
        )
        await worker._queue.join()
    finally:
        await worker.stop()

    # OCR sentinel exists, attachment row was promoted to ``done``.
    assert blob_store.ocr_done(sha)
    row = SessionAttachments(_USER_ID, _SESSION_ID).get(sha)
    assert row is not None
    assert row.ocr_status == "done"

    # Feature inlines the OCR text into the placeholder position; the
    # surrounding user text is preserved verbatim.
    feature = AttachmentsFeature(get_session_id=lambda: _SESSION_ID)

    class _Deps:
        user_id = _USER_ID
        language = "en"

    class _Ctx:
        deps = _Deps()
        scrubbed = ""

    out = await feature.expand_placeholders(
        f"test question [Image sha:{sha[:8]}]", _Ctx()
    )
    assert "test question " in out
    assert f'<image sha="{sha}">' in out
    assert "penicillin" in out  # from _OCR_TEXT


async def test_attachments_feature_skips_when_no_session_id(_ctx):
    """``None`` from the callback disables the feature without errors —
    placeholders pass through unchanged."""
    feature = AttachmentsFeature(get_session_id=lambda: None)

    class _Deps:
        user_id = _USER_ID
        language = "en"

    class _Ctx:
        deps = _Deps()
        scrubbed = ""

    assert await feature.pre_invoke(_Ctx()) == ""
    text = "Hello [Image sha:deadbeef]"
    assert await feature.expand_placeholders(text, _Ctx()) == text


def test_build_features_includes_attachments_when_session_id_provided():
    plugins = build_features(get_session_id=lambda: "sess-x")
    assert "attachments" in {p.name for p in plugins}


# ----- seven tools all write to their stores -------------------------------


def test_seven_ingest_tools_all_persist(dispatcher, _ctx):
    """End-to-end coverage of every tool the LLM can call.

    The contract under test isn't "did the model invoke the right tool"
    (that's the agent's job and is checked by the agent suite); it's
    "given a valid tool call, the side effect lands on the right store
    with the right shape". We hit every tool once, sequenced so each
    later call can reference state the earlier call produced
    (delete_record needs a record to delete)."""
    profile = ProfileStore(_USER_ID)

    # 1. save_medication — onset + end_date round-trip through to row.
    save_medication(
        {
            "name": "metformin",
            "dose": "500mg",
            "frequency": "BID",
            "onset_date": "2023-01-01",
        },
        dispatcher=dispatcher,
    )
    [met] = [m for m in profile.list_medications() if m.display == "metformin"]
    assert isinstance(met, Medication)
    assert met.onset_date.isoformat() == "2023-01-01"
    assert met.end_date is None  # currently taking

    # 2. save_allergy — onset_date round-trip.
    save_allergy(
        {
            "substance": "penicillin",
            "severity": "severe",
            "source": "self_report",
            "onset_date": "2010-05-01",
        },
        dispatcher=dispatcher,
    )
    [pen] = [a for a in profile.list_allergies() if a.substance == "penicillin"]
    assert isinstance(pen, Allergy)
    assert pen.onset_date.isoformat() == "2010-05-01"
    assert pen.end_date is None  # currently active

    # 3. save_condition — onset + end_date.
    save_condition(
        {
            "display": "hypertension",
            "onset_date": "2020-03-15",
        },
        dispatcher=dispatcher,
    )
    [hyp] = [c for c in profile.list_conditions() if c.display == "hypertension"]
    assert isinstance(hyp, Condition)
    assert hyp.onset_date == date(2020, 3, 15)
    assert hyp.end_date is None

    # 4. update_profile_field — proactive field (sex).
    update_profile_field({"field": "sex", "value": "female"}, dispatcher=dispatcher)
    assert profile.get_profile().sex == "female"

    # 5. save_record — needs a blob first.
    blob_store = BlobStore(_USER_ID)
    record_sha = blob_store.store(b"exam report pdf", "pdf")
    record_out = save_record(
        {
            "category": "exam-reports",
            "kind": "exam-report",
            "title": "Annual physical",
            "date": "2024-05-10",
            "attachments": [{"sha256": record_sha, "filename": "annual.pdf"}],
        },
        dispatcher=dispatcher,
    )
    record_path = record_out["record_path"]
    assert record_path.startswith("exam-reports/2024-05-10-")
    cat, slug = record_path.split("/")
    manifest = ManifestStore(_USER_ID, "records").read(cat, slug)
    assert manifest.title == "Annual physical"

    # 6. save_to_library — separate blob, library manifest.
    paper_sha = blob_store.store(b"paper content", "pdf")
    lib_out = save_to_library(
        {
            "title": "NEJM Hypertension Guidelines 2024",
            "attachments": [{"sha256": paper_sha, "filename": "nejm.pdf"}],
            "year": 2024,
            "public": True,
        },
        dispatcher=dispatcher,
    )
    lib_cat, lib_slug = lib_out["library_path"].split("/")
    lib_manifest = ManifestStore(_USER_ID, "library").read(lib_cat, lib_slug)
    assert lib_manifest.public is True
    assert lib_manifest.year == 2024

    # 7. delete_record — the record from step 5 must be gone after.
    delete_record(
        {"record_path": record_path, "confirm_kind": "exam-report"},
        dispatcher=dispatcher,
    )
    from claritymed.errors import RecordNotFound

    with pytest.raises(RecordNotFound):
        ManifestStore(_USER_ID, "records").read(cat, slug)


def test_seven_tools_registry_is_exactly_the_seven():
    """If a tool is added or removed without updating this list, the
    coverage above is no longer "every tool" — fail loudly so future
    contributors update this file alongside ``INGEST_TOOLS``."""
    assert set(INGEST_TOOLS) == {
        "save_record",
        "save_medication",
        "save_allergy",
        "save_condition",
        "update_profile_field",
        "save_to_library",
        "delete_record",
    }
