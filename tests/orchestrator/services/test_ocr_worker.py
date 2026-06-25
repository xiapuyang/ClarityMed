"""Tests for ``OcrWorker``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.ocr.base import ExtractResult, OcrEmpty, OcrError, OcrProvider
from claritymed.orchestrator.services.ocr_worker import (
    OcrCompleted,
    OcrJob,
    OcrWorker,
)
from claritymed.orchestrator.services.session_attachments import SessionAttachments
from claritymed.stores.blob_store import BlobStore


class _StubProvider(OcrProvider):
    is_local = True
    label = "stub"

    def __init__(
        self,
        *,
        text: str = "extracted text",
        raise_with: str | None = None,
        raise_empty: bool = False,
    ):
        self._text = text
        self._raise = raise_with
        self._raise_empty = raise_empty

    async def extract_text(self, path: Path) -> ExtractResult:
        if self._raise_empty:
            raise OcrEmpty("no text in image")
        if self._raise:
            raise OcrError(self._raise)
        return ExtractResult(
            text=self._text, provider_used=self.label, chain_tried=[self.label]
        )


@pytest.fixture
def _ctx():
    tokens = apply_context("20260611000000ABCDEF12", "test", "en")
    yield
    reset_context(tokens)


async def test_extract_done_writes_sentinel(_ctx):
    bs = BlobStore("test")
    sha = bs.store(b"some bytes", "pdf")
    sa = SessionAttachments("test", "sess-1")
    sa.add(sha256=sha, filename="r.pdf", mime="application/pdf", size=10)

    completions: list[OcrCompleted] = []

    async def _listen(c: OcrCompleted) -> None:
        completions.append(c)

    worker = OcrWorker(_StubProvider(text="hello"), listener=_listen)
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "pdf"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    assert bs.ocr_done(sha)
    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "done"
    # Sentinel records the real leaf provider + the chain that was tried,
    # not the wrapping RoutingOcrProvider class name. Regression guard:
    # previously this surfaced as "RoutingOcrProvider" in ocr.json.
    assert sentinel["provider"] == "stub"
    assert sentinel["chain_tried"] == ["stub"]
    # Worker records kind="ocr" + the source extension so the unified
    # reader knows to look at ocr.md (not content.pdf).
    assert sentinel["kind"] == "ocr"
    assert sentinel["ext"] == "pdf"
    assert completions and completions[0].status == "done"
    assert completions[0].provider == "stub"
    # Row was updated in the session tray.
    row = sa.get(sha)
    assert row is not None
    assert row.ocr_status == "done"


async def test_empty_text_marks_empty(_ctx):
    bs = BlobStore("test")
    sha = bs.store(b"some bytes", "pdf")
    sa = SessionAttachments("test", "sess-1")
    sa.add(sha256=sha, filename="r.pdf", mime="application/pdf", size=10)

    completions: list[OcrCompleted] = []
    worker = OcrWorker(_StubProvider(text=""), listener=lambda c: completions.append(c))
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "pdf"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "empty"
    assert completions[0].status == "empty"


async def test_provider_failure_marks_failed(_ctx):
    bs = BlobStore("test")
    sha = bs.store(b"some bytes", "pdf")
    sa = SessionAttachments("test", "sess-1")
    sa.add(sha256=sha, filename="r.pdf", mime="application/pdf", size=10)

    completions: list[OcrCompleted] = []
    worker = OcrWorker(
        _StubProvider(raise_with="provider go boom"),
        listener=lambda c: completions.append(c),
    )
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "pdf"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "failed"
    assert completions[0].status == "failed"
    row = sa.get(sha)
    assert row.ocr_status == "failed"


async def test_cache_short_circuit_when_sentinel_present(_ctx):
    """An ``ocr.json`` already on disk → worker skips extraction and emits
    a synthetic ``done`` event with provider='cache'."""
    bs = BlobStore("test")
    sha = bs.store(b"x", "txt")
    bs.ocr_path(sha).write_text("cached text", encoding="utf-8")
    bs.ocr_meta_path(sha).write_text(
        json.dumps({"status": "done", "provider": "pre-existing"}),
        encoding="utf-8",
    )

    completions: list[OcrCompleted] = []
    # Stub will raise if it's actually called — proving the cache hit.
    worker = OcrWorker(
        _StubProvider(raise_with="should not call"),
        listener=lambda c: completions.append(c),
    )
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "txt"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    assert completions[0].status == "done"
    assert completions[0].provider == "cache"


async def test_cache_propagates_empty_status(_ctx):
    """A cached ``empty`` sentinel must not be reported as ``done``.

    The old worker synthesized status=done on any cache hit, which lied
    about both empty and failed results. The empty case must keep its
    real status so the UI can show "no text extracted" instead of "ok".

    Empties from post-fix code always carry a non-empty ``chain_tried``
    (the routing provider populates it from the leaves it walked). The
    sentinel below uses that real shape; an empty ``chain_tried`` would
    trip the legacy-empty retry probe and re-extract — see
    ``test_cached_legacy_empty_triggers_retry`` for that path.
    """
    bs = BlobStore("test")
    sha = bs.store(b"x", "txt")
    bs.ocr_path(sha).write_text("", encoding="utf-8")
    bs.ocr_meta_path(sha).write_text(
        json.dumps(
            {
                "status": "empty",
                "provider": "pre-existing",
                "chain_tried": ["pre-existing"],
            }
        ),
        encoding="utf-8",
    )

    completions: list[OcrCompleted] = []
    worker = OcrWorker(
        _StubProvider(raise_with="should not call"),
        listener=lambda c: completions.append(c),
    )
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "txt"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    assert completions[0].status == "empty"
    assert completions[0].provider == "cache"


async def test_cached_failure_triggers_retry(_ctx):
    """A cached ``failed`` sentinel must NOT short-circuit.

    Otherwise a single bad run (e.g. provider env var not set) keeps the
    sha permanently marked failed even after the cause is fixed — the
    user has no way to retry short of manually deleting the sentinel.
    """
    bs = BlobStore("test")
    sha = bs.store(b"x", "pdf")
    bs.ocr_path(sha).write_text("", encoding="utf-8")
    bs.ocr_meta_path(sha).write_text(
        json.dumps(
            {
                "status": "failed",
                "provider": None,
                "reason": "no OCR providers available for .pdf",
                "chars": 0,
            }
        ),
        encoding="utf-8",
    )

    completions: list[OcrCompleted] = []
    worker = OcrWorker(
        _StubProvider(text="recovered text"),
        listener=lambda c: completions.append(c),
    )
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "pdf"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    assert completions[0].status == "done"
    assert completions[0].provider != "cache"
    # Sentinel overwritten with the fresh result.
    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "done"
    assert sentinel["chars"] == len("recovered text")


async def test_cached_legacy_empty_triggers_retry(_ctx):
    """A pre-fix ``empty`` sentinel (chain_tried=[]) must NOT short-circuit.

    Pre-fix OcrWorker code hardcoded ``chain_tried=[]`` on the empty path
    and skipped modality classification entirely. Those sentinels render
    as bare ``<image>`` tags and permanently break the LLM-side routing
    rules — the user's image goes "dead" across every future session.

    The post-fix writer always records the real chain it walked, so an
    empty ``chain_tried`` on an ``empty`` sentinel is the unambiguous
    "legacy buggy shape" probe. Re-extracting once promotes the sentinel
    into the new shape with proper modality / is_medical tags.
    """
    bs = BlobStore("test")
    sha = bs.store(b"\x89PNG", "png")
    sa = SessionAttachments("test", "sess-1")
    sa.add(sha256=sha, filename="scan.png", mime="image/png", size=4)
    # Hand-craft the legacy sentinel shape — what the buggy worker wrote
    # before the fix. Note chain_tried=[] but reason text mentions the
    # leaves that ran; this exact shape was reported in the original bug.
    bs.ocr_meta_path(sha).write_text(
        json.dumps(
            {
                "status": "empty",
                "kind": "ocr",
                "ext": "png",
                "provider": None,
                "chain_tried": [],
                "reason": "all providers returned no text (['llm', 'rapidocr'])",
                "chars": 0,
            }
        ),
        encoding="utf-8",
    )

    completions: list[OcrCompleted] = []
    worker = OcrWorker(
        _StubProvider(text="recovered text"),
        listener=lambda c: completions.append(c),
    )
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "png"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    # Worker treated the legacy sentinel as a miss and re-ran the provider.
    assert completions[0].provider != "cache"
    assert completions[0].status == "done"
    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "done"
    assert sentinel["chain_tried"] == ["stub"]


async def test_cached_new_empty_short_circuits(_ctx):
    """A post-fix ``empty`` sentinel (chain_tried populated) IS a cache hit.

    Once the worker has written a real chain into ``ocr.json`` the empty
    result is authoritative — re-pasting the same image must not pay the
    OCR cost again. This is the complement of the legacy-empty retry: the
    retry probe must be narrow enough to leave good empties alone.
    """
    bs = BlobStore("test")
    sha = bs.store(b"\x89PNG", "png")
    bs.ocr_meta_path(sha).write_text(
        json.dumps(
            {
                "status": "empty",
                "kind": "ocr",
                "ext": "png",
                "provider": None,
                "chain_tried": ["llm", "rapidocr"],
                "reason": "all providers returned no text (['llm', 'rapidocr'])",
                "modality": "ultrasound",
                "is_medical": True,
                "chars": 0,
            }
        ),
        encoding="utf-8",
    )

    completions: list[OcrCompleted] = []
    # Provider would raise if called — proving the cache short-circuit fired.
    worker = OcrWorker(
        _StubProvider(raise_with="should not call"),
        listener=lambda c: completions.append(c),
    )
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "png"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    assert completions[0].status == "empty"
    assert completions[0].provider == "cache"


class _HangingProvider(OcrProvider):
    """OcrProvider that blocks forever inside ``extract_text``.

    Used to verify the worker's per-job wall-clock timeout fires and
    leaves the queue ready for the next job. Without the timeout this
    provider would hold the queue indefinitely.
    """

    is_local = True
    label = "hanging"

    async def extract_text(self, path: Path):
        import asyncio as _aio

        await _aio.Event().wait()  # never set → blocks forever
        raise AssertionError("unreachable")  # pragma: no cover


async def test_ocr_worker_job_timeout_writes_failed_sentinel(_ctx, monkeypatch):
    """ADV-005 regression: a hung OCR provider must not block the queue.

    The worker enforces ``OCR_JOB_TIMEOUT_S`` via ``asyncio.wait_for``;
    on timeout it persists a ``failed`` sentinel (so the blob isn't
    stuck in ``pending`` across restarts) and the loop continues.
    """
    from claritymed.orchestrator.services import ocr_worker as ow_mod

    monkeypatch.setattr(ow_mod, "OCR_JOB_TIMEOUT_S", 0.1)

    bs = BlobStore("test")
    sha = bs.store(b"will hang", "pdf")
    sa = SessionAttachments("test", "sess-1")
    sa.add(sha256=sha, filename="hang.pdf", mime="application/pdf", size=9)

    completions: list[OcrCompleted] = []

    async def _listen(c: OcrCompleted) -> None:
        completions.append(c)

    worker = OcrWorker(_HangingProvider(), listener=_listen)
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "pdf"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "failed"
    assert "timed out" in sentinel["reason"]
    assert completions and completions[0].status == "failed"
    row = sa.get(sha)
    assert row is not None and row.ocr_status == "failed"


async def test_ocr_worker_timeout_does_not_block_next_job(_ctx, monkeypatch):
    """A timed-out job must not stall subsequent jobs in the queue.

    Mixed workload: first job hangs (will time out), second is fast.
    The fast job must complete in well under the timeout-recovery
    horizon — proves the loop returns to ``get()`` after the timeout
    path rather than serializing on the hung extract.
    """
    from claritymed.orchestrator.services import ocr_worker as ow_mod

    monkeypatch.setattr(ow_mod, "OCR_JOB_TIMEOUT_S", 0.1)

    bs = BlobStore("test")
    sha_hang = bs.store(b"hang one", "pdf")
    sha_ok = bs.store(b"hang two ok", "pdf")
    sa = SessionAttachments("test", "sess-1")
    sa.add(sha256=sha_hang, filename="a.pdf", mime="application/pdf", size=8)
    sa.add(sha256=sha_ok, filename="b.pdf", mime="application/pdf", size=11)

    # Provider that hangs on the first job and succeeds on the second.
    class _MixedProvider(OcrProvider):
        is_local = True
        label = "mixed"

        def __init__(self) -> None:
            self._calls = 0

        async def extract_text(self, path: Path):
            self._calls += 1
            if self._calls == 1:
                import asyncio as _aio

                await _aio.Event().wait()
                raise AssertionError("unreachable")  # pragma: no cover
            return ExtractResult(
                text="second job ok",
                provider_used=self.label,
                chain_tried=[self.label],
            )

    completions: list[OcrCompleted] = []
    worker = OcrWorker(_MixedProvider(), listener=lambda c: completions.append(c))
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha_hang,
            blob_path=bs.path(sha_hang, "pdf"),
        )
    )
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha_ok,
            blob_path=bs.path(sha_ok, "pdf"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    s_hang = json.loads(bs.ocr_meta_path(sha_hang).read_text(encoding="utf-8"))
    s_ok = json.loads(bs.ocr_meta_path(sha_ok).read_text(encoding="utf-8"))
    assert s_hang["status"] == "failed"
    assert s_ok["status"] == "done"
    assert {c.status for c in completions} == {"failed", "done"}


async def test_ocr_empty_marks_empty_not_failed(_ctx):
    """OcrEmpty from the provider writes status='empty', not 'failed'.

    Empty images (no extractable text) are not an error — the sentinel
    must distinguish them so the UI can show "no text found" instead of
    "extraction failed".
    """
    bs = BlobStore("test")
    sha = bs.store(b"\x89PNG", "png")
    sa = SessionAttachments("test", "sess-1")
    sa.add(sha256=sha, filename="blank.png", mime="image/png", size=4)

    completions: list[OcrCompleted] = []
    worker = OcrWorker(
        _StubProvider(raise_empty=True),
        listener=lambda c: completions.append(c),
    )
    worker.start()
    worker.enqueue(
        OcrJob(
            user_id="test",
            session_id="sess-1",
            sha256=sha,
            blob_path=bs.path(sha, "png"),
        )
    )
    await worker._queue.join()
    await worker.stop()

    sentinel = json.loads(bs.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "empty"
    assert completions[0].status == "empty"
    # Empty is NOT an error — the session row should reflect that.
    row = sa.get(sha)
    assert row is not None and row.ocr_status == "empty"


class _LLMStubProvider(OcrProvider):
    """Stub that returns ExtractResult with optional modality/is_medical."""

    is_local = True
    label = "stub"

    def __init__(
        self,
        text: str = "report",
        modality: str | None = None,
        is_medical: bool | None = None,
    ):
        self._text = text
        self._modality = modality
        self._is_medical = is_medical

    async def extract_text(self, path: Path) -> ExtractResult:
        return ExtractResult(
            text=self._text,
            provider_used=self.label,
            chain_tried=[self.label],
            modality=self._modality,
            is_medical=self._is_medical,
        )


@pytest.mark.asyncio
async def test_compute_vision_tags_llm_override_when_clip_unknown(tmp_path):
    """LLM modality/is_medical override medical-clip 'unknown' result."""
    from unittest.mock import AsyncMock, MagicMock

    from claritymed.core.ocr.base import ExtractResult
    from claritymed.errors import MedicalClipUnreachableError

    # Create a real PNG file so _is_image returns True
    img = tmp_path / "ct.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    fake_clip = MagicMock()
    # medical-clip is unreachable → tags["modality"] = "unknown", no is_medical
    fake_clip.classify_modality = AsyncMock(
        side_effect=MedicalClipUnreachableError("down")
    )

    worker = OcrWorker(
        _LLMStubProvider(),
        medical_clip_client=fake_clip,
    )
    job = OcrJob(user_id="test", session_id="s", sha256="ab" * 32, blob_path=img)
    result = ExtractResult(
        text="CT头颅平扫报告",
        provider_used="stub",
        chain_tried=["stub"],
        modality="ct",
        is_medical=True,
    )
    tags = await worker._compute_vision_tags(job, result)

    assert tags["modality"] == "ct"
    assert tags["is_medical"] is True
    assert "modality_from_llm_ocr_fallback" in tags.get("vision_warnings", [])


@pytest.mark.asyncio
async def test_compute_vision_tags_clip_result_not_overridden_when_confident(tmp_path):
    """Confident medical-clip result is NOT overridden even if LLM disagrees."""
    from unittest.mock import AsyncMock, MagicMock

    from claritymed.core.ocr.base import ExtractResult

    img = tmp_path / "us.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    clip_response = MagicMock()
    clip_response.modality = "ultrasound"
    clip_response.confidence = 0.95
    clip_response.is_medical = True

    fake_clip = MagicMock()
    fake_clip.classify_modality = AsyncMock(return_value=clip_response)

    worker = OcrWorker(_LLMStubProvider(), medical_clip_client=fake_clip)
    job = OcrJob(user_id="test", session_id="s", sha256="ab" * 32, blob_path=img)
    result = ExtractResult(
        text="B超: 肝胆脾胰肾未见异常",
        provider_used="stub",
        chain_tried=["stub"],
        modality="ct",  # LLM says ct — should NOT override confident clip
        is_medical=True,
    )
    tags = await worker._compute_vision_tags(job, result)

    assert tags["modality"] == "ultrasound"
    assert "modality_from_llm_ocr" not in tags.get("vision_warnings", [])


@pytest.mark.asyncio
async def test_compute_vision_tags_llm_override_when_clip_document_but_llm_histopath(
    tmp_path,
):
    """medical-clip drops H&E slides into ``document``; LLM-OCR must rescue."""
    from unittest.mock import AsyncMock, MagicMock

    from claritymed.core.ocr.base import ExtractResult

    img = tmp_path / "slide.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    clip_response = MagicMock()
    clip_response.modality = "document"
    clip_response.confidence = 0.78
    clip_response.is_medical = False  # gating forces False on photo/document/unknown

    fake_clip = MagicMock()
    fake_clip.classify_modality = AsyncMock(return_value=clip_response)

    worker = OcrWorker(
        _LLMStubProvider(modality="histopathology", is_medical=True),
        medical_clip_client=fake_clip,
    )
    job = OcrJob(user_id="test", session_id="s", sha256="ab" * 32, blob_path=img)
    result = ExtractResult(
        text="",
        provider_used="stub",
        chain_tried=["stub"],
        modality="histopathology",
        is_medical=True,
    )
    tags = await worker._compute_vision_tags(job, result)

    assert tags["modality"] == "histopathology"
    assert tags["is_medical"] is True
    assert "modality_from_llm_ocr" in tags.get("vision_warnings", [])
    assert "is_medical_from_llm_ocr" in tags.get("vision_warnings", [])


@pytest.mark.asyncio
async def test_compute_vision_tags_no_override_when_clip_document_llm_also_non_medical(
    tmp_path,
):
    """Don't trade one non-medical bucket for another — override must be monotone."""
    from unittest.mock import AsyncMock, MagicMock

    from claritymed.core.ocr.base import ExtractResult

    img = tmp_path / "receipt.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    clip_response = MagicMock()
    clip_response.modality = "document"
    clip_response.confidence = 0.82
    clip_response.is_medical = False

    fake_clip = MagicMock()
    fake_clip.classify_modality = AsyncMock(return_value=clip_response)

    worker = OcrWorker(
        _LLMStubProvider(modality="photo", is_medical=False),
        medical_clip_client=fake_clip,
    )
    job = OcrJob(user_id="test", session_id="s", sha256="ab" * 32, blob_path=img)
    result = ExtractResult(
        text="thank you",
        provider_used="stub",
        chain_tried=["stub"],
        modality="photo",
        is_medical=False,
    )
    tags = await worker._compute_vision_tags(job, result)

    assert tags["modality"] == "document"
    assert "modality_from_llm_ocr" not in tags.get("vision_warnings", [])
