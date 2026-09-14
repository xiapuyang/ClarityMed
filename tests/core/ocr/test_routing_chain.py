"""Tests for ``RoutingOcrProvider`` chain mode + phi_policy filtering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from claritymed.core.ocr.base import ExtractResult, OcrEmpty, OcrError, OcrProvider
from claritymed.core.ocr.routing_provider import (
    RoutingOcrProvider,
    _filter_chain_for_policy,
    _with_filename,
    reset_original_filename,
    set_original_filename,
)
from claritymed.errors import MinerUNotAllowed


class _LocalProvider(OcrProvider):
    is_local = True
    label = "local"
    # Default claims PDF so this provider participates in routing
    # derivation (matches the historical "this is a document-side
    # stub" intent of every test that uses it).
    supported_extensions = frozenset({".pdf"})

    def __init__(self, *, raise_with: str | None = None, text: str = "local text"):
        self._raise = raise_with
        self._text = text

    async def extract_text(self, path: Path) -> ExtractResult:
        if self._raise:
            raise OcrError(self._raise)
        return ExtractResult(
            text=self._text, provider_used=self.label, chain_tried=[self.label]
        )


class _CloudProvider(OcrProvider):
    is_local = False
    label = "cloud"

    def __init__(self, text: str = "cloud text"):
        self._text = text

    async def extract_text(self, path: Path) -> ExtractResult:
        return ExtractResult(
            text=self._text, provider_used=self.label, chain_tried=[self.label]
        )


def test_filter_chain_drops_cloud_under_local_only():
    chain = [_LocalProvider(), _CloudProvider(), _LocalProvider()]
    out = _filter_chain_for_policy(chain, "local-only")
    assert len(out) == 2
    assert all(p.is_local for p in out)


def test_filter_chain_passes_through_under_any():
    chain = [_LocalProvider(), _CloudProvider()]
    assert _filter_chain_for_policy(chain, "any") == chain


def test_chain_mode_requires_at_least_one_set():
    with pytest.raises(ValueError):
        RoutingOcrProvider()


async def test_chain_walks_until_first_success(tmp_path: Path):
    """First provider raises, second succeeds; audit lists both."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    router = RoutingOcrProvider(
        document_chain=[
            _LocalProvider(raise_with="empty"),
            _LocalProvider(text="from second"),
        ],
        image_chain=[],
    )
    captured: list[dict] = []
    with patch(
        "claritymed.core.ocr.routing_provider._emit_audit",
        side_effect=lambda p: captured.append(p),
    ):
        result = await router.extract_text(pdf)
    assert result.text == "from second"
    assert result.provider_used == "local"
    # Routing returns the full walk, not just the winner.
    assert result.chain_tried == ["local", "local"]
    assert captured[0]["status"] == "ok"
    assert len(captured[0]["chain_tried"]) == 2


async def test_chain_exhausted_raises_with_message(tmp_path: Path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    router = RoutingOcrProvider(
        document_chain=[
            _LocalProvider(raise_with="fail-one"),
            _LocalProvider(raise_with="fail-two"),
        ],
        image_chain=[],
    )
    with pytest.raises(OcrError, match="all providers exhausted"):
        await router.extract_text(pdf)


async def test_empty_chain_for_file_kind_raises(tmp_path: Path):
    """A file whose target chain is empty (e.g. all filtered by policy)
    should raise rather than silently return empty.

    Realistic scenario: ``phi_policy="local-only"`` strips every cloud
    provider out of ``image_chain``, leaving it empty; an image paste
    then routes to a chain with nobody to run it.
    """
    png = tmp_path / "scan.png"
    png.write_bytes(b"\x89PNG")
    router = RoutingOcrProvider(
        # document_chain claims .pdf via _LocalProvider's default, so
        # .png cannot route to it — it must route to the empty
        # image_chain and surface the explicit error.
        document_chain=[_LocalProvider()],
        image_chain=[],
    )
    with pytest.raises(OcrError, match="no OCR providers available"):
        await router.extract_text(png)


async def test_phi_policy_filters_chain_at_construction(tmp_path: Path):
    """When phi_policy='local-only', cloud providers are dropped at
    composition time so they cannot run even if the file is a document."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    cloud_only_chain = [_CloudProvider("cloud got it")]
    router = RoutingOcrProvider(
        document_chain=cloud_only_chain,
        image_chain=[],
        phi_policy="local-only",
    )
    # The chain becomes empty → OcrError, not cloud_got_it.
    with pytest.raises(OcrError):
        await router.extract_text(pdf)


# --- Supports-filter -------------------------------------------------


class _PdfOnlyProvider(OcrProvider):
    is_local = True
    label = "pdfonly"
    supported_extensions = frozenset({".pdf"})

    def __init__(self, *, text: str = "pdf text"):
        self._text = text

    async def extract_text(self, path: Path) -> ExtractResult:
        return ExtractResult(
            text=self._text, provider_used=self.label, chain_tried=[self.label]
        )


async def test_unsupported_provider_skipped_not_in_tried(tmp_path: Path):
    """A provider whose supported_extensions excludes the input ext is
    silently skipped and does NOT appear in chain_tried."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    # First provider only handles images; should be skipped on PDF input.
    img_only = _LocalProvider()
    img_only.supported_extensions = frozenset({".png"})
    router = RoutingOcrProvider(
        document_chain=[img_only, _PdfOnlyProvider(text="winner")],
        image_chain=[],
    )
    result = await router.extract_text(pdf)
    assert result.text == "winner"
    # Skipped provider absent from chain_tried — only the winner is recorded.
    assert result.chain_tried == ["pdfonly"]


async def test_all_unsupported_raises_distinct_error(tmp_path: Path):
    """When every chain provider declares the extension unsupported, the
    error message names that explicitly so an operator doesn't chase a
    phantom OCR failure."""
    csv = tmp_path / "data.csv"
    csv.write_bytes(b"a,b,c")
    img_only = _LocalProvider()
    img_only.supported_extensions = frozenset({".png"})
    pdf_only = _PdfOnlyProvider()
    # No non-vision provider in document_chain claims .csv, so derived
    # doc_ext is empty and .csv routes to image_chain. Both providers
    # in image_chain decline → "no chain provider supports".
    router = RoutingOcrProvider(
        document_chain=[],
        image_chain=[img_only, pdf_only],
    )
    with pytest.raises(OcrError, match="no chain provider supports"):
        await router.extract_text(csv)


async def test_supports_none_treats_as_supports_all(tmp_path: Path):
    """``supported_extensions=None`` (the base default) must mean "all" —
    a catch-all provider always gets tried by the runtime supports-filter,
    even though such providers don't participate in
    ``_derive_document_extensions``."""
    weird = tmp_path / "f.unusual"
    weird.write_bytes(b"x")
    catchall = _LocalProvider(text="catchall")
    # Override the class default ({".pdf"}) on this instance so the
    # router treats it as a catch-all provider at run time.
    catchall.supported_extensions = None
    router = RoutingOcrProvider(
        document_chain=[],
        image_chain=[catchall],
    )
    result = await router.extract_text(weird)
    assert result.text == "catchall"
    assert result.chain_tried == ["local"]


# --- Document extension derivation ----------------------------------


async def test_doc_ext_derived_from_non_vision_provider_supported_set(
    tmp_path: Path,
):
    """The doc-vs-image disambiguation set is derived from each
    document_chain provider's ``supported_extensions`` — not a magic
    constant. A docx file routes to document_chain because pandoc
    claims it; a vision LLM in the same chain doesn't sway routing."""

    class _PandocLike(OcrProvider):
        is_local = True
        label = "pandoc"
        supported_extensions = frozenset({".docx"})

        async def extract_text(self, path: Path) -> ExtractResult:
            return ExtractResult(
                text="docx", provider_used=self.label, chain_tried=[self.label]
            )

    class _VisionLLM(OcrProvider):
        is_local = True
        label = "llm"
        is_vision = True
        supported_extensions = frozenset({".docx", ".png"})

        async def extract_text(self, path: Path) -> ExtractResult:
            return ExtractResult(
                text="vision", provider_used=self.label, chain_tried=[self.label]
            )

    docx = tmp_path / "report.docx"
    docx.write_bytes(b"PK")
    router = RoutingOcrProvider(
        document_chain=[_PandocLike(), _VisionLLM()],
        image_chain=[],
    )
    # .docx → derived doc_ext = {.docx} (pandoc) → routes to document_chain
    result = await router.extract_text(docx)
    assert result.provider_used == "pandoc"

    # .png → derived doc_ext does NOT include .png even though the
    # vision LLM in document_chain "supports" it; routes to (empty)
    # image_chain instead and raises.
    png = tmp_path / "scan.png"
    png.write_bytes(b"\x89PNG")
    with pytest.raises(OcrError, match="no OCR providers available"):
        await router.extract_text(png)


# --- MineRU env-gate -------------------------------------------------


def test_mineru_constructor_raises_without_env(monkeypatch: pytest.MonkeyPatch):
    """Constructor refuses without CLARITYMED_ALLOW_MINERU=1."""
    monkeypatch.delenv("CLARITYMED_ALLOW_MINERU", raising=False)
    from claritymed.core.ocr.mineru_provider import MineRUOcrProvider

    with pytest.raises(MinerUNotAllowed):
        MineRUOcrProvider("sk-test")


def test_mineru_constructor_succeeds_with_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLARITYMED_ALLOW_MINERU", "1")
    from claritymed.core.ocr.mineru_provider import MineRUOcrProvider

    provider = MineRUOcrProvider("sk-test")
    assert provider.is_local is False


def test_mineru_is_local_false():
    """is_local=False is the structural PHI guard."""
    from claritymed.core.ocr.mineru_provider import MineRUOcrProvider

    assert MineRUOcrProvider.is_local is False


# --- OcrEmpty handling -----------------------------------------------


class _EmptyProvider(OcrProvider):
    """Provider that raises OcrEmpty (image has no extractable text)."""

    is_local = True
    label = "empty"
    supported_extensions = frozenset({".pdf"})

    async def extract_text(self, path: Path) -> ExtractResult:
        raise OcrEmpty("no text found")


async def test_all_empty_raises_ocr_empty(tmp_path: Path):
    """When every provider raises OcrEmpty, the router raises OcrEmpty (not OcrError)."""
    pdf = tmp_path / "blank.pdf"
    pdf.write_bytes(b"%PDF")
    router = RoutingOcrProvider(
        document_chain=[_EmptyProvider(), _EmptyProvider()],
        image_chain=[],
    )
    with pytest.raises(OcrEmpty, match="all providers returned no text"):
        await router.extract_text(pdf)


class _EmptyWithHintProvider(OcrProvider):
    """Empty provider that carries a modality/is_medical hint on OcrEmpty.

    Mirrors the LLMOcrProvider's empty-path behavior: even with no text,
    a vision LLM may have read the pixels and tagged modality/is_medical.
    The routing layer must forward that hint on the final OcrEmpty so the
    worker's empty branch can still write modality into ``ocr.json``.
    """

    is_local = True
    label = "llm-stub"
    supported_extensions = frozenset({".pdf", ".png"})

    async def extract_text(self, path: Path) -> ExtractResult:
        raise OcrEmpty(
            "no text found",
            extraction=ExtractResult(
                text="",
                provider_used=self.label,
                chain_tried=[self.label],
                modality="ultrasound",
                is_medical=True,
            ),
        )


async def test_all_empty_carries_leaf_hint_in_final_exc(tmp_path: Path):
    """Final OcrEmpty must surface the LLM leaf's modality/is_medical signal.

    Otherwise the worker writes a bare ``empty`` sentinel and the LLM-side
    routing rules in ``detect_disease_from_image_tool.yaml`` have nothing
    to branch on, so the vision tool never fires for an OCR-blank medical
    image.
    """
    blob = tmp_path / "scan.pdf"
    blob.write_bytes(b"%PDF")
    router = RoutingOcrProvider(
        document_chain=[_EmptyWithHintProvider(), _EmptyProvider()],
        image_chain=[],
    )
    with pytest.raises(OcrEmpty) as excinfo:
        await router.extract_text(blob)
    hint = excinfo.value.extraction
    assert hint is not None
    assert hint.modality == "ultrasound"
    assert hint.is_medical is True
    # chain_tried reflects every leaf actually walked, in order — the
    # worker reads this to populate ocr.json.chain_tried instead of "[]".
    assert hint.chain_tried == ["llm-stub", "empty"]


async def test_empty_then_real_error_raises_ocr_error(tmp_path: Path):
    """OcrEmpty followed by a real OcrError → OcrError wins (had_real_error=True)."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    router = RoutingOcrProvider(
        document_chain=[
            _EmptyProvider(),
            _LocalProvider(raise_with="provider crashed"),
        ],
        image_chain=[],
    )
    with pytest.raises(OcrError, match="all providers exhausted"):
        await router.extract_text(pdf)


async def test_empty_provider_then_success_returns_text(tmp_path: Path):
    """OcrEmpty from first provider → chain continues → second succeeds."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    router = RoutingOcrProvider(
        document_chain=[_EmptyProvider(), _LocalProvider(text="found it")],
        image_chain=[],
    )
    from unittest.mock import patch

    with patch("claritymed.core.ocr.routing_provider._emit_audit"):
        result = await router.extract_text(pdf)
    assert result.text == "found it"


# --- _with_filename --------------------------------------------------


def test_with_filename_attaches_name_when_set():
    payload = {"status": "ok"}
    token = set_original_filename("scan.pdf")
    try:
        out = _with_filename(payload)
    finally:
        reset_original_filename(token)
    assert out == {"status": "ok", "original_filename": "scan.pdf"}
    # Original dict is not mutated.
    assert "original_filename" not in payload


def test_with_filename_passthrough_when_unset():
    payload = {"status": "ok"}
    out = _with_filename(payload)
    assert out is payload
