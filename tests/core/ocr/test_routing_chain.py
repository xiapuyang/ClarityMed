"""Tests for ``RoutingOcrProvider`` chain mode + phi_policy filtering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from claritymed.core.ocr.base import ExtractResult, OcrError, OcrProvider
from claritymed.core.ocr.routing_provider import (
    RoutingOcrProvider,
    _filter_chain_for_policy,
)
from claritymed.errors import MinerUNotAllowed


class _LocalProvider(OcrProvider):
    is_local = True
    label = "local"

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
    """A document with an empty document_chain (e.g. all filtered by policy)
    should raise rather than silently return empty."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    router = RoutingOcrProvider(
        document_chain=[],
        image_chain=[_LocalProvider()],
    )
    with pytest.raises(OcrError, match="no OCR providers available"):
        await router.extract_text(pdf)


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
    router = RoutingOcrProvider(
        document_chain=[img_only, pdf_only],
        image_chain=[],
        # csv treated as a document so it routes to document_chain.
        document_extensions=frozenset({".csv"}),
    )
    with pytest.raises(OcrError, match="no chain provider supports"):
        await router.extract_text(csv)


async def test_supports_none_treats_as_supports_all(tmp_path: Path):
    """``supported_extensions=None`` (the base default) must mean "all" —
    a catch-all provider always gets tried."""
    weird = tmp_path / "f.unusual"
    weird.write_bytes(b"x")
    catchall = _LocalProvider(text="catchall")
    # supported_extensions stays None on _LocalProvider (inherited from base).
    router = RoutingOcrProvider(
        document_chain=[],
        image_chain=[catchall],
    )
    result = await router.extract_text(weird)
    assert result.text == "catchall"
    assert result.chain_tried == ["local"]


# --- Document extensions widening (for text_extensions) --------------


async def test_document_extensions_widened_routes_text_to_document_chain(
    tmp_path: Path,
):
    """When document_extensions includes text-style suffixes, csv/md
    files route to document_chain instead of image_chain."""
    md = tmp_path / "note.md"
    md.write_bytes(b"# hi")
    router = RoutingOcrProvider(
        document_chain=[_LocalProvider(text="doc-side")],
        image_chain=[_LocalProvider(text="image-side")],
        document_extensions=frozenset({".pdf", ".md"}),
    )
    result = await router.extract_text(md)
    assert result.text == "doc-side"


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
