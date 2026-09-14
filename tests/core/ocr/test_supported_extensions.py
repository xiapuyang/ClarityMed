"""Tests for ``chain_supported_extensions`` — the union the paste gate uses."""

from __future__ import annotations

from claritymed.core.ocr.factory import chain_supported_extensions
from claritymed.core.schemas.ocr import ChainEntry, OcrConfig


def test_local_only_filters_cloud_provider_extensions():
    """Under ``phi_policy=local-only`` the mineru extensions must NOT
    appear in the gate's supported set — mineru is structurally
    excluded from the runtime chain, so the gate must mirror that."""
    cfg = OcrConfig(
        document_chain=[
            ChainEntry(name="pymupdf"),
            ChainEntry(name="mineru"),  # cloud — should be filtered
        ],
        image_chain=[],
        phi_policy="local-only",
    )
    supported = chain_supported_extensions(cfg)
    assert supported is not None
    assert ".pdf" in supported
    # MineRU's extensions (.docx, .png, ...) must NOT widen the gate.
    assert ".docx" not in supported
    assert ".png" not in supported


def test_any_policy_includes_cloud_provider_extensions():
    """Under ``phi_policy=any`` mineru's extensions DO widen the gate."""
    cfg = OcrConfig(
        document_chain=[ChainEntry(name="mineru")],
        image_chain=[],
        phi_policy="any",
    )
    supported = chain_supported_extensions(cfg)
    assert supported is not None
    # MineRU's full set is now visible to the gate.
    assert {".pdf", ".docx", ".png", ".jpg"}.issubset(supported)


def test_text_extensions_are_unioned_in():
    """Plain-text extensions are accepted via the paste fast-path and
    must show up in the gate even though they aren't claimed by any
    OCR provider's ``supported_extensions``."""
    cfg = OcrConfig(
        document_chain=[ChainEntry(name="pymupdf")],
        image_chain=[],
        phi_policy="local-only",
        text_extensions=[".csv", ".md"],
    )
    supported = chain_supported_extensions(cfg)
    assert supported is not None
    assert ".csv" in supported
    assert ".md" in supported


def test_returns_none_when_catch_all_provider_present(monkeypatch):
    """If any provider declares ``supported_extensions=None`` (catch-all),
    the helper returns ``None`` so the gate disables itself rather than
    over-restricting."""
    # Mock pymupdf to declare catch-all (None) — the helper should
    # short-circuit and return None.
    from claritymed.core.ocr.pymupdf_provider import PyMuPDFOcrProvider

    monkeypatch.setattr(PyMuPDFOcrProvider, "supported_extensions", None)
    cfg = OcrConfig(
        document_chain=[ChainEntry(name="pymupdf")],
        image_chain=[],
        phi_policy="local-only",
    )
    assert chain_supported_extensions(cfg) is None
