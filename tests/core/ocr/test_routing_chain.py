"""Tests for ``RoutingOcrProvider`` chain mode + phi_policy filtering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from claritymed.core.ocr.base import OcrError, OcrProvider
from claritymed.core.ocr.routing_provider import (
    RoutingOcrProvider,
    _filter_chain_for_policy,
)
from claritymed.errors import MinerUNotAllowed


class _LocalProvider(OcrProvider):
    is_local = True

    def __init__(self, *, raise_with: str | None = None, text: str = "local text"):
        self._raise = raise_with
        self._text = text

    async def extract_text(self, path: Path) -> str:
        if self._raise:
            raise OcrError(self._raise)
        return self._text


class _CloudProvider(OcrProvider):
    is_local = False

    def __init__(self, text: str = "cloud text"):
        self._text = text

    async def extract_text(self, path: Path) -> str:
        return self._text


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


def test_chain_mode_rejects_mixed_construction_args():
    with pytest.raises(ValueError):
        RoutingOcrProvider(
            document_provider=_LocalProvider(),
            document_chain=[_LocalProvider()],
        )


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
    assert result == "from second"
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
