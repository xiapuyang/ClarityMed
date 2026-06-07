"""Tests for ``PhiGuard.filter_chunks_for_provider`` (retrieval layer)."""

from __future__ import annotations

import importlib
from datetime import datetime

import pytest

from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.orchestrator import ChunkFilterReport, PhiGuard


@pytest.fixture
def guard():
    from claritymed import config as _cfg

    importlib.reload(_cfg)
    return PhiGuard.from_config()


def _chunk(text: str, *, is_phi: bool, can_cloud: bool, source="user_rag"):
    return RetrievedChunk(
        text=text,
        source=source,
        score=0.9,
        doc_id="d1",
        chunk_index=0,
        is_phi=is_phi,
        can_cloud=can_cloud,
        user_id=42 if source == "user_rag" else None,
        ingested_at=datetime(2026, 6, 7),
    )


def test_local_provider_keeps_all_chunks(guard):
    chunks = [
        _chunk("phi-ish", is_phi=True, can_cloud=False),
        _chunk("public", is_phi=False, can_cloud=True),
    ]
    kept, report = guard.filter_chunks_for_provider(chunks, provider_kind="local")
    assert len(kept) == 2
    assert report.total == 2
    assert report.kept == 2
    assert report.filtered_phi == 0


def test_cloud_provider_drops_phi_chunks(guard):
    chunks = [
        _chunk("phi a", is_phi=True, can_cloud=False),
        _chunk("phi b", is_phi=True, can_cloud=False),
        _chunk("public c", is_phi=False, can_cloud=True),
    ]
    kept, report = guard.filter_chunks_for_provider(chunks, provider_kind="cloud")
    assert len(kept) == 1
    assert kept[0].text == "public c"
    assert report.filtered_phi == 2
    assert report.kept == 1


def test_cloud_provider_keeps_can_cloud_overrides(guard):
    """is_phi=True but can_cloud=True (user-marked public) stays for cloud."""
    chunks = [
        _chunk("opted-in public", is_phi=True, can_cloud=True),
        _chunk("normal phi", is_phi=True, can_cloud=False),
    ]
    kept, _ = guard.filter_chunks_for_provider(chunks, provider_kind="cloud")
    assert len(kept) == 1
    assert kept[0].text == "opted-in public"


def test_cloud_provider_all_non_phi_keeps_all(guard):
    chunks = [
        _chunk("ref a", is_phi=False, can_cloud=True, source="system_rag"),
        _chunk("ref b", is_phi=False, can_cloud=True, source="system_rag"),
    ]
    kept, report = guard.filter_chunks_for_provider(chunks, provider_kind="cloud")
    assert len(kept) == 2
    assert report.filtered_phi == 0


def test_empty_input(guard):
    kept, report = guard.filter_chunks_for_provider([], provider_kind="cloud")
    assert kept == []
    assert report.total == 0
    assert report.kept == 0


def test_report_is_audit_safe(guard):
    """ChunkFilterReport must not record chunk content — only counts."""
    chunks = [_chunk("very phi content", is_phi=True, can_cloud=False)]
    _, report = guard.filter_chunks_for_provider(chunks, provider_kind="cloud")
    assert isinstance(report, ChunkFilterReport)
    payload = report.model_dump()
    for value in payload.values():
        if isinstance(value, str) and "phi" in value.lower():
            # provider_kind is allowed to be "cloud"/"local"
            if value not in {"cloud", "local"}:
                pytest.fail(f"ChunkFilterReport leaks content: {value!r}")
