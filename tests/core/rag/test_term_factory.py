"""Unit tests for ``claritymed.core.rag.terms.factory``."""

from __future__ import annotations

import pytest

from claritymed.core.rag.schemas import TermServiceConfig
from claritymed.core.rag.terms.factory import (
    _reset_term_service_singleton,
    build_term_service,
    get_term_service,
)
from claritymed.core.rag.terms.umls_cmekg import NoOpTermService


@pytest.fixture(autouse=True)
def _reset():
    _reset_term_service_singleton()
    yield
    _reset_term_service_singleton()


def _noop_config() -> TermServiceConfig:
    return TermServiceConfig(active="none", catalog=[{"id": "none", "kind": "noop"}])


def test_get_term_service_builds_and_caches(monkeypatch):
    """First call to get_term_service() builds the singleton; second reuses it."""
    from claritymed.core.rag.terms import factory as _factory

    monkeypatch.setattr(
        _factory,
        "build_term_service",
        lambda config=None: NoOpTermService(),
    )
    svc1 = get_term_service()
    svc2 = get_term_service()
    assert svc1 is svc2
    assert isinstance(svc1, NoOpTermService)


def test_reset_term_service_singleton_clears_cache(monkeypatch):
    """After reset, the next call to get_term_service() rebuilds."""
    from claritymed.core.rag.terms import factory as _factory

    calls = []

    def _make():
        calls.append(1)
        return NoOpTermService()

    monkeypatch.setattr(_factory, "build_term_service", _make)

    get_term_service()
    _reset_term_service_singleton()
    get_term_service()
    assert len(calls) == 2


def test_build_term_service_noop():
    """build_term_service with id=none returns NoOpTermService."""
    svc = build_term_service(_noop_config())
    assert isinstance(svc, NoOpTermService)


def test_build_term_service_unknown_id_raises():
    """build_term_service with an unrecognised id raises UnknownTermServiceError."""
    from claritymed.errors import UnknownTermServiceError

    cfg = TermServiceConfig(
        active="mystery_id",
        catalog=[{"id": "mystery_id", "kind": "noop"}],
    )
    with pytest.raises(UnknownTermServiceError):
        build_term_service(cfg)
