"""Resolve the active ``TermService`` from ``configs/retrieval.yaml``."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

from claritymed.core.rag.schemas import TermServiceConfig, load_retrieval_config
from claritymed.core.rag.terms.base import TermService
from claritymed.core.rag.terms.umls_cmekg import (
    NoOpTermService,
    UmlsCmekgLocalService,
)
from claritymed.errors import UnknownTermServiceError
from claritymed.stores.paths import shared_terminology_jsonl

_singleton: TermService | None = None
_singleton_lock = Lock()


def get_term_service() -> TermService:
    """Process-wide :class:`TermService` singleton.

    First call constructs from the default
    :func:`load_retrieval_config` view; subsequent calls return the
    same instance. Use this from production code paths (RAG retriever,
    ingest prepare, symptoms eligibility) instead of repeatedly calling
    :func:`build_term_service`, which would re-parse the JSONL on each
    invocation.

    Tests that need a fresh instance (or a different
    :class:`TermServiceConfig`) should call
    :func:`build_term_service` directly and reset via
    :func:`_reset_term_service_singleton`.
    """
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = build_term_service()
        return _singleton


def _reset_term_service_singleton() -> None:
    """Test helper — drop the cached singleton so the next call rebuilds."""
    global _singleton
    with _singleton_lock:
        _singleton = None


def build_term_service(config: TermServiceConfig | None = None) -> TermService:
    """Instantiate the active term service.

    When ``data_dir`` is omitted, falls back to
    ``shared_terminology_jsonl()`` (i.e. ``SHARED_DIR/terminology/``).
    Operators only set ``data_dir`` to point at a non-default location
    (e.g. an NAS-mounted export).

    Raises:
        UnknownTermServiceError: Active id has no factory branch.
        FileNotFoundError: ``umls_cmekg_local`` configured but
            ``concepts.jsonl`` is missing. Message points at the seed
            script so operators can recover without grepping the codebase.
    """
    cfg = config or load_retrieval_config().term_service
    entry = cfg.resolved()
    if entry.id == "none":
        return NoOpTermService()
    if entry.id == "umls_cmekg_local":
        jsonl_path = (
            Path(entry.data_dir) / "concepts.jsonl"
            if entry.data_dir
            else shared_terminology_jsonl()
        )
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"term_service.umls_cmekg_local: {jsonl_path} not found. "
                f"Run `uv run python scripts/init_terminology.py --seed` to "
                f"write a starter dataset, or point `term_service.catalog[*]"
                f".data_dir` at an existing UMLS+CMeKG export directory."
            )
        return UmlsCmekgLocalService(jsonl_path=jsonl_path)
    raise UnknownTermServiceError(
        f"build_term_service has no factory branch for id={entry.id!r}"
    )
