"""Resolve the active ``TermService`` from ``configs/retrieval.yaml``."""

from __future__ import annotations

from pathlib import Path

from claritymed.core.rag.schemas import TermServiceConfig, load_retrieval_config
from claritymed.core.rag.terms.base import TermService
from claritymed.core.rag.terms.umls_cmekg import (
    NoOpTermService,
    UmlsCmekgLocalService,
)
from claritymed.errors import UnknownTermServiceError


def build_term_service(config: TermServiceConfig | None = None) -> TermService:
    """Instantiate the active term service.

    Raises:
        UnknownTermServiceError: Active id has no factory branch.
        FileNotFoundError: ``umls_cmekg_local`` configured but
            ``concepts.jsonl`` is missing under ``data_dir``.
    """
    cfg = config or load_retrieval_config().term_service
    entry = cfg.resolved()
    if entry.id == "none":
        return NoOpTermService()
    if entry.id == "umls_cmekg_local":
        if not entry.data_dir:
            raise ValueError("term_service.umls_cmekg_local requires data_dir")
        jsonl_path = Path(entry.data_dir) / "concepts.jsonl"
        return UmlsCmekgLocalService(jsonl_path=jsonl_path)
    raise UnknownTermServiceError(
        f"build_term_service has no factory branch for id={entry.id!r}"
    )
