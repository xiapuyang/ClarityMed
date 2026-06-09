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
from claritymed.stores.paths import shared_terminology_jsonl


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
