"""Resolve the active ``Chunker`` from ``configs/retrieval.yaml``."""

from __future__ import annotations

from claritymed.core.rag.chunking.base import Chunker
from claritymed.core.rag.chunking.parent_child import LlamaIndexParentChildChunker
from claritymed.core.rag.schemas import ChunkerConfig, load_retrieval_config
from claritymed.errors import UnknownChunkerError


def build_chunker(config: ChunkerConfig | None = None) -> Chunker:
    """Instantiate the active chunker from the retrieval config.

    Args:
        config: Optional config override. When ``None``, loads from
            ``configs/retrieval.yaml``.

    Returns:
        Chunker ready to call ``.chunk(doc)``.

    Raises:
        UnknownChunkerError: Active id has no factory branch.
    """
    cfg = config or load_retrieval_config().chunker
    entry = cfg.resolved()
    if entry.id == "parent_child":
        return LlamaIndexParentChildChunker(
            parent_tok=entry.parent_tok,
            child_tok=entry.child_tok,
            overlap_tok=entry.overlap_tok,
        )
    raise UnknownChunkerError(
        f"build_chunker has no factory branch for id={entry.id!r}"
    )
