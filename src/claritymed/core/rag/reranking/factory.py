"""Resolve the active ``Reranker`` from ``configs/retrieval.yaml``."""

from __future__ import annotations

from claritymed.core.rag.reranking.base import Reranker
from claritymed.core.rag.reranking.bge_v2_m3 import BgeRerankerV2M3HttpReranker
from claritymed.core.rag.schemas import RerankerConfig, load_retrieval_config
from claritymed.errors import UnknownRerankerError


def build_reranker(config: RerankerConfig | None = None) -> Reranker:
    """Instantiate the active reranker from the retrieval config.

    Args:
        config: Optional config override. When ``None``, loads from
            ``configs/retrieval.yaml``.

    Returns:
        Reranker ready for async use.

    Raises:
        UnknownRerankerError: Active id has no factory branch yet.
    """
    cfg = config or load_retrieval_config().rerankers
    entry = cfg.resolved()
    if entry.id == "bge_v2_m3_http":
        return BgeRerankerV2M3HttpReranker(
            base_url=entry.base_url,
            batch_size=entry.batch_size,
            timeout_s=entry.timeout_s,
            api_key_env=entry.api_key_env,
        )
    raise UnknownRerankerError(
        f"build_reranker has no factory branch for id={entry.id!r}"
    )
