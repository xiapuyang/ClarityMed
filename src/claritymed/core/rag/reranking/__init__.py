"""Reranking subsystem.

``Reranker`` protocol + factory. v1 ships ``BgeRerankerV2M3HttpReranker``
(cross-encoder, talks to a TEI ``/rerank`` server). Adding Qwen3-Reranker
is a YAML catalog entry + a new client class + a factory branch.
"""

from claritymed.core.rag.reranking.base import RerankHit, Reranker
from claritymed.core.rag.reranking.bge_v2_m3 import BgeRerankerV2M3HttpReranker
from claritymed.core.rag.reranking.factory import build_reranker

__all__ = [
    "BgeRerankerV2M3HttpReranker",
    "RerankHit",
    "Reranker",
    "build_reranker",
]
