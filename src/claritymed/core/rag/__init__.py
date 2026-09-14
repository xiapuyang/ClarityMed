"""RAG subsystem.

Layering:

    AskService
        └── RagStrategy (protocol; v1 impl: NaiveHybridStrategy)
                └── HybridRetriever
                        ├── Embedder (dense + sparse)
                        ├── Reranker (cross-encoder)
                        ├── TermService (query expansion)
                        ├── CollectionRouter (per-query active set)
                        ├── ParentStore (parent-chunk SQLite KV)
                        └── KnowledgeStore / UserRagStore (Qdrant)

Each layer is a Protocol with a catalog + active-id selection in
``configs/retrieval.yaml`` — mirror of ``stores/models.py``'s pattern.
"""

from claritymed.core.rag.retriever_factory import build_hybrid_retriever
from claritymed.core.rag.schemas import (
    ChunkerConfig,
    CollectionMetadata,
    EmbedderConfig,
    EvidenceBundle,
    GraderConfig,
    GraderReport,
    NaiveHybridStrategyConfig,
    ParentChildChunkerConfig,
    RagBootstrapConfig,
    RerankerConfig,
    RetrievalConfig,
    RetrievalTrace,
    RouterConfig,
    StrategiesConfig,
    SystemRagConfig,
    TermServiceConfig,
    UserRagConfig,
    load_retrieval_config,
)

__all__ = [
    "ChunkerConfig",
    "CollectionMetadata",
    "EmbedderConfig",
    "EvidenceBundle",
    "GraderConfig",
    "GraderReport",
    "NaiveHybridStrategyConfig",
    "ParentChildChunkerConfig",
    "RagBootstrapConfig",
    "RerankerConfig",
    "RetrievalConfig",
    "RetrievalTrace",
    "RouterConfig",
    "StrategiesConfig",
    "SystemRagConfig",
    "TermServiceConfig",
    "UserRagConfig",
    "build_hybrid_retriever",
    "load_retrieval_config",
]
