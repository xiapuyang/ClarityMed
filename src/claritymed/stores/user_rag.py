"""Per-user RAG store backed by Qdrant.

Each user gets their own Qdrant collection (``user_rag_<user_id>``) — a
forgotten ``user_id`` filter cannot leak chunks across users because there
is no shared collection to leak from. Removing a user is a single
``drop_collection`` call.

Two enforcement points wire into ``PhiGuard``:

1. ``add_document`` scrubs each chunk's free text with
   ``PhiGuard.scrub_free_text`` *before* the chunk is embedded or
   persisted. The original (un-scrubbed) text is held only in memory and
   discarded as soon as the embedding + write completes. When the caller
   sets ``public=True`` (user-marked public reference, e.g. a published
   paper), the scrub is skipped and ``can_cloud`` is recorded as True.

2. ``search`` returns ``RetrievedChunk`` objects whose ``is_phi`` /
   ``can_cloud`` payload feeds the retrieval-layer filter in
   ``PhiGuard.filter_chunks_for_provider``.

Embedding is pluggable via the ``Embedder`` protocol — the default
implementation lazy-loads ``fastembed`` for CPU-friendly local embeddings;
tests inject a deterministic stub so they need not download model weights.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.orchestrator import PhiGuard

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_DIM = 384  # fastembed BAAI/bge-small-en-v1.5


class Embedder(Protocol):
    """Anything that turns text into a dense vector and reports its dim."""

    def embed(self, text: str) -> list[float]: ...

    @property
    def dimension(self) -> int: ...


def _collection_name(user_id: str) -> str:
    return f"user_rag_{user_id}"


class UserRagStore:
    """CRUD for per-user RAG collections with PHI scrubbing baked in."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        guard: PhiGuard,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._guard = guard

    @classmethod
    def from_defaults(cls, qdrant_path: str | None = None) -> "UserRagStore":
        """Build a store with a local Qdrant client and the default embedder.

        Pass ``qdrant_path=":memory:"`` to get an in-process store useful for
        tests; otherwise the client persists under ``DATA_DIR/qdrant/user_rag/``.
        """
        from claritymed.stores.paths import user_rag_qdrant_dir

        path = qdrant_path or str(user_rag_qdrant_dir())
        client = (
            QdrantClient(path=path) if path != ":memory:" else QdrantClient(":memory:")
        )
        return cls(
            client=client,
            embedder=_LazyFastEmbedEmbedder(),
            guard=PhiGuard.from_config(),
        )

    def ensure_collection(self, user_id: str) -> None:
        """Create the user's collection if it does not exist."""
        name = _collection_name(user_id)
        if self._client.collection_exists(name):
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=self._embedder.dimension,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def add_document(
        self,
        user_id: str,
        doc_id: str,
        chunks: list[str],
        metadata: dict[str, Any] | None = None,
        public: bool = False,
    ) -> int:
        """Scrub, embed, and persist chunks. Returns chunk count written.

        When ``public=False`` (the default), each chunk's text is passed
        through ``PhiGuard.scrub_free_text`` before embedding — the original
        text never reaches Qdrant. When ``public=True`` the caller has
        explicitly declared the document is public reference material
        (e.g. a published paper) and the scrub is skipped.
        """
        self.ensure_collection(user_id)

        now = datetime.now(UTC).isoformat()
        is_phi = not public
        can_cloud = public
        source_uri = (metadata or {}).get("source_uri")

        points: list[qmodels.PointStruct] = []
        for i, raw_text in enumerate(chunks):
            text = raw_text
            if not public:
                text, _ = self._guard.scrub_free_text(raw_text)

            vector = self._embedder.embed(text)
            point_id = str(uuid.uuid4())
            payload = {
                "doc_id": doc_id,
                "chunk_index": i,
                "text": text,
                "user_id": user_id,
                "is_phi": is_phi,
                "can_cloud": can_cloud,
                "source_uri": source_uri,
                "ingested_at": now,
            }
            points.append(
                qmodels.PointStruct(id=point_id, vector=vector, payload=payload)
            )

        self._client.upsert(
            collection_name=_collection_name(user_id),
            points=points,
        )
        return len(points)

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        only_cloud_safe: bool = False,
    ) -> list[RetrievedChunk]:
        """Return top-k chunks for the user. Empty list if no collection yet.

        ``only_cloud_safe=True`` filters down to chunks whose ``can_cloud=True``
        — equivalent to the retrieval-layer PHI filter for the cloud provider.
        Most callers should retrieve unfiltered and let
        ``PhiGuard.filter_chunks_for_provider`` decide later, but this knob
        helps when the caller already knows the destination is cloud.
        """
        name = _collection_name(user_id)
        if not self._client.collection_exists(name):
            return []

        query_vector = self._embedder.embed(query)
        query_filter: qmodels.Filter | None = None
        if only_cloud_safe:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="can_cloud", match=qmodels.MatchValue(value=True)
                    )
                ]
            )

        hits = self._client.query_points(
            collection_name=name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        ).points

        return [self._point_to_chunk(hit) for hit in hits]

    def delete_document(self, user_id: str, doc_id: str) -> int:
        """Delete every chunk whose payload ``doc_id`` matches."""
        name = _collection_name(user_id)
        if not self._client.collection_exists(name):
            return 0
        result = self._client.delete(
            collection_name=name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="doc_id", match=qmodels.MatchValue(value=doc_id)
                        )
                    ]
                )
            ),
        )
        # qdrant returns UpdateResult; the per-doc count isn't exposed but the
        # collection now has fewer points. Best-effort return value.
        return getattr(result, "operation_id", 0) or 0

    def drop_user(self, user_id: str) -> bool:
        """Remove the user's collection entirely (PHI-clean erase)."""
        name = _collection_name(user_id)
        if not self._client.collection_exists(name):
            return False
        self._client.delete_collection(collection_name=name)
        return True

    @staticmethod
    def _point_to_chunk(point: qmodels.ScoredPoint) -> RetrievedChunk:
        p = point.payload or {}
        return RetrievedChunk(
            text=p.get("text", ""),
            source="user_rag",
            score=point.score,
            doc_id=p.get("doc_id", ""),
            chunk_index=p.get("chunk_index", 0),
            is_phi=p.get("is_phi", True),
            can_cloud=p.get("can_cloud", False),
            user_id=None,  # int cast not used here; user_id is str in our system
            source_uri=p.get("source_uri"),
            ingested_at=_parse_iso(p.get("ingested_at")),
        )


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


class _LazyFastEmbedEmbedder:
    """Default embedder. Lazily imports fastembed so the module loads fast
    and tests that inject a stub never need fastembed installed.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding  # imported lazily

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._ensure_model()
        return next(iter(model.embed([text]))).tolist()

    @property
    def dimension(self) -> int:
        return DEFAULT_VECTOR_DIM
