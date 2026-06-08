"""Single-collection Qdrant store with bge-m3 dense+sparse hybrid query.

Why this is not LlamaIndex ``QdrantVectorStore``: that wrapper calls its
``sparse_doc_fn`` / ``sparse_query_fn`` synchronously even on the async
ingest/query paths, which forces a sync-over-async bridge around our
async ``BgeM3HttpEmbedder``. Our retrieval hot path runs inside
``AskService._run_scoped`` (async); a sync bridge there is a foot-gun.

Direct ``AsyncQdrantClient`` usage keeps the hot path fully async and
costs us roughly 80 LoC of named-vector setup + RRF query construction
— LlamaIndex's QdrantVectorStore mostly does the same wrapping.

Collection layout (per Qdrant 1.18+ named-vector semantics):

* ``vectors_config = {"dense": VectorParams(size=dense_dim, COSINE)}``
* ``sparse_vectors_config = {"sparse": SparseVectorParams()}``
* per-point payload carries ``text`` / ``doc_id`` / ``parent_id`` /
  ``chunk_index`` / ``is_phi`` / ``can_cloud`` / ``source_uri`` /
  ``ingested_at``.

Hybrid query uses Qdrant's native ``Prefetch`` + RRF fusion, so dense and
sparse hits merge at the engine, not in Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from claritymed.core.rag.chunking.base import ChildChunk
from claritymed.core.rag.embedding.base import SparseVector

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# RRF fusion balances dense + sparse at the engine. Per-stream prefetch
# limit is larger than the final limit so the fusion has enough candidates
# to merge meaningfully.
PREFETCH_MULTIPLIER = 4


@dataclass(frozen=True)
class QdrantHit:
    """One point returned from a hybrid search, mapped to project types."""

    text: str
    score: float
    payload: dict[str, Any]


class RagCollectionStore:
    """Async hybrid-search wrapper around one Qdrant collection."""

    def __init__(
        self,
        aclient: AsyncQdrantClient,
        collection_name: str,
        dense_dim: int,
    ) -> None:
        self._aclient = aclient
        self._collection = collection_name
        self._dense_dim = dense_dim

    # --- collection lifecycle -----------------------------------------

    async def ensure_collection(self) -> None:
        """Create the collection if missing (named dense + sparse layout).

        We deliberately do **not** create a ``doc_id`` payload index here:
        qdrant local mode emits ``UserWarning: payload indexes have no
        effect in the local Qdrant`` because the local store ignores
        them. ``ingest_corpus`` instead pre-loads existing ``doc_id``s
        into an in-memory set for O(1) resume lookups. When this project
        moves to a real qdrant server, add a
        ``create_payload_index(field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD)`` call here to keep
        ``has_doc`` / ``delete_by_doc_id`` fast there too.
        """
        if await self._aclient.collection_exists(self._collection):
            return
        await self._aclient.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR_NAME: qm.VectorParams(
                    size=self._dense_dim,
                    distance=qm.Distance.COSINE,
                )
            },
            sparse_vectors_config={SPARSE_VECTOR_NAME: qm.SparseVectorParams()},
        )

    async def drop_collection(self) -> bool:
        if not await self._aclient.collection_exists(self._collection):
            return False
        await self._aclient.delete_collection(self._collection)
        return True

    async def count(self) -> int:
        if not await self._aclient.collection_exists(self._collection):
            return 0
        info = await self._aclient.count(self._collection, exact=True)
        return info.count

    async def list_doc_ids(self) -> set[str]:
        """Scroll the whole collection and collect every unique ``doc_id``.

        Used at ingest startup to pre-build an in-memory resume set when
        the backing Qdrant has no usable payload index. Qdrant local mode
        ignores ``create_payload_index`` (it emits ``UserWarning: payload
        indexes have no effect in the local Qdrant``), so per-doc
        ``has_doc`` calls degrade to ~300 ms full-scan filtered counts;
        a single sweep ahead of the loop is O(N) once instead.
        """
        if not await self._aclient.collection_exists(self._collection):
            return set()
        ids: set[str] = set()
        offset: Any | None = None
        while True:
            batch, offset = await self._aclient.scroll(
                collection_name=self._collection,
                limit=2048,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            for point in batch:
                if point.payload and (did := point.payload.get("doc_id")):
                    ids.add(did)
            if offset is None:
                break
        return ids

    async def has_doc(self, doc_id: str) -> bool:
        """Return True iff at least one child with this ``doc_id`` is indexed.

        Cheap existence probe for ingest resume — ``exact=False`` lets
        Qdrant short-circuit instead of counting every match.
        """
        if not await self._aclient.collection_exists(self._collection):
            return False
        info = await self._aclient.count(
            self._collection,
            count_filter=qm.Filter(
                must=[
                    qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
                ]
            ),
            exact=False,
        )
        return info.count > 0

    # --- writes -------------------------------------------------------

    async def upsert(
        self,
        children: list[ChildChunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[SparseVector],
        *,
        is_phi: bool,
        can_cloud: bool,
    ) -> int:
        """Write children + their pre-computed embeddings to Qdrant.

        Caller embeds (we never embed inside the store — keeps the
        store's I/O surface narrow). PHI flags are set per-batch by the
        caller so a single store can serve both system (is_phi=False)
        and user_rag (is_phi=True) populations.

        Returns the number of points written.
        """
        if not children:
            return 0
        if not (len(children) == len(dense_vectors) == len(sparse_vectors)):
            raise ValueError(
                "children / dense_vectors / sparse_vectors must align: "
                f"{len(children)} / {len(dense_vectors)} / {len(sparse_vectors)}"
            )
        await self.ensure_collection()
        now = datetime.now().isoformat()
        points = [
            qm.PointStruct(
                id=child.child_id,
                vector={
                    DENSE_VECTOR_NAME: dense_vectors[i],
                    SPARSE_VECTOR_NAME: self._to_sparse(sparse_vectors[i]),
                },
                payload={
                    "text": child.text,
                    "doc_id": child.doc_id,
                    "parent_id": child.parent_id,
                    "chunk_index": child.chunk_index,
                    "collection": self._collection,
                    "is_phi": is_phi,
                    "can_cloud": can_cloud,
                    "source_uri": child.metadata.get("source_uri"),
                    "ingested_at": now,
                    **{k: v for k, v in child.metadata.items() if k != "source_uri"},
                },
            )
            for i, child in enumerate(children)
        ]
        await self._aclient.upsert(collection_name=self._collection, points=points)
        return len(points)

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Remove every child whose payload ``doc_id`` matches."""
        if not await self._aclient.collection_exists(self._collection):
            return
        await self._aclient.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="doc_id", match=qm.MatchValue(value=doc_id)
                        )
                    ]
                )
            ),
        )

    # --- hybrid query --------------------------------------------------

    async def search_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: SparseVector,
        top_k: int,
        *,
        only_cloud_safe: bool = False,
    ) -> list[QdrantHit]:
        """RRF-fused dense + sparse query, returning top-``k`` hits."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not await self._aclient.collection_exists(self._collection):
            return []

        per_stream_limit = max(top_k, top_k * PREFETCH_MULTIPLIER)
        query_filter = self._build_filter(only_cloud_safe)
        result = await self._aclient.query_points(
            collection_name=self._collection,
            prefetch=[
                qm.Prefetch(
                    query=dense_vector,
                    using=DENSE_VECTOR_NAME,
                    limit=per_stream_limit,
                    filter=query_filter,
                ),
                qm.Prefetch(
                    query=self._to_sparse(sparse_vector),
                    using=SPARSE_VECTOR_NAME,
                    limit=per_stream_limit,
                    filter=query_filter,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [self._point_to_hit(p) for p in result.points]

    # --- internals -----------------------------------------------------

    @staticmethod
    def _build_filter(only_cloud_safe: bool) -> qm.Filter | None:
        if not only_cloud_safe:
            return None
        return qm.Filter(
            must=[qm.FieldCondition(key="can_cloud", match=qm.MatchValue(value=True))]
        )

    @staticmethod
    def _to_sparse(sparse: SparseVector) -> qm.SparseVector:
        if not sparse:
            # Qdrant accepts empty sparse vectors; gives a zero-score
            # contribution that RRF effectively ignores.
            return qm.SparseVector(indices=[], values=[])
        indices = list(sparse.keys())
        values = [float(sparse[i]) for i in indices]
        return qm.SparseVector(indices=indices, values=values)

    @staticmethod
    def _point_to_hit(point: qm.ScoredPoint) -> QdrantHit:
        payload = dict(point.payload or {})
        text = payload.pop("text", "")
        return QdrantHit(text=text, score=point.score, payload=payload)
