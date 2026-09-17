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
import os
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from claritymed.core.rag.chunking.base import ChildChunk
from claritymed.core.rag.embedding.base import SparseVector
from claritymed.errors import MissingApiKeyError

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# qdrant-client's default 5-second per-request timeout is too tight for
# warmup scrolls and full-collection counts on user libraries with
# thousands of chunks. 30s matches typical Docker/cloud cold-start
# behaviour. Override per deployment via ``CLARITYMED_QDRANT_TIMEOUT_S``.
_QDRANT_DEFAULT_TIMEOUT_S = 30

# Bound on how long we wait for the process-level client cache lock.
# CLAUDE.md mandates timeout-bounded acquires so a deadlock surfaces as
# a visible RuntimeError instead of an opaque hang. 5s is generous —
# the protected section only does a dict lookup or a single
# AsyncQdrantClient construction.
_LOCAL_CLIENTS_LOCK_TIMEOUT_S = 5.0

# Process-level cache: one AsyncQdrantClient per resolved directory path.
# Qdrant local mode writes a .lock file on open; a second open on the same
# path raises "already accessed by another instance" — even within the same
# process. Caching here lets the retriever factories and the post-tool embed
# tasks share the same client without conflicting.
_local_clients: dict[str, AsyncQdrantClient] = {}
_local_clients_lock = threading.Lock()


def open_local_qdrant_client(user_dir: Path) -> AsyncQdrantClient:
    """Open path-mode AsyncQdrantClient with stale-lock recovery.

    Qdrant's local mode writes ``user_dir/.lock`` on open and removes it
    on close. A crashed TUI leaves the file behind; the next
    ``AsyncQdrantClient(path=...)`` raises ``"already accessed by another
    instance"`` even when no process is alive — the user has to manually
    ``rm .lock`` to get unstuck. Defeats unattended use (cron, scheduled
    sync) which is why we handle it here.

    Recovery is conservative: only delete the lock if ``lsof`` confirms
    no other process holds it open. When ``lsof`` is unavailable or some
    process *is* holding it, the original error propagates with an
    actionable hint instead of unsafely nuking a real lock.

    A process-level cache keyed by resolved path ensures the retriever
    factories and post-tool embed tasks share one client per directory,
    avoiding the "already accessed" conflict within the same process.
    """
    key = str(user_dir.resolve())
    if not _local_clients_lock.acquire(timeout=_LOCAL_CLIENTS_LOCK_TIMEOUT_S):
        raise RuntimeError(
            f"qdrant: failed to acquire local-clients lock within "
            f"{_LOCAL_CLIENTS_LOCK_TIMEOUT_S}s — likely a deadlock; "
            "send SIGUSR1 for a thread dump (faulthandler)."
        )
    try:
        existing = _local_clients.get(key)
        if existing is not None:
            return existing
        client = _open_local_qdrant_client_uncached(user_dir)
        _local_clients[key] = client
        return client
    finally:
        _local_clients_lock.release()


def _open_local_qdrant_client_uncached(user_dir: Path) -> AsyncQdrantClient:
    try:
        return AsyncQdrantClient(path=str(user_dir))
    except RuntimeError as exc:
        if "already accessed" not in str(exc):
            raise
        lock_file = user_dir / ".lock"
        if not lock_file.exists():
            # Race: lock vanished between qdrant's check and ours. Just retry.
            return AsyncQdrantClient(path=str(user_dir))
        if _lock_is_held_by_another_process(lock_file):
            raise RuntimeError(
                f"{exc}\n"
                f"Another process holds {lock_file}. Stop the other "
                f"claritymed instance, or if you're sure none is running, "
                f"manually: rm {lock_file}"
            ) from exc
        logger.warning(
            "qdrant: stale lock at %s (no process holds it); removing and retrying.",
            lock_file,
        )
        lock_file.unlink(missing_ok=True)
        return AsyncQdrantClient(path=str(user_dir))


def _lock_is_held_by_another_process(lock_file: Path) -> bool:
    """True iff lsof reports any pid (other than ours) holding lock_file open.

    Conservative default: when lsof is unavailable, returns True so we
    refuse to delete rather than risk nuking a real lock.
    """
    try:
        result = subprocess.run(
            ["lsof", "-t", str(lock_file)],
            capture_output=True,
            check=False,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "lsof unavailable (%s); cannot confirm qdrant lock state, "
            "treating as held to avoid unsafe deletion.",
            exc,
        )
        return True
    my_pid = str(os.getpid())
    others = [p for p in result.stdout.decode().split() if p and p != my_pid]
    return bool(others)


# RRF fusion balances dense + sparse at the engine. Per-stream prefetch
# limit is larger than the final limit so the fusion has enough candidates
# to merge meaningfully.
PREFETCH_MULTIPLIER = 4

# Cap per-request point count on upsert. A single guideline PDF can chunk
# into 500-2000 children; sending them as one PointStruct[] inflates the
# request body to several MB of JSON and routinely trips Qdrant's HTTP
# transport (server tears down the connection mid-write, surfacing as
# httpx.ReadError). 128 sits well inside Qdrant's recommended 100-200
# batch size and keeps each request comfortably under typical body
# limits. Tune via this constant only — do not parameterise per-call,
# the right value is a property of the wire transport, not the caller.
_UPSERT_BATCH_SIZE = 128


def build_qdrant_client(
    *,
    url: str,
    api_key_env: str | None = None,
) -> AsyncQdrantClient:
    """Connect to a Qdrant server (Docker, native binary, or Qdrant Cloud).

    Local file-locked mode is intentionally not supported. The SQLite
    layout used by ``qdrant-client.local`` is incompatible with the
    server's segment format (no in-place migration; switching costs a
    full re-embed), and the file lock forces a single-process model
    that breaks under ingest + TUI concurrency. Run a real server.

    ``api_key_env`` names an env var holding the Qdrant Cloud key.
    Declaring it without setting the env fail-louds rather than
    silently sending unauthenticated requests — a misconfig that would
    otherwise surface as opaque HTTP errors deep in the query path.
    """
    api_key = None
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise MissingApiKeyError(
                f"qdrant.api_key_env={api_key_env} is set but the env "
                "var is empty or unset",
            )
    timeout_s = int(
        os.environ.get("CLARITYMED_QDRANT_TIMEOUT_S", _QDRANT_DEFAULT_TIMEOUT_S)
    )
    return AsyncQdrantClient(
        url=url, api_key=api_key, timeout=timeout_s, trust_env=False
    )


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

        Also creates a keyword payload index on ``doc_id`` so resume
        probes (``has_doc``) and bulk deletes (``delete_by_doc_id``) hit
        an index instead of full-scanning. Server-only — the local
        backend used to ignore this with a warning, but local mode is
        no longer supported.
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
        await self._aclient.create_payload_index(
            collection_name=self._collection,
            field_name="doc_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        await self._aclient.create_payload_index(
            collection_name=self._collection,
            field_name="source_uri",
            field_schema=qm.PayloadSchemaType.KEYWORD,
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

        Used at ingest startup to pre-build an in-memory resume set so the
        per-doc check is O(1). For ~250k chunks against localhost Docker
        this takes ~5s (≈125 round-trips × 2048 batch). Faster than per-doc
        ``has_doc`` even with the payload index, since we'd still pay the
        network round-trip per doc.
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

    async def find_doc_id_by_source_uri(self, source_uri: str) -> str | None:
        """Return the ``doc_id`` of the first chunk matching *source_uri*, or None.

        Uses the ``source_uri`` payload index, so this is a single indexed
        lookup rather than a full scan.
        """
        if not await self._aclient.collection_exists(self._collection):
            return None
        batch, _ = await self._aclient.scroll(
            collection_name=self._collection,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="source_uri", match=qm.MatchValue(value=source_uri)
                    )
                ]
            ),
            limit=1,
            with_payload=["doc_id"],
            with_vectors=False,
        )
        if not batch:
            return None
        return (batch[0].payload or {}).get("doc_id")

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
        # Batch to stay under Qdrant's HTTP body limits (see
        # ``_UPSERT_BATCH_SIZE`` for context). The fan-out is sequential
        # on purpose: each upsert acks a write to the same collection,
        # so concurrent batches would only contend for the same WAL.
        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            batch = points[start : start + _UPSERT_BATCH_SIZE]
            await self._aclient.upsert(collection_name=self._collection, points=batch)
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

    async def search_dense_max_score(self, dense_vector: list[float]) -> float | None:
        """Return the cosine score of the closest existing chunk, or None.

        Single dense-only query against the collection — no sparse leg,
        no RRF fusion. Used by per-chunk upload dedupe to ask "is there
        already something this similar in the user's library?" before
        spending the upsert bandwidth.

        Returns ``None`` when the collection doesn't exist yet
        (new user) or scroll yields no points. Returns the raw cosine
        score (Qdrant stores cosine distance, queries return similarity
        in ``[-1, 1]``); callers compare against their configured
        threshold (typically ``0.92``-``0.95``).
        """
        if not await self._aclient.collection_exists(self._collection):
            return None
        result = await self._aclient.query_points(
            collection_name=self._collection,
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        if not result.points:
            return None
        return float(result.points[0].score)

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
            # Silent [] return on missing collection is a footgun: the
            # router still announces the collection as 'active', but
            # rag.retrieval audit shows num_chunks=0 with no hint that
            # the collection itself is absent. Most common cause is
            # ingesting via one backend (local path) and querying via
            # another (Docker server) — the storage isn't shared.
            logger.warning(
                "qdrant collection %r not found on the active client "
                "(check CLARITYMED_QDRANT_URL vs ingest backend)",
                self._collection,
            )
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
