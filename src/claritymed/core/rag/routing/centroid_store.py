"""Collection centroid storage and refresh.

Centroids are mean dense vectors sampled from a Qdrant collection.
They are used by embedding-based collection routers to rank collections
by semantic similarity to a query — a more robust alternative to
keyword topic matching when the collection catalog grows large.

Storage: one JSON file per collection under ``data/shared/centroids/``.
Each file records the vector, the point count at compute time, and a
timestamp so staleness is auditable.

Delta-based refresh: ``maybe_refresh`` compares the live Qdrant point
count against the saved ``point_count``. When the delta exceeds
``delta_threshold`` (default 200 chunks) the centroid is recomputed.
This keeps the centroid reasonably fresh after incremental ingests
without burning a full scroll on every run.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# How many new chunks must accumulate before a recompute is triggered.
DEFAULT_DELTA_THRESHOLD = 200
# Number of points sampled from Qdrant for the centroid computation.
SAMPLE_SIZE = 500


class CentroidStore:
    """Persist and retrieve per-collection centroid vectors."""

    def __init__(self, base_dir: Path) -> None:
        self._dir = base_dir

    def _path(self, collection: str) -> Path:
        return self._dir / f"{collection}.json"

    def load(self, collection: str) -> dict | None:
        """Return the stored record or None if not found / corrupt."""
        p = self._path(collection)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            logger.warning("centroid file corrupt, ignoring: %s", p)
            return None

    def save(self, collection: str, vector: list[float], point_count: int) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        record = {
            "collection": collection,
            "point_count": point_count,
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "vector": vector,
        }
        self._path(collection).write_text(json.dumps(record))
        logger.info(
            "centroid saved: collection=%s point_count=%d dim=%d",
            collection,
            point_count,
            len(vector),
        )

    def vector(self, collection: str) -> list[float] | None:
        record = self.load(collection)
        return record["vector"] if record else None


async def compute_centroid(
    aclient,
    collection_name: str,
    *,
    sample_size: int = SAMPLE_SIZE,
) -> tuple[list[float], int]:
    """Sample up to ``sample_size`` dense vectors and return (centroid, total_count).

    Uses Qdrant ``count`` for the total and ``scroll`` for the sample.
    Raises ``ValueError`` when the collection is empty or has no dense vectors.
    """

    count_result = await aclient.count(collection_name)
    total = count_result.count
    if total == 0:
        raise ValueError(f"collection {collection_name!r} is empty")

    batch, _ = await aclient.scroll(
        collection_name,
        limit=sample_size,
        with_vectors=["dense"],
        with_payload=False,
    )
    vecs = []
    for point in batch:
        v = None
        if isinstance(point.vector, dict):
            v = point.vector.get("dense")
        elif isinstance(point.vector, list):
            v = point.vector
        if v:
            vecs.append(v)

    if not vecs:
        raise ValueError(f"no dense vectors found in {collection_name!r}")

    dim = len(vecs[0])
    centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
    return centroid, total


async def maybe_refresh(
    aclient,
    collection_name: str,
    store: CentroidStore,
    *,
    delta_threshold: int = DEFAULT_DELTA_THRESHOLD,
    force: bool = False,
) -> bool:
    """Recompute and save the centroid if the collection has grown enough.

    Returns True when a recompute happened, False when skipped.
    ``force=True`` bypasses the delta check (useful for first-time runs
    and the ``refresh-centroid`` CLI command).
    """
    if not force:
        record = store.load(collection_name)
        if record is not None:
            count_result = await aclient.count(collection_name)
            delta = count_result.count - record["point_count"]
            if delta < delta_threshold:
                logger.debug(
                    "centroid skip: collection=%s delta=%d < threshold=%d",
                    collection_name,
                    delta,
                    delta_threshold,
                )
                return False

    logger.info("centroid recompute: collection=%s", collection_name)
    centroid, total = await compute_centroid(aclient, collection_name)
    store.save(collection_name, centroid, total)
    return True
