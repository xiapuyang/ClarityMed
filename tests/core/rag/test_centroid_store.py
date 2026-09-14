"""Unit tests for core/rag/routing/centroid_store.py.

Uses an in-memory AsyncQdrantClient so no external Qdrant process is needed.
"""

from __future__ import annotations


import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from claritymed.core.rag.routing.centroid_store import (
    CentroidStore,
    compute_centroid,
    maybe_refresh,
)

DENSE_DIM = 4
COLLECTION = "test_col"


# --- helpers ------------------------------------------------------------


async def _seed_collection(aclient: AsyncQdrantClient, n: int = 10) -> None:
    """Create a collection and upsert n points with deterministic dense vectors."""
    await aclient.create_collection(
        COLLECTION,
        vectors_config={
            "dense": qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE)
        },
    )
    points = [
        qm.PointStruct(
            id=i,
            vector={"dense": [float(i % 4) / 4, 0.5, 0.25, 0.1]},
            payload={},
        )
        for i in range(n)
    ]
    await aclient.upsert(COLLECTION, points=points)


# --- CentroidStore persistence ------------------------------------------


def test_load_returns_none_for_missing(tmp_path):
    store = CentroidStore(tmp_path / "centroids")
    assert store.load("nonexistent") is None


def test_load_returns_none_for_corrupt_file(tmp_path):
    d = tmp_path / "centroids"
    d.mkdir()
    (d / "bad.json").write_text("not json{{")
    store = CentroidStore(d)
    assert store.load("bad") is None


def test_save_and_load_roundtrip(tmp_path):
    store = CentroidStore(tmp_path / "centroids")
    vec = [0.1, 0.2, 0.3, 0.4]
    store.save("col_a", vec, point_count=500)

    record = store.load("col_a")
    assert record is not None
    assert record["collection"] == "col_a"
    assert record["point_count"] == 500
    assert record["vector"] == vec
    assert "computed_at" in record


def test_vector_helper_returns_none_when_absent(tmp_path):
    store = CentroidStore(tmp_path / "centroids")
    assert store.vector("missing") is None


def test_vector_helper_returns_list_when_saved(tmp_path):
    store = CentroidStore(tmp_path / "centroids")
    vec = [0.5, 0.5, 0.5, 0.5]
    store.save("col_b", vec, point_count=100)
    assert store.vector("col_b") == vec


def test_save_creates_parent_dirs(tmp_path):
    store = CentroidStore(tmp_path / "deep" / "nested" / "centroids")
    store.save("col_c", [0.0, 0.0, 0.0, 0.0], point_count=1)
    assert (tmp_path / "deep" / "nested" / "centroids" / "col_c.json").exists()


def test_save_overwrites_previous(tmp_path):
    store = CentroidStore(tmp_path / "centroids")
    store.save("col_d", [0.1, 0.1, 0.1, 0.1], point_count=10)
    store.save("col_d", [0.9, 0.9, 0.9, 0.9], point_count=20)
    assert store.vector("col_d") == [0.9, 0.9, 0.9, 0.9]


# --- compute_centroid ---------------------------------------------------


@pytest.mark.asyncio
async def test_compute_centroid_returns_correct_dim():
    aclient = AsyncQdrantClient(":memory:")
    await _seed_collection(aclient, n=8)
    vec, total = await compute_centroid(aclient, COLLECTION)
    assert len(vec) == DENSE_DIM
    assert total == 8


@pytest.mark.asyncio
async def test_compute_centroid_total_matches_count():
    aclient = AsyncQdrantClient(":memory:")
    await _seed_collection(aclient, n=15)
    _, total = await compute_centroid(aclient, COLLECTION)
    assert total == 15


@pytest.mark.asyncio
async def test_compute_centroid_values_are_mean_of_samples():
    aclient = AsyncQdrantClient(":memory:")
    await aclient.create_collection(
        COLLECTION,
        vectors_config={"dense": qm.VectorParams(size=2, distance=qm.Distance.COSINE)},
    )
    await aclient.upsert(
        COLLECTION,
        points=[
            qm.PointStruct(id=0, vector={"dense": [0.0, 1.0]}, payload={}),
            qm.PointStruct(id=1, vector={"dense": [1.0, 0.0]}, payload={}),
        ],
    )
    vec, _ = await compute_centroid(aclient, COLLECTION, sample_size=10)
    assert vec == pytest.approx([0.5, 0.5], abs=1e-6)


@pytest.mark.asyncio
async def test_compute_centroid_raises_on_empty_collection():
    aclient = AsyncQdrantClient(":memory:")
    await aclient.create_collection(
        COLLECTION,
        vectors_config={
            "dense": qm.VectorParams(size=DENSE_DIM, distance=qm.Distance.COSINE)
        },
    )
    with pytest.raises(ValueError, match="empty"):
        await compute_centroid(aclient, COLLECTION)


@pytest.mark.asyncio
async def test_compute_centroid_sample_size_caps_scroll():
    aclient = AsyncQdrantClient(":memory:")
    await _seed_collection(aclient, n=20)
    vec, total = await compute_centroid(aclient, COLLECTION, sample_size=5)
    assert total == 20
    assert len(vec) == DENSE_DIM


# --- maybe_refresh ------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_refresh_force_always_computes(tmp_path):
    aclient = AsyncQdrantClient(":memory:")
    await _seed_collection(aclient, n=5)
    store = CentroidStore(tmp_path / "centroids")

    refreshed = await maybe_refresh(aclient, COLLECTION, store, force=True)

    assert refreshed is True
    assert store.vector(COLLECTION) is not None


@pytest.mark.asyncio
async def test_maybe_refresh_computes_when_no_existing_record(tmp_path):
    aclient = AsyncQdrantClient(":memory:")
    await _seed_collection(aclient, n=5)
    store = CentroidStore(tmp_path / "centroids")

    refreshed = await maybe_refresh(aclient, COLLECTION, store)

    assert refreshed is True


@pytest.mark.asyncio
async def test_maybe_refresh_skips_when_delta_below_threshold(tmp_path):
    aclient = AsyncQdrantClient(":memory:")
    await _seed_collection(aclient, n=10)
    store = CentroidStore(tmp_path / "centroids")
    # Pre-save a centroid with point_count=9 → delta=1, below default 200
    store.save(COLLECTION, [0.0] * DENSE_DIM, point_count=9)

    refreshed = await maybe_refresh(aclient, COLLECTION, store, delta_threshold=200)

    assert refreshed is False


@pytest.mark.asyncio
async def test_maybe_refresh_triggers_when_delta_exceeds_threshold(tmp_path):
    aclient = AsyncQdrantClient(":memory:")
    await _seed_collection(aclient, n=10)
    store = CentroidStore(tmp_path / "centroids")
    # Pre-save centroid with point_count=0 → delta=10, above threshold=5
    store.save(COLLECTION, [0.0] * DENSE_DIM, point_count=0)

    refreshed = await maybe_refresh(aclient, COLLECTION, store, delta_threshold=5)

    assert refreshed is True
    record = store.load(COLLECTION)
    assert record["point_count"] == 10


@pytest.mark.asyncio
async def test_maybe_refresh_updates_centroid_after_recompute(tmp_path):
    aclient = AsyncQdrantClient(":memory:")
    await _seed_collection(aclient, n=6)
    store = CentroidStore(tmp_path / "centroids")
    old_vec = [0.0] * DENSE_DIM
    store.save(COLLECTION, old_vec, point_count=0)

    await maybe_refresh(aclient, COLLECTION, store, delta_threshold=1)

    new_vec = store.vector(COLLECTION)
    assert new_vec != old_vec
