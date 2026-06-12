"""Unit 6.2: RagCollectionStore async hybrid against in-memory Qdrant."""

from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

from claritymed.core.rag.chunking.base import ChildChunk
from claritymed.core.rag.qdrant_store import RagCollectionStore

DENSE_DIM = 4


@pytest.fixture
async def aclient() -> AsyncQdrantClient:
    return AsyncQdrantClient(":memory:")


def _child(
    cid: str,
    text: str = "hello",
    parent_id: str = "p0",
    chunk_index: int = 0,
    source_uri: str | None = None,
) -> ChildChunk:
    return ChildChunk(
        child_id=cid,
        text=text,
        parent_id=parent_id,
        doc_id="d1",
        chunk_index=chunk_index,
        metadata={"source_uri": source_uri} if source_uri else {},
    )


def _vec(seed: float) -> list[float]:
    return [seed, seed, seed, seed]


# --- collection lifecycle ----------------------------------------------


async def test_ensure_collection_idempotent(aclient):
    store = RagCollectionStore(aclient, "test_col", DENSE_DIM)
    await store.ensure_collection()
    await store.ensure_collection()  # second call is no-op
    assert await aclient.collection_exists("test_col")


async def test_drop_collection(aclient):
    store = RagCollectionStore(aclient, "drop_me", DENSE_DIM)
    await store.ensure_collection()
    assert await store.drop_collection() is True
    assert await aclient.collection_exists("drop_me") is False
    assert await store.drop_collection() is False  # second drop returns False


async def test_count_returns_zero_when_missing(aclient):
    store = RagCollectionStore(aclient, "ghost", DENSE_DIM)
    assert await store.count() == 0


# --- upsert + count ---------------------------------------------------


async def test_upsert_and_count(aclient):
    store = RagCollectionStore(aclient, "ingest_test", DENSE_DIM)
    children = [_child("11111111-1111-1111-1111-111111111111", text="a")]
    dense = [_vec(0.1)]
    sparse = [{1: 0.5}]
    n = await store.upsert(children, dense, sparse, is_phi=False, can_cloud=True)
    assert n == 1
    assert await store.count() == 1


async def test_upsert_misaligned_lengths_raise(aclient):
    store = RagCollectionStore(aclient, "x", DENSE_DIM)
    with pytest.raises(ValueError):
        await store.upsert([_child("1")], [], [], is_phi=False, can_cloud=True)


async def test_upsert_empty_is_noop(aclient):
    store = RagCollectionStore(aclient, "x", DENSE_DIM)
    assert await store.upsert([], [], [], is_phi=False, can_cloud=True) == 0


async def test_payload_carries_phi_and_cloud_flags(aclient):
    store = RagCollectionStore(aclient, "phi_test", DENSE_DIM)
    children = [_child("22222222-2222-2222-2222-222222222222")]
    await store.upsert(children, [_vec(0.2)], [{1: 0.5}], is_phi=True, can_cloud=False)
    hits = await store.search_hybrid(_vec(0.2), {1: 0.5}, top_k=1)
    assert hits[0].payload["is_phi"] is True
    assert hits[0].payload["can_cloud"] is False


# --- hybrid query ----------------------------------------------------


async def _seed_three_chunks(store: RagCollectionStore) -> None:
    """Three chunks, with carefully chosen vectors so we can predict
    which one a given query will rank highest."""
    chunks = [
        _child("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaa01", text="aspirin info"),
        _child("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaa02", text="ibuprofen info"),
        _child("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaa03", text="acetaminophen info"),
    ]
    # Embedding "vectors" — orthogonal-ish so search picks the right one.
    denses = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    sparses = [{10: 0.9}, {20: 0.9}, {30: 0.9}]
    await store.upsert(chunks, denses, sparses, is_phi=False, can_cloud=True)


async def test_search_hybrid_returns_closest_match(aclient):
    store = RagCollectionStore(aclient, "match_test", DENSE_DIM)
    await _seed_three_chunks(store)
    # Query close to chunk 1 (aspirin)
    hits = await store.search_hybrid([0.95, 0.05, 0.0, 0.0], {10: 0.8}, top_k=1)
    assert len(hits) == 1
    assert "aspirin" in hits[0].text


async def test_search_hybrid_dense_only_path_still_works(aclient):
    store = RagCollectionStore(aclient, "dense_only", DENSE_DIM)
    await _seed_three_chunks(store)
    # Empty sparse — RRF still ranks via dense
    hits = await store.search_hybrid([0.95, 0.05, 0.0, 0.0], {}, top_k=2)
    assert "aspirin" in hits[0].text


async def test_search_hybrid_only_cloud_safe_filters_phi(aclient):
    store = RagCollectionStore(aclient, "phi_filter", DENSE_DIM)
    # Two chunks: one PHI (can_cloud=False), one public (can_cloud=True)
    await store.upsert(
        [_child("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaa10")],
        [_vec(0.5)],
        [{1: 0.5}],
        is_phi=True,
        can_cloud=False,
    )
    await store.upsert(
        [_child("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaa11", text="public reference")],
        [_vec(0.5)],
        [{1: 0.5}],
        is_phi=False,
        can_cloud=True,
    )
    hits = await store.search_hybrid(_vec(0.5), {1: 0.5}, top_k=5, only_cloud_safe=True)
    assert len(hits) == 1
    assert hits[0].payload["can_cloud"] is True


async def test_search_hybrid_missing_collection_returns_empty(aclient):
    store = RagCollectionStore(aclient, "never_created", DENSE_DIM)
    assert await store.search_hybrid(_vec(0.5), {1: 0.5}, top_k=5) == []


async def test_search_hybrid_top_k_zero_raises(aclient):
    store = RagCollectionStore(aclient, "x", DENSE_DIM)
    with pytest.raises(ValueError):
        await store.search_hybrid(_vec(0.5), {1: 0.5}, top_k=0)


# --- delete ------------------------------------------------------------


async def test_delete_by_doc_id(aclient):
    store = RagCollectionStore(aclient, "del_test", DENSE_DIM)
    await _seed_three_chunks(store)
    assert await store.count() == 3
    await store.delete_by_doc_id("d1")
    assert await store.count() == 0


async def test_delete_by_doc_id_missing_collection_no_op(aclient):
    store = RagCollectionStore(aclient, "ghost", DENSE_DIM)
    # Just doesn't raise
    await store.delete_by_doc_id("anything")


# --- build_qdrant_client ----------------------------------------------------


def test_build_qdrant_client_returns_remote_client():
    """build_qdrant_client always returns a server-mode client."""
    from claritymed.core.rag.qdrant_store import build_qdrant_client

    client = build_qdrant_client(url="http://localhost:6333")
    # _client is AsyncQdrantRemote in server mode; local mode was removed.
    assert "Remote" in type(client._client).__name__


def test_build_qdrant_client_missing_api_key_env_raises(monkeypatch):
    """Declaring api_key_env without setting it must fail loud."""
    from claritymed.core.rag.qdrant_store import build_qdrant_client
    from claritymed.errors import MissingApiKeyError

    monkeypatch.delenv("QDRANT_CLOUD_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        build_qdrant_client(
            url="https://cloud.example.com",
            api_key_env="QDRANT_CLOUD_KEY",
        )


def test_build_qdrant_client_reads_api_key_from_env(monkeypatch):
    """When the env var is set, the key is forwarded to the client."""
    from claritymed.core.rag.qdrant_store import build_qdrant_client

    monkeypatch.setenv("QDRANT_CLOUD_KEY", "secret-token")
    # No raise — construction completes; key flows through to AsyncQdrantClient.
    client = build_qdrant_client(
        url="https://cloud.example.com",
        api_key_env="QDRANT_CLOUD_KEY",
    )
    assert "Remote" in type(client._client).__name__


# ---------------------------------------------------------------------------
# open_local_qdrant_client — stale-lock recovery
# ---------------------------------------------------------------------------


def test_open_local_qdrant_client_clean_dir(tmp_path):
    """No stale lock present → construction succeeds first try."""
    from claritymed.core.rag.qdrant_store import open_local_qdrant_client

    user_dir = tmp_path / "qdrant"
    user_dir.mkdir()
    client = open_local_qdrant_client(user_dir)
    assert client is not None


def test_open_local_qdrant_client_removes_stale_lock(tmp_path, monkeypatch, caplog):
    """A leftover .lock with no process holding it gets cleaned and the
    second AsyncQdrantClient(path=...) call succeeds.

    Regression: prior to the recovery helper, a crashed TUI left a lock
    file that bricked every subsequent run with ``already accessed by
    another instance`` until the user manually rm'd the file.
    """
    import logging

    import claritymed.core.rag.qdrant_store as mod

    user_dir = tmp_path / "qdrant"
    user_dir.mkdir()
    lock_file = user_dir / ".lock"
    lock_file.write_text("tmp lock file")

    # Pretend the first construct fails with the qdrant "already accessed"
    # signal; the second succeeds (after the helper deletes the stale lock).
    call_count = {"n": 0}

    class _FakeClient:
        pass

    def _fake_async_qdrant_client(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError(
                f"Storage folder {user_dir} is already accessed by another "
                f"instance of Qdrant client."
            )
        return _FakeClient()

    monkeypatch.setattr(mod, "AsyncQdrantClient", _fake_async_qdrant_client)
    # No process holds the lock.
    monkeypatch.setattr(mod, "_lock_is_held_by_another_process", lambda _: False)

    with caplog.at_level(logging.WARNING, logger="claritymed.core.rag.qdrant_store"):
        client = mod.open_local_qdrant_client(user_dir)

    assert isinstance(client, _FakeClient)
    assert call_count["n"] == 2  # one failed + one retry
    assert not lock_file.exists()  # stale lock deleted
    assert any("stale lock" in r.message for r in caplog.records), [
        r.message for r in caplog.records
    ]


def test_open_local_qdrant_client_refuses_when_lock_truly_held(tmp_path, monkeypatch):
    """If lsof confirms another process holds the lock, propagate the
    original error with an actionable hint — never silently nuke a real
    lock that some live process is depending on.
    """
    import claritymed.core.rag.qdrant_store as mod

    user_dir = tmp_path / "qdrant"
    user_dir.mkdir()
    lock_file = user_dir / ".lock"
    lock_file.write_text("tmp lock file")

    def _fake_async_qdrant_client(*args, **kwargs):
        raise RuntimeError(
            f"Storage folder {user_dir} is already accessed by another "
            f"instance of Qdrant client."
        )

    monkeypatch.setattr(mod, "AsyncQdrantClient", _fake_async_qdrant_client)
    monkeypatch.setattr(mod, "_lock_is_held_by_another_process", lambda _: True)

    with pytest.raises(RuntimeError, match="Another process holds"):
        mod.open_local_qdrant_client(user_dir)

    assert lock_file.exists()  # never touched


def test_open_local_qdrant_client_other_runtime_error_propagates(tmp_path, monkeypatch):
    """Only ``already accessed`` triggers recovery — unrelated RuntimeError
    must bubble up unchanged so genuine bugs aren't masked.
    """
    import claritymed.core.rag.qdrant_store as mod

    user_dir = tmp_path / "qdrant"
    user_dir.mkdir()

    def _fake_async_qdrant_client(*args, **kwargs):
        raise RuntimeError("disk corrupted")

    monkeypatch.setattr(mod, "AsyncQdrantClient", _fake_async_qdrant_client)

    with pytest.raises(RuntimeError, match="disk corrupted"):
        mod.open_local_qdrant_client(user_dir)
