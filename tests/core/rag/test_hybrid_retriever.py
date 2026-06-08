"""Unit 6.3: HybridRetriever orchestrator end-to-end.

Real qdrant ``:memory:``, real ParentStore (LlamaIndex SimpleDocumentStore),
real CollectionRouter, real TermService (NoOp for simplicity); mock embedder
+ reranker so we control ordering and exercise fail-soft paths.
"""

from __future__ import annotations


import pytest
from qdrant_client import AsyncQdrantClient

from claritymed.core.rag.chunking.base import ChildChunk, ParentChunk
from claritymed.core.rag.embedding.base import Embedder, SparseVector
from claritymed.core.rag.parent_store import ParentStore
from claritymed.core.rag.qdrant_store import RagCollectionStore
from claritymed.core.rag.reranking.base import RerankHit, Reranker
from claritymed.core.rag.retriever import HybridRetriever
from claritymed.core.rag.routing.collection_router import CollectionRouter
from claritymed.core.rag.schemas import CollectionMetadata, RouterEntry
from claritymed.core.rag.terms.umls_cmekg import NoOpTermService
from claritymed.errors import RerankerUnreachableError

DENSE_DIM = 4


# --- stubs --------------------------------------------------------------


class StubEmbedder(Embedder):
    """Deterministic embedder. text → (sum of char codes mod 7) → vector."""

    @property
    def dimension(self) -> int:
        return DENSE_DIM

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [{abs(hash(t)) % 100: 0.5} for t in texts]

    @staticmethod
    def _vec(t: str) -> list[float]:
        seed = (sum(ord(c) for c in t) % 7) / 7.0
        return [seed, seed * 0.5, 1.0 - seed, 0.1]


class StubReranker(Reranker):
    """Re-ranks by string length descending — predictable for assertions."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, int]] = []

    async def rerank(self, query: str, docs: list[str], top_k: int) -> list[RerankHit]:
        self.calls.append((query, len(docs)))
        if self.fail:
            raise RerankerUnreachableError("simulated rerank failure")
        scored = sorted(
            ((i, float(len(d))) for i, d in enumerate(docs)),
            key=lambda t: -t[1],
        )
        return [RerankHit(index=i, score=s) for i, s in scored[:top_k]]


# --- fixtures -----------------------------------------------------------


@pytest.fixture
async def aclient() -> AsyncQdrantClient:
    return AsyncQdrantClient(":memory:")


@pytest.fixture
def shared_parent_store(tmp_path) -> ParentStore:
    return ParentStore(tmp_path / "shared_docstore.json")


def _meta(name: str, language: str = "en", tier: int = 1) -> CollectionMetadata:
    return CollectionMetadata(
        name=name,
        language=language,
        cross_lingual=True,
        authority_tier=tier,
        size_chunks=10,
    )


def _router(catalog: list[CollectionMetadata]) -> CollectionRouter:
    return CollectionRouter(
        catalog=catalog,
        config=RouterEntry(
            id="rule_based", max_active=3, authority_bias={1: 0.0, 2: 0.2, 3: 0.5}
        ),
    )


async def _seed(
    store: RagCollectionStore,
    parent_store: ParentStore,
    *,
    embedder: Embedder,
    items: list[tuple[str, str, str, str]],
    is_phi: bool = False,
    can_cloud: bool = True,
) -> None:
    """``items`` is [(child_id, text, parent_id, parent_text)]."""
    children = [
        ChildChunk(
            child_id=cid,
            text=text,
            parent_id=pid,
            doc_id="d1",
            chunk_index=i,
        )
        for i, (cid, text, pid, _) in enumerate(items)
    ]
    parent_store.bulk_put(
        [
            ParentChunk(
                parent_id=pid,
                text=ptext,
                doc_id="d1",
                parent_index=i,
            )
            for i, (_, _, pid, ptext) in enumerate(items)
        ]
    )
    parent_store.persist()
    dense = await embedder.embed_dense([t for _, t, _, _ in items])
    sparse = await embedder.embed_sparse([t for _, t, _, _ in items])
    await store.upsert(children, dense, sparse, is_phi=is_phi, can_cloud=can_cloud)


def _make_retriever(
    *,
    aclient: AsyncQdrantClient,
    shared_parent_store: ParentStore,
    user_parent_store: ParentStore | None = None,
    router: CollectionRouter,
    embedder: StubEmbedder,
    reranker: StubReranker,
) -> HybridRetriever:
    def system_factory(name: str) -> RagCollectionStore:
        return RagCollectionStore(aclient, name, DENSE_DIM)

    async def user_factory(user_id: str) -> RagCollectionStore | None:
        return RagCollectionStore(aclient, f"user_rag_{user_id}", DENSE_DIM)

    def user_parent_factory(user_id: str) -> ParentStore | None:
        return user_parent_store

    return HybridRetriever(
        embedder=embedder,
        reranker=reranker,
        term_service=NoOpTermService(),
        router=router,
        system_store_factory=system_factory,
        system_parent_store=shared_parent_store,
        user_store_factory=user_factory,
        user_parent_store_factory=user_parent_factory,
        rerank_top_k=3,
    )


# --- end-to-end --------------------------------------------------------


async def test_retrieve_system_only_hydrates_parent(
    aclient, shared_parent_store, tmp_path
):
    sys_store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    embedder = StubEmbedder()
    reranker = StubReranker()
    await _seed(
        sys_store,
        shared_parent_store,
        embedder=embedder,
        items=[
            (
                "11111111-1111-1111-1111-111111111111",
                "short",
                "p0",
                "PARENT_TEXT_FOR_P0",
            ),
            (
                "22222222-2222-2222-2222-222222222222",
                "a longer chunk text",
                "p1",
                "PARENT_TEXT_FOR_P1",
            ),
            (
                "33333333-3333-3333-3333-333333333333",
                "the longest chunk text here",
                "p2",
                "PARENT_TEXT_FOR_P2",
            ),
        ],
    )
    retriever = _make_retriever(
        aclient=aclient,
        shared_parent_store=shared_parent_store,
        router=_router([_meta("statpearls_en")]),
        embedder=embedder,
        reranker=reranker,
    )

    bundle = await retriever.retrieve("aspirin", language="en", user_id="alice")
    assert len(bundle.chunks) == 3
    # Reranker (length-desc) puts the longest chunk first.
    assert bundle.chunks[0].text.startswith("the longest")
    assert bundle.chunks[0].parent_text == "PARENT_TEXT_FOR_P2"
    assert bundle.chunks[0].collection_name == "statpearls_en"
    assert bundle.chunks[0].rerank_score is not None
    assert bundle.chunks[0].source == "system_rag"
    assert bundle.chunks[0].is_phi is False


async def test_retrieve_merges_system_and_user(aclient, shared_parent_store, tmp_path):
    sys_store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    user_store = RagCollectionStore(aclient, "user_rag_alice", DENSE_DIM)
    user_parent_store = ParentStore(tmp_path / "user_docstore.json")
    embedder = StubEmbedder()
    reranker = StubReranker()
    await _seed(
        sys_store,
        shared_parent_store,
        embedder=embedder,
        items=[
            (
                "11111111-1111-1111-1111-111111111111",
                "system chunk",
                "sp0",
                "SYSTEM_PARENT",
            )
        ],
    )
    await _seed(
        user_store,
        user_parent_store,
        embedder=embedder,
        items=[
            (
                "22222222-2222-2222-2222-222222222222",
                "personal upload chunk",
                "up0",
                "USER_PARENT",
            )
        ],
        is_phi=True,
        can_cloud=False,
    )

    retriever = _make_retriever(
        aclient=aclient,
        shared_parent_store=shared_parent_store,
        user_parent_store=user_parent_store,
        router=_router([_meta("statpearls_en")]),
        embedder=embedder,
        reranker=reranker,
    )
    bundle = await retriever.retrieve("query", language="en", user_id="alice")
    sources = {c.source for c in bundle.chunks}
    assert sources == {"system_rag", "user_rag"}
    user_chunk = next(c for c in bundle.chunks if c.source == "user_rag")
    assert user_chunk.is_phi is True
    assert user_chunk.can_cloud is False
    assert user_chunk.parent_text == "USER_PARENT"
    assert user_chunk.collection_name == "user_rag_alice"


async def test_retrieve_empty_active_returns_empty_bundle_with_trace(
    aclient, shared_parent_store
):
    embedder = StubEmbedder()
    reranker = StubReranker()
    retriever = _make_retriever(
        aclient=aclient,
        shared_parent_store=shared_parent_store,
        router=_router([_meta("statpearls_en")]),
        embedder=embedder,
        reranker=reranker,
    )
    # Empty whitelist = user opted out → no system hits; user_rag also empty
    bundle = await retriever.retrieve(
        "any", language="en", user_id="alice", user_whitelist=[]
    )
    assert bundle.chunks == []
    assert bundle.trace.active_collections == []


async def test_retrieve_reranker_failure_falls_back_to_rrf_order(
    aclient, shared_parent_store
):
    sys_store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    embedder = StubEmbedder()
    reranker = StubReranker(fail=True)
    await _seed(
        sys_store,
        shared_parent_store,
        embedder=embedder,
        items=[
            ("11111111-1111-1111-1111-111111111111", "alpha", "p0", "P0"),
            ("22222222-2222-2222-2222-222222222222", "beta", "p1", "P1"),
        ],
    )
    retriever = _make_retriever(
        aclient=aclient,
        shared_parent_store=shared_parent_store,
        router=_router([_meta("statpearls_en")]),
        embedder=embedder,
        reranker=reranker,
    )
    bundle = await retriever.retrieve("q", language="en", user_id="alice")
    # Still returns hits (RRF order), just without rerank_score
    assert len(bundle.chunks) > 0
    for chunk in bundle.chunks:
        assert chunk.rerank_score is None


async def test_retrieve_missing_parent_degrades_gracefully(
    aclient, shared_parent_store
):
    sys_store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    embedder = StubEmbedder()
    reranker = StubReranker()
    # Seed child with parent_id 'orphan' that we DO NOT put in docstore.
    children = [
        ChildChunk(
            child_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaa01",
            text="orphan child",
            parent_id="orphan",
            doc_id="d1",
            chunk_index=0,
        )
    ]
    dense = await embedder.embed_dense(["orphan child"])
    sparse = await embedder.embed_sparse(["orphan child"])
    await sys_store.upsert(children, dense, sparse, is_phi=False, can_cloud=True)

    retriever = _make_retriever(
        aclient=aclient,
        shared_parent_store=shared_parent_store,
        router=_router([_meta("statpearls_en")]),
        embedder=embedder,
        reranker=reranker,
    )
    bundle = await retriever.retrieve("q", language="en", user_id="alice")
    assert len(bundle.chunks) == 1
    assert bundle.chunks[0].parent_text is None
    # Child text is still present so the prompt assembler can fall back
    assert bundle.chunks[0].text == "orphan child"


async def test_trace_records_active_collections_and_timings(
    aclient, shared_parent_store
):
    sys_store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    embedder = StubEmbedder()
    reranker = StubReranker()
    await _seed(
        sys_store,
        shared_parent_store,
        embedder=embedder,
        items=[("11111111-1111-1111-1111-111111111111", "doc", "p0", "P0")],
    )
    retriever = _make_retriever(
        aclient=aclient,
        shared_parent_store=shared_parent_store,
        router=_router([_meta("statpearls_en")]),
        embedder=embedder,
        reranker=reranker,
    )
    bundle = await retriever.retrieve("query for trace", language="en", user_id="u")
    assert bundle.trace.active_collections == ["statpearls_en"]
    assert bundle.trace.strategy == "naive_hybrid"
    assert bundle.trace.embed_ms >= 0
    assert bundle.trace.search_ms >= 0
    assert bundle.trace.rerank_ms >= 0
    assert bundle.trace.fallback_triggered is False
    assert bundle.trace.grader is None


async def test_only_cloud_safe_filters_phi_chunks(
    aclient, shared_parent_store, tmp_path
):
    """When the downstream provider is cloud (kind=cloud), the retriever
    should pre-filter user_rag PHI chunks at the Qdrant query layer."""
    sys_store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    user_store = RagCollectionStore(aclient, "user_rag_alice", DENSE_DIM)
    user_parent_store = ParentStore(tmp_path / "user_docstore.json")
    embedder = StubEmbedder()
    reranker = StubReranker()
    await _seed(
        sys_store,
        shared_parent_store,
        embedder=embedder,
        items=[("11111111-1111-1111-1111-111111111111", "public", "p0", "P0")],
        can_cloud=True,
    )
    await _seed(
        user_store,
        user_parent_store,
        embedder=embedder,
        items=[
            ("22222222-2222-2222-2222-222222222222", "private upload", "up0", "UP0")
        ],
        is_phi=True,
        can_cloud=False,
    )
    retriever = _make_retriever(
        aclient=aclient,
        shared_parent_store=shared_parent_store,
        user_parent_store=user_parent_store,
        router=_router([_meta("statpearls_en")]),
        embedder=embedder,
        reranker=reranker,
    )
    bundle = await retriever.retrieve(
        "q", language="en", user_id="alice", only_cloud_safe=True
    )
    # Private user_rag chunk is gone; only public system chunk survives.
    sources = {c.source for c in bundle.chunks}
    assert sources == {"system_rag"}


async def test_reranker_called_with_original_query_not_expanded(
    aclient, shared_parent_store
):
    """Expansion broadens recall via the embedder; rerank uses the user's
    original query to keep relevance honest."""
    sys_store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    embedder = StubEmbedder()
    reranker = StubReranker()
    await _seed(
        sys_store,
        shared_parent_store,
        embedder=embedder,
        items=[("11111111-1111-1111-1111-111111111111", "doc", "p0", "P0")],
    )
    retriever = _make_retriever(
        aclient=aclient,
        shared_parent_store=shared_parent_store,
        router=_router([_meta("statpearls_en")]),
        embedder=embedder,
        reranker=reranker,
    )
    await retriever.retrieve(
        "exact original query string", language="en", user_id="alice"
    )
    assert reranker.calls == [("exact original query string", 1)]
