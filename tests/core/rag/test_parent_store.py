"""Unit 6.1: ParentStore round-trip + persistence + delete-by-doc."""

from __future__ import annotations

from claritymed.core.rag.chunking.base import ParentChunk
from claritymed.core.rag.parent_store import ParentStore


def _p(
    parent_id: str, text: str = "t", doc_id: str = "d1", index: int = 0
) -> ParentChunk:
    return ParentChunk(
        parent_id=parent_id,
        text=text,
        doc_id=doc_id,
        parent_index=index,
        metadata={"source": "statpearls"},
    )


def test_put_then_get_text(tmp_path):
    store = ParentStore(tmp_path / "docstore.json")
    store.put(_p("d1#p0", text="parent paragraph"))
    assert store.get_text("d1#p0") == "parent paragraph"


def test_get_missing_returns_none(tmp_path):
    store = ParentStore(tmp_path / "docstore.json")
    assert store.get_text("never_added") is None


def test_bulk_put_counts(tmp_path):
    store = ParentStore(tmp_path / "docstore.json")
    n = store.bulk_put([_p(f"d1#p{i}", text=f"parent {i}", index=i) for i in range(5)])
    assert n == 5
    assert len(store) == 5
    for i in range(5):
        assert store.get_text(f"d1#p{i}") == f"parent {i}"


def test_bulk_put_empty_returns_zero(tmp_path):
    store = ParentStore(tmp_path / "docstore.json")
    assert store.bulk_put([]) == 0


def test_put_is_idempotent_upsert(tmp_path):
    store = ParentStore(tmp_path / "docstore.json")
    store.put(_p("k", text="v1"))
    store.put(_p("k", text="v2"))
    assert store.get_text("k") == "v2"


def test_persist_round_trip(tmp_path):
    path = tmp_path / "docstore.json"
    store = ParentStore(path)
    store.bulk_put([_p(f"d#p{i}", text=f"t{i}", index=i) for i in range(3)])
    store.persist()
    assert path.exists()

    # Reload from disk
    store2 = ParentStore(path)
    assert len(store2) == 3
    assert store2.get_text("d#p1") == "t1"


def test_delete_removes_one(tmp_path):
    store = ParentStore(tmp_path / "docstore.json")
    store.put(_p("a"))
    store.put(_p("b"))
    assert store.delete("a") is True
    assert store.get_text("a") is None
    assert store.get_text("b") == "t"
    # delete of missing returns False, not error
    assert store.delete("never") is False


def test_delete_by_doc_id_scrubs_all_parents(tmp_path):
    store = ParentStore(tmp_path / "docstore.json")
    store.bulk_put(
        [
            _p("doc1#p0", doc_id="doc1", index=0),
            _p("doc1#p1", doc_id="doc1", index=1),
            _p("doc2#p0", doc_id="doc2", index=0),
        ]
    )
    deleted = store.delete_by_doc_id("doc1")
    assert deleted == 2
    assert store.get_text("doc1#p0") is None
    assert store.get_text("doc1#p1") is None
    assert store.get_text("doc2#p0") == "t"


def test_exists(tmp_path):
    store = ParentStore(tmp_path / "docstore.json")
    store.put(_p("present"))
    assert store.exists("present") is True
    assert store.exists("absent") is False


def test_creates_parent_dir(tmp_path):
    path = tmp_path / "nested" / "deeper" / "docstore.json"
    # Parent dirs do not exist
    assert not path.parent.exists()
    store = ParentStore(path)
    store.put(_p("k"))
    store.persist()
    assert path.exists()
