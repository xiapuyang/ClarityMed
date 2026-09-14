"""Behavior tests for ``LlamaIndexParentChildChunker``.

LlamaIndex's HierarchicalNodeParser is the upstream that does the actual
splitting; these tests cover the adapter's responsibilities:

* Output shape (separate parents + children, parent_id mapping)
* CJK handling via the upstream's secondary regex (no separate code path)
* Edge cases (empty input, single-tiny doc, malformed config)
* Deterministic id generation across re-chunking of the same input
"""

from __future__ import annotations

import pytest

from claritymed.core.rag.chunking import (
    Chunker,
    LlamaIndexParentChildChunker,
    RawDocument,
    build_chunker,
)
from claritymed.core.rag.schemas import ChunkerConfig
from claritymed.errors import UnknownChunkerError


def _make() -> LlamaIndexParentChildChunker:
    return LlamaIndexParentChildChunker(parent_tok=200, child_tok=80, overlap_tok=10)


# --- happy path -------------------------------------------------------


def test_chunker_implements_protocol():
    assert isinstance(_make(), Chunker)


def test_long_english_doc_produces_parents_and_children():
    text = (
        "Aspirin is commonly used to relieve mild to moderate pain. "
        "It is also used to reduce fever and inflammation. "
        "Patients with bleeding disorders should consult their physician. "
        "Common side effects include stomach upset and bruising. "
        "Severe side effects are rare but can include gastrointestinal bleeding. "
        "Long-term use should be monitored by a healthcare professional. "
        "Aspirin should not be given to children under 16 due to Reye's syndrome risk. "
        "Pregnant patients should avoid aspirin in the third trimester. "
    ) * 6  # large enough to force multiple parents
    doc = RawDocument(doc_id="aspirin_1", text=text)

    chunked = _make().chunk(doc)
    assert len(chunked.parents) >= 1
    assert len(chunked.children) >= len(chunked.parents)
    # Every child must point at a known parent.
    parent_ids = {p.parent_id for p in chunked.parents}
    for c in chunked.children:
        assert c.parent_id in parent_ids


def test_parent_ids_are_doc_scoped_and_ordered():
    text = "Sentence one. " * 100
    chunked = _make().chunk(RawDocument(doc_id="dx", text=text))
    indices = [p.parent_index for p in chunked.parents]
    assert indices == list(range(len(chunked.parents)))
    for p in chunked.parents:
        assert p.parent_id == f"dx#p{p.parent_index}"


def test_child_chunk_index_resets_per_parent():
    text = "Sentence one. " * 200
    chunked = _make().chunk(RawDocument(doc_id="dx", text=text))
    # Group children by parent; chunk_index in each group must start at 0.
    by_parent: dict[str, list[int]] = {}
    for c in chunked.children:
        by_parent.setdefault(c.parent_id, []).append(c.chunk_index)
    for indices in by_parent.values():
        assert indices == list(range(len(indices)))


def test_short_doc_produces_one_parent_no_split():
    doc = RawDocument(doc_id="short", text="Hello world. This is a short note.")
    chunked = _make().chunk(doc)
    assert len(chunked.parents) == 1
    # Child count may be 1 (LlamaIndex may emit a single child) — covered
    # by the "every child points at a parent" invariant; we only assert
    # the parent count here.


def test_chinese_input_handled_via_secondary_regex():
    # If the secondary regex (CJK punctuation) is wired correctly, a
    # paragraph of Chinese sentences chunks into more than one node.
    text = (
        "阿司匹林是一种常用的解热镇痛药。"
        "它可以缓解轻度到中度的疼痛。"
        "也可以用于退烧和减轻炎症。"
        "出血性疾病患者应当谨慎使用。"
        "常见副作用包括胃部不适和瘀斑。"
        "严重副作用罕见，但可能包括胃肠道出血。"
    ) * 8
    chunked = _make().chunk(RawDocument(doc_id="zh", text=text, language="zh"))
    assert len(chunked.parents) >= 1
    assert len(chunked.children) >= 1


# --- edge cases ------------------------------------------------------


def test_empty_doc_returns_empty_chunked_document():
    chunked = _make().chunk(RawDocument(doc_id="e", text=""))
    assert chunked.parents == []
    assert chunked.children == []


def test_rejects_parent_tok_le_child_tok():
    with pytest.raises(ValueError):
        LlamaIndexParentChildChunker(parent_tok=100, child_tok=100, overlap_tok=0)


def test_rejects_negative_overlap():
    with pytest.raises(ValueError):
        LlamaIndexParentChildChunker(parent_tok=200, child_tok=80, overlap_tok=-1)


def test_metadata_is_propagated_to_chunks():
    chunked = _make().chunk(
        RawDocument(
            doc_id="dx",
            text="Sentence. " * 80,
            metadata={"source": "statpearls", "license": "CC"},
        )
    )
    for parent in chunked.parents:
        assert parent.metadata["source"] == "statpearls"
        assert parent.metadata["license"] == "CC"
    for child in chunked.children:
        assert child.metadata["source"] == "statpearls"


# --- determinism ----------------------------------------------------


def test_repeated_chunking_yields_stable_ids():
    text = "Sentence one. " * 100
    doc = RawDocument(doc_id="stable", text=text)
    out_a = _make().chunk(doc)
    out_b = _make().chunk(doc)
    assert [p.parent_id for p in out_a.parents] == [p.parent_id for p in out_b.parents]
    assert [c.child_id for c in out_a.children] == [c.child_id for c in out_b.children]


# --- factory --------------------------------------------------------


def test_build_chunker_factory_happy_path():
    cfg = ChunkerConfig(
        active="parent_child",
        catalog=[
            {  # type: ignore[list-item]
                "id": "parent_child",
                "child_tok": 80,
                "parent_tok": 200,
                "overlap_tok": 10,
            }
        ],
    )
    assert isinstance(build_chunker(cfg), LlamaIndexParentChildChunker)


def test_build_chunker_unknown_id_raises():
    from claritymed.core.rag.schemas import ChunkerConfig, ParentChildChunkerConfig

    entry = ParentChildChunkerConfig.model_construct(
        id="raptor",
        child_tok=100,
        parent_tok=400,
        overlap_tok=10,
    )
    cfg = ChunkerConfig.model_construct(active="raptor", catalog=[entry])
    with pytest.raises(UnknownChunkerError):
        build_chunker(cfg)
