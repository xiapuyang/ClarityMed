"""Unit tests for MedRAG textbooks corpus adapter."""

from __future__ import annotations

import json

import pytest

from claritymed.ingest.corpus.textbooks import (
    COLLECTION_NAME,
    LANGUAGE,
    TextbooksSource,
    _doc_from_json,
)


def test_source_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        TextbooksSource(tmp_path / "nope")


def test_source_iterates_chunk_jsonl(tmp_path):
    chunk = tmp_path / "chunk"
    chunk.mkdir()
    (chunk / "First_Aid_Step1.jsonl").write_text(
        json.dumps(
            {
                "id": "First_Aid_Step1_0",
                "title": "First Aid",
                "content": "Aspirin uses ...",
                "contents": "redundant",
            }
        )
        + "\n"
        + json.dumps(
            {"id": "First_Aid_Step1_1", "title": "First Aid", "content": "second"}
        )
        + "\n",
        encoding="utf-8",
    )
    docs = list(TextbooksSource(tmp_path).iter_raw_docs())
    assert len(docs) == 2
    assert docs[0].doc_id == "First_Aid_Step1_0"
    assert docs[0].language == LANGUAGE
    assert docs[0].metadata["collection"] == COLLECTION_NAME
    assert docs[0].metadata["doc_title"] == "First Aid"
    assert docs[0].metadata["source_uri"] is None
    # Title prepended to text.
    assert docs[0].text.startswith("First Aid")
    assert "Aspirin uses" in docs[0].text


def test_source_walks_flattened_root(tmp_path):
    """No `chunk/` subdir — adapter must still pick up jsonl at root."""
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"id": "x", "title": "T", "content": "body"}),
        encoding="utf-8",
    )
    docs = list(TextbooksSource(tmp_path).iter_raw_docs())
    assert len(docs) == 1
    assert docs[0].doc_id == "x"


def test_source_warns_when_empty(tmp_path, caplog):
    docs = list(TextbooksSource(tmp_path).iter_raw_docs())
    assert docs == []
    assert any("No .jsonl files found" in rec.message for rec in caplog.records)


def test_source_skips_bad_json_line(tmp_path, caplog):
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"id": "ok", "title": "T", "content": "body"})
        + "\n"
        + "{not valid json"
        + "\n"
        + ""  # empty line
        + "\n"
        + json.dumps({"id": "ok2", "title": "T", "content": "body2"})
        + "\n",
        encoding="utf-8",
    )
    docs = list(TextbooksSource(tmp_path).iter_raw_docs())
    assert [d.doc_id for d in docs] == ["ok", "ok2"]
    assert any("bad JSON" in rec.message for rec in caplog.records)


def test_source_skips_doc_without_content(tmp_path, caplog):
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"id": "no_text", "title": "T"}) + "\n",
        encoding="utf-8",
    )
    docs = list(TextbooksSource(tmp_path).iter_raw_docs())
    assert docs == []
    assert any("missing id/content" in rec.message for rec in caplog.records)


def test_doc_from_json_no_title_skips_prefix():
    doc = _doc_from_json({"id": "x", "content": "raw body"})
    assert doc is not None
    assert doc.text == "raw body"
    assert doc.metadata["doc_title"] == ""


def test_doc_from_json_missing_id_returns_none():
    assert _doc_from_json({"content": "body"}) is None


def test_doc_from_json_missing_content_returns_none():
    assert _doc_from_json({"id": "x", "title": "T"}) is None
