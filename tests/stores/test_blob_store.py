"""Tests for ``claritymed.stores.blob_store.BlobStore``."""

from __future__ import annotations

import hashlib

import pytest

from claritymed.errors import InvalidUserIdError
from claritymed.stores.blob_store import BlobStore, make_blob_store


def test_store_writes_content_addressable_path():
    store = BlobStore("test")
    sha = store.store(b"hello world", "txt")
    assert sha == hashlib.sha256(b"hello world").hexdigest()
    assert store.path(sha, "txt").exists()
    assert store.path(sha, "txt").read_bytes() == b"hello world"
    # Two-level directory layout: <sha[:2]>/<sha>/content.txt
    assert store.dir(sha).name == sha
    assert store.dir(sha).parent.name == sha[:2]


def test_store_is_idempotent_no_rewrite():
    store = BlobStore("test")
    sha1 = store.store(b"hello", "txt")
    target = store.path(sha1, "txt")
    mtime_before = target.stat().st_mtime_ns
    sha2 = store.store(b"hello", "txt")
    mtime_after = target.stat().st_mtime_ns
    assert sha1 == sha2
    assert mtime_before == mtime_after, "idempotent path must not rewrite"


def test_store_rejects_empty_bytes():
    store = BlobStore("test")
    with pytest.raises(ValueError):
        store.store(b"", "pdf")


def test_blob_store_rejects_invalid_user_id():
    with pytest.raises(InvalidUserIdError):
        BlobStore("../etc/passwd")


def test_store_rejects_invalid_extension():
    store = BlobStore("test")
    with pytest.raises(ValueError):
        store.store(b"hello", "")
    with pytest.raises(ValueError):
        store.store(b"hello", ".pdf")
    with pytest.raises(ValueError):
        store.store(b"hello", "sub/dir")


def test_path_rejects_bad_sha():
    store = BlobStore("test")
    with pytest.raises(ValueError):
        store.path("not-a-sha", "pdf")


def test_ocr_done_false_when_only_partial(tmp_path):
    """``ocr_done`` must NOT return true for a half-written ocr.md without
    the ocr.json sentinel — the whole point of the sentinel is to defend
    against half-written extraction results."""
    store = BlobStore("test")
    sha = store.store(b"hello", "txt")
    # Manually drop ocr.md but no ocr.json yet.
    store.ocr_path(sha).write_text("# partial", encoding="utf-8")
    assert not store.ocr_done(sha)
    store.ocr_meta_path(sha).write_text("{}", encoding="utf-8")
    assert store.ocr_done(sha)


def test_write_ocr_result_ocr_kind_writes_md_and_sentinel(tmp_path):
    """kind=ocr lands both ocr.md + ocr.json; sentinel carries kind/ext."""
    import json

    store = BlobStore("test")
    sha = store.store(b"%PDF fake", "pdf")
    store.write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext="pdf",
        provider="pymupdf",
        chain_tried=["pymupdf"],
        reason=None,
        text="extracted body",
    )
    assert store.ocr_path(sha).read_text(encoding="utf-8") == "extracted body"
    sentinel = json.loads(store.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel == {
        "status": "done",
        "kind": "ocr",
        "ext": "pdf",
        "provider": "pymupdf",
        "chain_tried": ["pymupdf"],
        "reason": None,
        "chars": len("extracted body"),
    }
    assert store.ocr_done(sha)


def test_write_ocr_result_text_kind_skips_ocr_md(tmp_path):
    """kind=text writes only the sentinel — content.<ext> already is the
    extracted text, so duplicating it as ocr.md would waste bytes."""
    import json

    store = BlobStore("test")
    sha = store.store(b"col1,col2\n1,2\n", "csv")
    store.write_ocr_result(
        sha,
        status="done",
        kind="text",
        ext="csv",
        provider="text",
        chain_tried=["text"],
        reason=None,
        text="col1,col2\n1,2\n",
    )
    # ocr.md is NOT written — that's the whole point of text kind.
    assert not store.ocr_path(sha).exists()
    # Sentinel records kind + ext so readers know to consult content.csv.
    sentinel = json.loads(store.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["kind"] == "text"
    assert sentinel["ext"] == "csv"
    assert store.ocr_done(sha)


def test_write_ocr_result_records_failure(tmp_path):
    """Failure path: empty text, populated reason, no provider. ocr.md
    is still written (empty) so any stale orphan is overwritten."""
    import json

    store = BlobStore("test")
    sha = store.store(b"some pdf", "pdf")
    store.write_ocr_result(
        sha,
        status="failed",
        kind="ocr",
        ext="pdf",
        provider=None,
        chain_tried=[],
        reason="all providers exhausted",
        text="",
    )
    sentinel = json.loads(store.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["status"] == "failed"
    assert sentinel["provider"] is None
    assert sentinel["chain_tried"] == []
    assert sentinel["chars"] == 0


def test_read_extracted_text_text_kind_reads_content_file(tmp_path):
    """For kind=text, the reader pulls from content.<ext> directly,
    not from ocr.md (which doesn't exist for text blobs)."""
    store = BlobStore("test")
    payload = "col1,col2\n1,2\n3,4\n"
    sha = store.store(payload.encode("utf-8"), "csv")
    store.write_ocr_result(
        sha,
        status="done",
        kind="text",
        ext="csv",
        provider="text",
        chain_tried=["text"],
        reason=None,
        text=payload,
    )
    assert store.read_extracted_text(sha) == payload


def test_read_extracted_text_ocr_kind_reads_ocr_md(tmp_path):
    """For kind=ocr, the reader pulls from ocr.md (the OCR output)."""
    store = BlobStore("test")
    sha = store.store(b"%PDF fake", "pdf")
    store.write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext="pdf",
        provider="pymupdf",
        chain_tried=["pymupdf"],
        reason=None,
        text="OCR output text",
    )
    assert store.read_extracted_text(sha) == "OCR output text"


def test_exists_filters_partial_writes():
    store = BlobStore("test")
    sha = store.store(b"abc", "pdf")
    assert store.exists(sha)
    # Synthetic .tmp file should be ignored by exists().
    (store.dir(sha) / "content.pdf.tmp").write_bytes(b"x")
    assert store.exists(sha)


def test_factory_returns_blob_store():
    assert isinstance(make_blob_store("test"), BlobStore)


def test_blob_store_dedupes_across_extensions_via_sha():
    """Same bytes → same sha → same blob dir regardless of caller-passed ext.

    The second call's ``ext`` is honored at the per-file level (we write
    ``content.<ext>``), but the deduplication key is the bytes' sha. So
    storing the same bytes with two different exts produces two files in
    one directory — not two directories."""
    store = BlobStore("test")
    sha = store.store(b"same bytes", "txt")
    sha2 = store.store(b"same bytes", "txt")  # exact dedup
    assert sha == sha2
    assert store.dir(sha).exists()


def test_disk_full_simulation_cleans_tmp(monkeypatch):
    """If write_bytes fails, the .tmp file must be removed; original path stays absent."""
    store = BlobStore("test")

    def boom(self, data):
        # Create the file then raise — simulates partial disk-full where
        # the tmp landed but rename never happened.
        from pathlib import Path  # noqa: PLC0415

        Path.write_bytes_original(self, data)  # type: ignore[attr-defined]
        raise OSError("disk full")

    from pathlib import Path

    Path.write_bytes_original = Path.write_bytes  # type: ignore[attr-defined]
    monkeypatch.setattr(Path, "write_bytes", boom)
    with pytest.raises(OSError):
        store.store(b"will fail", "pdf")
    # No content.pdf should be left behind.
    sha = hashlib.sha256(b"will fail").hexdigest()
    assert not store.path(sha, "pdf").exists()
    assert not store.path(sha, "pdf").with_suffix(".pdf.tmp").exists()


# --- original_filename in ocr.json (disaster-recovery anchor) -------


def test_ocr_sentinel_carries_original_filename(tmp_path):
    """A paste-supplied filename lands in ``ocr.json`` so disaster
    recovery from the blob CAS alone has a name to show the user."""
    import json

    store = BlobStore("test")
    sha = store.store(b"%PDF fake", "pdf")
    store.write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext="pdf",
        provider="pymupdf",
        chain_tried=["pymupdf"],
        reason=None,
        text="extracted body",
        original_filename="Lab Report 2026.pdf",
    )
    sentinel = json.loads(store.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["original_filename"] == "Lab Report 2026.pdf"


def test_ocr_sentinel_omits_filename_field_when_none(tmp_path):
    """Caller didn't supply a name → the field is absent (not ``null``).

    Keeps the sentinel slim for headless / migrated blobs without a
    known display name, and matches the back-compat shape so old
    parsers don't see a surprise key."""
    import json

    store = BlobStore("test")
    sha = store.store(b"data", "pdf")
    store.write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext="pdf",
        provider="pymupdf",
        chain_tried=["pymupdf"],
        reason=None,
        text="x",
    )
    sentinel = json.loads(store.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert "original_filename" not in sentinel


def test_ocr_sentinel_first_write_wins_on_reuse(tmp_path):
    """Re-extracting the same blob keeps the *first* name on disk.

    Blob CAS dedupes by bytes — the same PDF re-pasted under a
    different display name shares the sentinel. SessionAttachments
    is the per-upload authority for naming; ocr.json is just an
    anchor pointing at "what the user first called this blob."
    """
    import json

    store = BlobStore("test")
    sha = store.store(b"%PDF fake", "pdf")
    store.write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext="pdf",
        provider="pymupdf",
        chain_tried=["pymupdf"],
        reason=None,
        text="x",
        original_filename="first.pdf",
    )
    store.write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext="pdf",
        provider="pymupdf",
        chain_tried=["pymupdf"],
        reason=None,
        text="x",
        original_filename="second.pdf",
    )
    sentinel = json.loads(store.ocr_meta_path(sha).read_text(encoding="utf-8"))
    assert sentinel["original_filename"] == "first.pdf"


def test_ocr_sentinel_chinese_filename_not_escaped(tmp_path):
    """Chinese names land verbatim on disk — that's the whole point of
    ensure_ascii=False, otherwise the sentinel becomes unreadable
    ``\\u5316\\u9a8c\\u5355`` and grepping for a name fails."""
    store = BlobStore("test")
    sha = store.store(b"%PDF fake", "pdf")
    store.write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext="pdf",
        provider="pymupdf",
        chain_tried=["pymupdf"],
        reason=None,
        text="x",
        original_filename="化验单_2026年6月.pdf",
    )
    raw = store.ocr_meta_path(sha).read_text(encoding="utf-8")
    assert "化验单_2026年6月.pdf" in raw
    # Negative guard: the escape form must NOT appear.
    assert "\\u5316" not in raw
