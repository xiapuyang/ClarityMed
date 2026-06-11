"""Tests for ``claritymed.stores.blob_store.BlobStore``."""

from __future__ import annotations

import hashlib

import pytest

from claritymed.errors import InvalidUserIdError
from claritymed.stores.blob_store import BlobStore, make_blob_store


def test_store_writes_content_addressable_path():
    store = BlobStore("alice")
    sha = store.store(b"hello world", "txt")
    assert sha == hashlib.sha256(b"hello world").hexdigest()
    assert store.path(sha, "txt").exists()
    assert store.path(sha, "txt").read_bytes() == b"hello world"
    # Two-level directory layout: <sha[:2]>/<sha>/content.txt
    assert store.dir(sha).name == sha
    assert store.dir(sha).parent.name == sha[:2]


def test_store_is_idempotent_no_rewrite():
    store = BlobStore("alice")
    sha1 = store.store(b"hello", "txt")
    target = store.path(sha1, "txt")
    mtime_before = target.stat().st_mtime_ns
    sha2 = store.store(b"hello", "txt")
    mtime_after = target.stat().st_mtime_ns
    assert sha1 == sha2
    assert mtime_before == mtime_after, "idempotent path must not rewrite"


def test_store_rejects_empty_bytes():
    store = BlobStore("alice")
    with pytest.raises(ValueError):
        store.store(b"", "pdf")


def test_blob_store_rejects_invalid_user_id():
    with pytest.raises(InvalidUserIdError):
        BlobStore("../etc/passwd")


def test_store_rejects_invalid_extension():
    store = BlobStore("alice")
    with pytest.raises(ValueError):
        store.store(b"hello", "")
    with pytest.raises(ValueError):
        store.store(b"hello", ".pdf")
    with pytest.raises(ValueError):
        store.store(b"hello", "sub/dir")


def test_path_rejects_bad_sha():
    store = BlobStore("alice")
    with pytest.raises(ValueError):
        store.path("not-a-sha", "pdf")


def test_ocr_done_false_when_only_partial(tmp_path):
    """``ocr_done`` must NOT return true for a half-written ocr.md without
    the ocr.json sentinel — the whole point of the sentinel is to defend
    against half-written extraction results."""
    store = BlobStore("alice")
    sha = store.store(b"hello", "txt")
    # Manually drop ocr.md but no ocr.json yet.
    store.ocr_path(sha).write_text("# partial", encoding="utf-8")
    assert not store.ocr_done(sha)
    store.ocr_meta_path(sha).write_text("{}", encoding="utf-8")
    assert store.ocr_done(sha)


def test_exists_filters_partial_writes():
    store = BlobStore("alice")
    sha = store.store(b"abc", "pdf")
    assert store.exists(sha)
    # Synthetic .tmp file should be ignored by exists().
    (store.dir(sha) / "content.pdf.tmp").write_bytes(b"x")
    assert store.exists(sha)


def test_factory_returns_blob_store():
    assert isinstance(make_blob_store("alice"), BlobStore)


def test_blob_store_dedupes_across_extensions_via_sha():
    """Same bytes → same sha → same blob dir regardless of caller-passed ext.

    The second call's ``ext`` is honored at the per-file level (we write
    ``content.<ext>``), but the deduplication key is the bytes' sha. So
    storing the same bytes with two different exts produces two files in
    one directory — not two directories."""
    store = BlobStore("alice")
    sha = store.store(b"same bytes", "txt")
    sha2 = store.store(b"same bytes", "txt")  # exact dedup
    assert sha == sha2
    assert store.dir(sha).exists()


def test_disk_full_simulation_cleans_tmp(monkeypatch):
    """If write_bytes fails, the .tmp file must be removed; original path stays absent."""
    store = BlobStore("alice")

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
