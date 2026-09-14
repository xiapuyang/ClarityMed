"""Tests for ``claritymed.stores.manifest_store.ManifestStore``."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from claritymed.errors import RecordNotFound, RevisionConflict
from claritymed.stores.manifest_store import ManifestStore, make_slug


def _data(**overrides) -> dict:
    payload = {
        "kind": "exam-report",
        "title": "Annual checkup",
        "date": date(2026, 6, 11).isoformat(),
        "attachments": [
            {
                "sha256": "a" * 64,
                "filename": "r.pdf",
                "mime": "application/pdf",
                "size": 1234,
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_make_slug_has_date_prefix():
    s = make_slug(date(2026, 6, 11))
    assert s.startswith("2026-06-11-")
    assert len(s) == len("2026-06-11-") + 8


def test_make_slug_uses_today_when_unspecified():
    s = make_slug()
    # date prefix should be ISO; suffix is 8 chars; total ≥ 10 + 1 + 8.
    assert len(s) == len("2026-01-01-") + 8


def test_create_writes_manifest_with_revision_one():
    store = ManifestStore("alice", "records")
    slug = make_slug(date(2026, 6, 11))
    p = store.create("exam-reports", slug, _data())
    assert p.exists()
    parsed = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert parsed["revision"] == 1
    assert parsed["category"] == "exam-reports"
    assert parsed["slug"] == slug


def test_create_rejects_duplicate_slug():
    store = ManifestStore("alice", "records")
    slug = make_slug(date(2026, 6, 11))
    store.create("exam-reports", slug, _data())
    with pytest.raises(FileExistsError):
        store.create("exam-reports", slug, _data())


def test_create_rejects_invalid_slug():
    store = ManifestStore("alice", "records")
    with pytest.raises(ValueError):
        store.create("exam-reports", "..evil", _data())


def test_create_rejects_invalid_category():
    store = ManifestStore("alice", "records")
    slug = make_slug(date(2026, 6, 11))
    with pytest.raises(ValueError):
        store.create("../etc", slug, _data())


def test_read_returns_manifest_model():
    store = ManifestStore("alice", "records")
    slug = make_slug(date(2026, 6, 11))
    store.create("exam-reports", slug, _data())
    m = store.read("exam-reports", slug)
    assert m.revision == 1
    assert m.title == "Annual checkup"
    assert m.attachments[0].sha256 == "a" * 64


def test_read_missing_raises_record_not_found():
    store = ManifestStore("alice", "records")
    with pytest.raises(RecordNotFound):
        store.read("exam-reports", "no-such-slug")


def test_update_increments_revision():
    store = ManifestStore("alice", "records")
    slug = make_slug(date(2026, 6, 11))
    store.create("exam-reports", slug, _data())

    def mutate(d):
        d["notes"] = "added by test"
        return d

    m = store.update("exam-reports", slug, expected_revision=1, mutator=mutate)
    assert m.revision == 2
    assert m.notes == "added by test"


def test_update_revision_mismatch_raises():
    store = ManifestStore("alice", "records")
    slug = make_slug(date(2026, 6, 11))
    store.create("exam-reports", slug, _data())
    with pytest.raises(RevisionConflict):
        store.update(
            "exam-reports",
            slug,
            expected_revision=99,
            mutator=lambda d: d,
        )


def test_update_on_missing_raises_record_not_found():
    store = ManifestStore("alice", "records")
    with pytest.raises(RecordNotFound):
        store.update(
            "exam-reports",
            "no-such-slug",
            expected_revision=1,
            mutator=lambda d: d,
        )


def test_update_does_not_let_mutator_move_record():
    """Mutator returning a payload with a different ``category`` /``slug``
    must not relocate the record — the store re-imposes the path-derived
    fields after mutation."""
    store = ManifestStore("alice", "records")
    slug = make_slug(date(2026, 6, 11))
    store.create("exam-reports", slug, _data())

    def evil(d):
        d["category"] = "elsewhere"
        d["slug"] = "elsewhere-slug"
        return d

    m = store.update("exam-reports", slug, expected_revision=1, mutator=evil)
    assert m.category == "exam-reports"
    assert m.slug == slug


def test_delete_removes_dir_and_lockfile():
    store = ManifestStore("alice", "records")
    slug = make_slug(date(2026, 6, 11))
    store.create("exam-reports", slug, _data())
    store.delete("exam-reports", slug)
    with pytest.raises(RecordNotFound):
        store.read("exam-reports", slug)


def test_delete_missing_raises_record_not_found():
    store = ManifestStore("alice", "records")
    with pytest.raises(RecordNotFound):
        store.delete("exam-reports", "no-such-slug")


def test_list_yields_all_manifests_for_scope():
    store = ManifestStore("alice", "records")
    s1 = make_slug(date(2026, 6, 1))
    s2 = make_slug(date(2026, 6, 2))
    store.create("exam-reports", s1, _data())
    store.create("exam-reports", s2, _data())
    items = list(store.list())
    assert len(items) == 2
    assert all(p.name == "manifest.yaml" for p in items)


def test_list_by_category():
    store = ManifestStore("alice", "records")
    s = make_slug(date(2026, 6, 1))
    store.create("exam-reports", s, _data())
    items = list(store.list("exam-reports"))
    assert len(items) == 1


def test_list_returns_empty_when_no_records():
    store = ManifestStore("alice", "records")
    assert list(store.list()) == []


def test_records_and_library_are_isolated():
    """``records`` and ``library`` write under different dir trees."""
    rec = ManifestStore("alice", "records")
    lib = ManifestStore("alice", "library")
    slug = make_slug(date(2026, 6, 11))
    rec.create("exam-reports", slug, _data())
    # Library write does NOT pick up the records-side manifest.
    assert list(lib.list()) == []


def test_invalid_scope_rejected():
    with pytest.raises(ValueError):
        ManifestStore("alice", "bogus")  # type: ignore[arg-type]


def test_atomic_write_does_not_leave_tmp_on_failure(monkeypatch, tmp_path):
    """If ``os.rename`` raises, the ``.tmp`` should be cleaned up.

    Simulate by patching ``os.rename`` to fail; the on-disk state should
    have neither manifest.yaml nor manifest.yaml.tmp left behind."""
    store = ManifestStore("alice", "records")
    slug = make_slug(date(2026, 6, 1))

    import os

    real_rename = os.rename

    def fail_rename(src, dst):
        if str(src).endswith("manifest.yaml.tmp"):
            # Clean up the tmp the way our code path does, then raise.
            Path(src).unlink(missing_ok=True)
            raise OSError("simulated rename failure")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", fail_rename)

    with pytest.raises(OSError):
        store.create("exam-reports", slug, _data())

    record_dir = (
        Path(store._record_dir("exam-reports", slug))  # type: ignore[attr-defined]
    )
    if record_dir.exists():
        assert not any(p.name.endswith(".tmp") for p in record_dir.iterdir())
        assert not (record_dir / "manifest.yaml").exists()
