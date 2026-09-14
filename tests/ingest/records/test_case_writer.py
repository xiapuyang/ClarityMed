"""Tests for ``ingest.records.case_writer``.

Two responsibilities to pin:

1. The slug formula — full-case_id sha256 prefix (not raw prefix) so
   shared-long-prefix case_ids get distinct slugs.
2. The OCR-status cache read — pre-existing OCR sentinel → ``done``,
   missing sentinel → ``pending`` (locks the single-OCR guarantee).
"""

from __future__ import annotations

from datetime import date

import pytest

from claritymed.ingest.records.case_writer import (
    SLUG_HASH_LEN,
    apply_case,
    compute_slug,
)
from claritymed.ingest.records.template_schema import CaseAttachment, CaseEntry
from claritymed.stores.account import init_user


# --- fixtures ----------------------------------------------------------


@pytest.fixture
def redirected_data_dir(monkeypatch, tmp_path):
    """Repoint CLARITYMED_DATA_DIR so the test never touches real user data."""
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path / "data"))
    import importlib

    from claritymed import config as _cfg

    importlib.reload(_cfg)
    return tmp_path


@pytest.fixture
def test_user(redirected_data_dir):
    return init_user("test", display_name="Test")


def _case(**overrides) -> CaseEntry:
    base = dict(
        case_id="notion-abcdef12",
        event_date=date(2024, 1, 15),
        title="Annual checkup",
        kind="exam-report",
        category="exam-reports",
        body_md="BP 120/80",
    )
    base.update(overrides)
    return CaseEntry(**base)


# --- compute_slug -----------------------------------------------------


def test_compute_slug_format():
    case = _case(case_id="notion-abcdef12", event_date=date(2024, 1, 15))
    slug = compute_slug(case)
    assert slug.startswith("2024-01-15-tmpl-")
    assert len(slug) == len("2024-01-15-tmpl-") + SLUG_HASH_LEN


def test_compute_slug_uses_full_case_id_not_prefix():
    """Two case_ids sharing the first 8 chars (the original ``[:8]`` formula
    would collide here) must produce distinct slugs because the hash is
    over the full case_id."""
    a = _case(case_id="notion-0db3d4f9-page", event_date=date(2024, 1, 15))
    b = _case(case_id="notion-0db3d4f9-extract", event_date=date(2024, 1, 15))
    assert compute_slug(a) != compute_slug(b)


def test_compute_slug_deterministic():
    case = _case()
    assert compute_slug(case) == compute_slug(case)


# --- apply_case happy path --------------------------------------------


def test_apply_case_writes_manifest(test_user):
    result = apply_case("test", _case())
    assert result.status == "done"
    assert result.error_detail is None

    from claritymed.stores.manifest_store import ManifestStore

    manifest = ManifestStore("test", scope="records").read("exam-reports", result.slug)
    assert manifest.title == "Annual checkup"
    assert manifest.kind == "exam-report"
    assert manifest.body == "BP 120/80"
    assert manifest.embed_status == "pending_retry"


def test_apply_case_with_no_attachments_writes_empty_attachments_list(test_user):
    result = apply_case("test", _case())
    assert result.status == "done"

    from claritymed.stores.manifest_store import ManifestStore

    manifest = ManifestStore("test", scope="records").read("exam-reports", result.slug)
    assert manifest.attachments == []


def test_apply_case_idempotent_skipped_on_rerun(test_user):
    first = apply_case("test", _case())
    assert first.status == "done"
    second = apply_case("test", _case())
    assert second.status == "skipped"
    assert second.error_detail == "already_imported"
    assert second.slug == first.slug


def test_apply_case_long_prefix_case_ids_dont_collide(test_user):
    """Two cases whose case_ids share the first 8 chars AND have the
    same event_date land at distinct slugs. Pins the prefix-collision
    fix in the slug formula."""
    a = apply_case(
        "test",
        _case(
            case_id="notion-0db3d4f9-page",
            event_date=date(2024, 1, 15),
            title="Page A",
        ),
    )
    b = apply_case(
        "test",
        _case(
            case_id="notion-0db3d4f9-extract",
            event_date=date(2024, 1, 15),
            title="Extract B",
        ),
    )
    assert a.status == "done"
    assert b.status == "done"
    assert a.slug != b.slug


# --- attachments -------------------------------------------------------


def test_apply_case_with_attachment_writes_blob_and_pending_ocr(test_user, tmp_path):
    """Default path: no pre-existing OCR sentinel → ocr_status='pending'."""
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 some-fake-pdf-bytes\n")

    case = _case(
        attachments=[
            CaseAttachment(
                path=str(pdf), original_filename="report.pdf", mime="application/pdf"
            )
        ]
    )
    result = apply_case("test", case)
    assert result.status == "done"

    from claritymed.stores.blob_store import BlobStore
    from claritymed.stores.manifest_store import ManifestStore

    manifest = ManifestStore("test", scope="records").read("exam-reports", result.slug)
    assert len(manifest.attachments) == 1
    att = manifest.attachments[0]
    assert att.ocr_status == "pending"
    assert att.mime == "application/pdf"
    assert att.filename == "report.pdf"
    assert BlobStore("test").exists(att.sha256)


def test_apply_case_with_cached_ocr_writes_done(test_user, tmp_path):
    """Pre-populated OCR sentinel → ocr_status='done' (single-OCR
    guarantee — skill-time OCR cache makes the worker a no-op)."""
    import hashlib

    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF cached")
    sha = hashlib.sha256(b"%PDF cached").hexdigest()

    from claritymed.stores.blob_store import BlobStore

    blob = BlobStore("test")
    blob.store(b"%PDF cached", "pdf")
    blob.write_ocr_result(
        sha,
        status="ok",
        kind="ocr",
        ext="pdf",
        provider="rapidocr",
        chain_tried=["rapidocr"],
        reason=None,
        text="cached body text",
        original_filename="report.pdf",
    )
    assert blob.ocr_done(sha)

    case = _case(
        attachments=[
            CaseAttachment(
                path=str(pdf), original_filename="report.pdf", mime="application/pdf"
            )
        ]
    )
    result = apply_case("test", case)
    assert result.status == "done"

    from claritymed.stores.manifest_store import ManifestStore

    manifest = ManifestStore("test", scope="records").read("exam-reports", result.slug)
    assert manifest.attachments[0].ocr_status == "done"


def test_apply_case_with_missing_attachment_returns_error(test_user):
    case = _case(
        attachments=[
            CaseAttachment(
                path="/nonexistent/file.pdf",
                original_filename="missing.pdf",
                mime="application/pdf",
            )
        ]
    )
    result = apply_case("test", case)
    assert result.status == "error"
    assert result.error_detail is not None
    # Manifest must NOT have been written on the error path.
    from claritymed.errors import RecordNotFound
    from claritymed.stores.manifest_store import ManifestStore

    with pytest.raises(RecordNotFound):
        ManifestStore("test", scope="records").read("exam-reports", result.slug)


def test_apply_case_cas_dedupes_shared_attachment_bytes(test_user, tmp_path):
    """Two cases pointing at byte-identical files land on the same sha
    (BlobStore CAS) — no duplicate storage, both manifests reference it."""
    pdf_path = tmp_path / "shared.pdf"
    pdf_path.write_bytes(b"%PDF shared bytes")

    case_a = _case(
        case_id="case-a",
        attachments=[
            CaseAttachment(
                path=str(pdf_path),
                original_filename="shared.pdf",
                mime="application/pdf",
            )
        ],
    )
    case_b = _case(
        case_id="case-b",
        attachments=[
            CaseAttachment(
                path=str(pdf_path),
                original_filename="shared-renamed.pdf",
                mime="application/pdf",
            )
        ],
    )
    ra = apply_case("test", case_a)
    rb = apply_case("test", case_b)
    assert ra.status == "done"
    assert rb.status == "done"

    from claritymed.stores.manifest_store import ManifestStore

    ma = ManifestStore("test", scope="records").read("exam-reports", ra.slug)
    mb = ManifestStore("test", scope="records").read("exam-reports", rb.slug)
    assert ma.attachments[0].sha256 == mb.attachments[0].sha256


# --- error path on missing category ----------------------------------


def test_apply_case_with_no_category_raises_value_error(test_user):
    """The loader is responsible for backfilling category. Reaching
    apply_case with category=None is a programming error and should
    not be silently swallowed as ``error`` — it's distinct from a
    runtime error like a missing attachment."""
    case = CaseEntry(
        case_id="no-category",
        event_date=date(2024, 1, 15),
        title="No category",
        kind="exam-report",
        category=None,
    )
    with pytest.raises(ValueError, match="no category"):
        apply_case("test", case)
