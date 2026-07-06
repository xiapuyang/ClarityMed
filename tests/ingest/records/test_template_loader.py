"""Tests for ``ingest.records.template_loader.load_template``.

Each test points at a fixture directory under ``fixtures/templates/``
to keep the contract under git review (the fixtures double as docs for
the skill author about what a valid template looks like).

Two responsibilities to pin:

1. Every invariant from the plan's §"Template directory shape" — flat
   directory, declared user_ids match files, case_id is globally unique,
   ``kind`` required, attachments exist + are not symlinks.
2. ``import_id`` content-canonicalization — two templates identical
   except for ``_meta.created_at`` or YAML key ordering produce the same
   id; ``.DS_Store`` contamination cannot move the id (it's rejected
   before the hash walks).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from claritymed.errors import TemplateValidationError
from claritymed.ingest.records.template_loader import (
    CLEANUP_FAILED_MARKER,
    IMPORT_ID_LENGTH,
    LoadedTemplate,
    load_template,
)

FIXTURES = Path(__file__).parent / "fixtures" / "templates"


# --- happy paths -------------------------------------------------------


def test_minimal_single_user_loads_cleanly():
    loaded = load_template(FIXTURES / "minimal_single_user")
    assert isinstance(loaded, LoadedTemplate)
    assert set(loaded.bundles.keys()) == {"test"}
    bundle = loaded.bundles["test"]
    assert len(bundle.cases) == 1
    assert bundle.cases[0].case_id == "notion-abcdef12"
    # category backfilled from _meta.default_category
    assert bundle.cases[0].category == "exam-reports"
    # facts round-tripped
    assert bundle.facts.profile == {"sex": "female"}
    assert len(bundle.facts.allergies) == 1


def test_minimal_single_user_import_id_format():
    loaded = load_template(FIXTURES / "minimal_single_user")
    assert isinstance(loaded.import_id, str)
    assert len(loaded.import_id) == IMPORT_ID_LENGTH
    assert all(c in "0123456789abcdef" for c in loaded.import_id)


def test_multi_user_50_cases_loads_with_cross_file_uniqueness():
    loaded = load_template(FIXTURES / "multi_user_50_cases")
    assert set(loaded.bundles.keys()) == {"alpha", "bravo", "charlie"}
    total = sum(len(b.cases) for b in loaded.bundles.values())
    assert total == 50
    all_case_ids = {case.case_id for b in loaded.bundles.values() for case in b.cases}
    assert len(all_case_ids) == 50


# --- error paths -------------------------------------------------------


def test_unexpected_file_in_template_rejected():
    """Plan says ``stray.yaml`` is rejected. The loader can surface
    either the file name or the user_id stem (``stray``) — the user
    can find the file from the stem and only ``test`` is declared."""
    with pytest.raises(TemplateValidationError, match="stray"):
        load_template(FIXTURES / "invalid_extra_file")


def test_declared_user_without_file_rejected():
    with pytest.raises(TemplateValidationError, match="other"):
        load_template(FIXTURES / "invalid_missing_user")


def test_cross_file_case_id_collision_rejected():
    with pytest.raises(TemplateValidationError) as exc_info:
        load_template(FIXTURES / "invalid_case_id_collision")
    msg = str(exc_info.value)
    assert "shared-id" in msg
    assert "test.yaml" in msg
    assert "other.yaml" in msg


def test_missing_template_directory_rejected(tmp_path):
    with pytest.raises(TemplateValidationError, match="does not exist"):
        load_template(tmp_path / "no-such-dir")


def test_missing_meta_yaml_rejected(tmp_path):
    (tmp_path / "test.yaml").write_text("cases: []\n")
    with pytest.raises(TemplateValidationError, match="_meta.yaml"):
        load_template(tmp_path)


def test_subdir_in_template_rejected(tmp_path):
    _write_meta(tmp_path, ["test"])
    (tmp_path / "test.yaml").write_text("cases: []\n")
    (tmp_path / "extra_subdir").mkdir()
    with pytest.raises(TemplateValidationError, match="subdirs not allowed"):
        load_template(tmp_path)


def test_ds_store_in_template_rejected(tmp_path):
    """`.DS_Store` is the canonical Spotlight contamination scenario.
    Plan rejects it explicitly so the hash isn't moved by macOS metadata."""
    _write_meta(tmp_path, ["test"])
    (tmp_path / "test.yaml").write_text("cases: []\n")
    (tmp_path / ".DS_Store").write_bytes(b"\x00\x01\x02")
    with pytest.raises(TemplateValidationError, match=".DS_Store"):
        load_template(tmp_path)


def test_symlink_in_template_rejected(tmp_path):
    """Symlinks would let a template silently reach outside its tree."""
    _write_meta(tmp_path, ["test"])
    (tmp_path / "test.yaml").write_text("cases: []\n")
    real_target = tmp_path.parent / "real_target.yaml"
    real_target.write_text("cases: []\n")
    (tmp_path / "extra.yaml").symlink_to(real_target)
    with pytest.raises(TemplateValidationError, match="symlink"):
        load_template(tmp_path)


def test_non_yaml_file_rejected(tmp_path):
    _write_meta(tmp_path, ["test"])
    (tmp_path / "test.yaml").write_text("cases: []\n")
    (tmp_path / "README.md").write_text("notes\n")
    with pytest.raises(TemplateValidationError, match="README.md"):
        load_template(tmp_path)


def test_invalid_user_id_filename_rejected(tmp_path):
    """Filename `<user_id>.yaml` must match USER_ID_RE."""
    _write_meta(tmp_path, ["test"])
    (tmp_path / "test.yaml").write_text("cases: []\n")
    (tmp_path / "has space.yaml").write_text("cases: []\n")
    with pytest.raises(TemplateValidationError, match="has space.yaml"):
        load_template(tmp_path)


def test_extra_key_in_user_bundle_rejected(tmp_path):
    """If a maintainer slips ``user_id:`` into the body, the per-user
    bundle's ``extra='forbid'`` catches it BEFORE the orchestrator
    runs — locking the filename-authoritative discipline."""
    _write_meta(tmp_path, ["test"])
    (tmp_path / "test.yaml").write_text("user_id: alice\ncases: []\nfacts: {}\n")
    with pytest.raises(TemplateValidationError, match="test.yaml"):
        load_template(tmp_path)


def test_case_without_kind_rejected(tmp_path):
    _write_meta(tmp_path, ["test"], default_category="exam-reports")
    (tmp_path / "test.yaml").write_text(
        "cases:\n"
        "  - case_id: no-kind\n"
        '    event_date: "2024-01-15"\n'
        "    title: Missing kind\n"
    )
    with pytest.raises(TemplateValidationError, match="kind"):
        load_template(tmp_path)


def test_case_without_category_and_no_default_rejected(tmp_path):
    _write_meta(tmp_path, ["test"])  # no default_category
    (tmp_path / "test.yaml").write_text(
        "cases:\n"
        "  - case_id: no-category\n"
        '    event_date: "2024-01-15"\n'
        "    title: Missing category\n"
        "    kind: exam-report\n"
    )
    with pytest.raises(TemplateValidationError, match="no category"):
        load_template(tmp_path)


def test_attachment_with_missing_path_rejected(tmp_path):
    _write_meta(tmp_path, ["test"], default_category="exam-reports")
    (tmp_path / "test.yaml").write_text(
        "cases:\n"
        "  - case_id: missing-att\n"
        '    event_date: "2024-01-15"\n'
        "    title: Missing attachment file\n"
        "    kind: exam-report\n"
        "    attachments:\n"
        "      - path: /nonexistent/path/to/file.pdf\n"
        "        original_filename: file.pdf\n"
        "        mime: application/pdf\n"
    )
    with pytest.raises(TemplateValidationError, match="does not exist"):
        load_template(tmp_path)


def test_symlinked_attachment_rejected(tmp_path):
    _write_meta(tmp_path, ["test"], default_category="exam-reports")
    real = tmp_path.parent / "real_attachment.pdf"
    real.write_bytes(b"%PDF-1.4 fake\n")
    sym = tmp_path / "link_attachment.pdf"
    sym.symlink_to(real)
    (tmp_path / "test.yaml").write_text(
        "cases:\n"
        "  - case_id: sym-att\n"
        '    event_date: "2024-01-15"\n'
        "    title: Sym attachment\n"
        "    kind: exam-report\n"
        "    attachments:\n"
        f"      - path: {sym}\n"
        "        original_filename: link_attachment.pdf\n"
        "        mime: application/pdf\n"
    )
    with pytest.raises(TemplateValidationError, match="symlink"):
        load_template(tmp_path)


def test_cleanup_failed_marker_blocks_load(tmp_path):
    _write_meta(tmp_path, ["test"], default_category="exam-reports")
    (tmp_path / "test.yaml").write_text("cases: []\n")
    (tmp_path / CLEANUP_FAILED_MARKER).write_text("rmtree failed: EBUSY\n")
    with pytest.raises(TemplateValidationError, match="cleanup_failed"):
        load_template(tmp_path)


# --- import_id canonicalization ----------------------------------------


def test_import_id_invariant_to_parent_path(tmp_path):
    """Same template content in two different parent dirs → same id."""
    a = tmp_path / "parent_a" / "template"
    b = tmp_path / "parent_b" / "template"
    shutil.copytree(FIXTURES / "minimal_single_user", a)
    shutil.copytree(FIXTURES / "minimal_single_user", b)
    assert load_template(a).import_id == load_template(b).import_id


def test_import_id_invariant_to_meta_created_at(tmp_path):
    """`_meta.created_at` is stripped before hashing — two skill runs
    over identical content (different timestamps) produce one id."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    shutil.copytree(FIXTURES / "minimal_single_user", a)
    shutil.copytree(FIXTURES / "minimal_single_user", b)
    (a / "_meta.yaml").write_text(
        "schema_version: 1\n"
        'created_at: "2024-01-01T00:00:00Z"\n'
        "default_category: exam-reports\n"
        "user_ids:\n  - test\n"
    )
    (b / "_meta.yaml").write_text(
        "schema_version: 1\n"
        'created_at: "2030-12-31T23:59:59Z"\n'
        "default_category: exam-reports\n"
        "user_ids:\n  - test\n"
    )
    assert load_template(a).import_id == load_template(b).import_id


def test_import_id_invariant_to_yaml_key_ordering(tmp_path):
    """Two per-user YAMLs with the same logical content but different
    key ordering produce the same id — the canonicalization re-dumps
    with ``sort_keys=True``."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_meta(a, ["test"], default_category="exam-reports")
    _write_meta(b, ["test"], default_category="exam-reports")
    (a / "test.yaml").write_text(
        "cases:\n"
        "  - title: First\n"
        '    event_date: "2024-01-15"\n'
        "    case_id: case-1\n"
        "    kind: exam-report\n"
    )
    (b / "test.yaml").write_text(
        "cases:\n"
        "  - case_id: case-1\n"
        '    event_date: "2024-01-15"\n'
        "    kind: exam-report\n"
        "    title: First\n"
    )
    assert load_template(a).import_id == load_template(b).import_id


def test_import_id_changes_on_real_content_change(tmp_path):
    """Sanity: a real edit to a case's title moves the id. If this
    passes for the wrong reason (e.g. canonicalization too aggressive),
    test_import_id_invariant_to_yaml_key_ordering would also pass — so
    they pin the desired sensitivity together."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_meta(a, ["test"], default_category="exam-reports")
    _write_meta(b, ["test"], default_category="exam-reports")
    (a / "test.yaml").write_text(
        "cases:\n"
        "  - case_id: case-1\n"
        '    event_date: "2024-01-15"\n'
        "    title: First\n"
        "    kind: exam-report\n"
    )
    (b / "test.yaml").write_text(
        "cases:\n"
        "  - case_id: case-1\n"
        '    event_date: "2024-01-15"\n'
        "    title: Different\n"
        "    kind: exam-report\n"
    )
    assert load_template(a).import_id != load_template(b).import_id


# --- warnings ---------------------------------------------------------


def test_warning_when_kind_equals_category(tmp_path):
    """`kind == category` likely means a skill author confused the two
    (kind singular vs category plural). Surfaced as a warning, not a
    hard reject — locks the plan's "loader's earlier warning has surfaced"
    invariant."""
    _write_meta(tmp_path, ["test"], default_category="lab-reports")
    (tmp_path / "test.yaml").write_text(
        "cases:\n"
        "  - case_id: warned\n"
        '    event_date: "2024-01-15"\n'
        "    title: Warn me\n"
        "    kind: lab-reports\n"
    )
    loaded = load_template(tmp_path)
    assert any("kind == category" in w for w in loaded.warnings)


# --- helpers ---------------------------------------------------------


def _write_meta(
    template_dir: Path,
    user_ids: list[str],
    default_category: str | None = None,
) -> None:
    parts = [
        "schema_version: 1",
        'created_at: "2026-06-25T00:00:00Z"',
    ]
    if default_category is not None:
        parts.append(f"default_category: {default_category}")
    parts.append("user_ids:")
    for uid in user_ids:
        parts.append(f"  - {uid}")
    (template_dir / "_meta.yaml").write_text("\n".join(parts) + "\n")
