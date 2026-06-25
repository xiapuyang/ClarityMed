"""Tests for the retrieval.yaml auto-append helper."""

from __future__ import annotations

import pytest

from claritymed.web.admin.retrieval_yaml import append_system_rag_collection


_SNIPPET_NEW = """    - name: brand_new_en
      language: en
      cross_lingual: false
      authority_tier: 2
      topics: []
      disease_codes: []
      source_uri_prefix: null
      license: null"""


_SAMPLE_RETRIEVAL = """\
# top-level comment
qdrant:
  url: http://localhost:6333

system_rag:
  default_active: []
  score_threshold: 0.4
  collections:
    # FROZEN: collection names below are Qdrant index names on disk.
    - name: statpearls_en
      language: en
      cross_lingual: true
      authority_tier: 1
      topics:
        - clinical
      disease_codes: []
      source_uri_prefix: "https://ncbi"
      license: "CC"
    - name: textbooks_en
      language: en
      cross_lingual: true
      authority_tier: 2
      topics:
        - pediatrics
      disease_codes: []
      source_uri_prefix: null
      license: null

routing:
  classifier: centroid
"""


def test_append_inserts_new_entry(tmp_path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text(_SAMPLE_RETRIEVAL, encoding="utf-8")

    appended = append_system_rag_collection(
        _SNIPPET_NEW, name="brand_new_en", path=path
    )
    assert appended is True

    after = path.read_text(encoding="utf-8")
    assert "- name: brand_new_en" in after
    # New entry sits inside the system_rag.collections block, BEFORE
    # the routing: sibling key.
    sysrag_start = after.index("system_rag:")
    routing_start = after.index("routing:")
    new_pos = after.index("- name: brand_new_en")
    assert sysrag_start < new_pos < routing_start


def test_append_preserves_comments(tmp_path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text(_SAMPLE_RETRIEVAL, encoding="utf-8")

    append_system_rag_collection(_SNIPPET_NEW, name="brand_new_en", path=path)
    after = path.read_text(encoding="utf-8")
    assert "# top-level comment" in after
    assert "# FROZEN: collection names below" in after


def test_append_is_idempotent_when_name_exists(tmp_path) -> None:
    """Idempotency: appending an existing name must NOT add a duplicate."""
    path = tmp_path / "retrieval.yaml"
    path.write_text(_SAMPLE_RETRIEVAL, encoding="utf-8")
    original = path.read_text(encoding="utf-8")

    # Snippet for an existing name — must not be inserted.
    snippet = _SNIPPET_NEW.replace("brand_new_en", "statpearls_en")
    appended = append_system_rag_collection(snippet, name="statpearls_en", path=path)
    assert appended is False
    assert path.read_text(encoding="utf-8") == original


def test_append_appends_after_existing_entries(tmp_path) -> None:
    """Snippet lands at the END of the collections block, not the start."""
    path = tmp_path / "retrieval.yaml"
    path.write_text(_SAMPLE_RETRIEVAL, encoding="utf-8")

    append_system_rag_collection(_SNIPPET_NEW, name="brand_new_en", path=path)
    after = path.read_text(encoding="utf-8")

    # Order: statpearls → textbooks → brand_new_en.
    sp = after.index("- name: statpearls_en")
    tb = after.index("- name: textbooks_en")
    nw = after.index("- name: brand_new_en")
    assert sp < tb < nw


def test_append_raises_when_block_missing(tmp_path) -> None:
    path = tmp_path / "retrieval.yaml"
    path.write_text("qdrant:\n  url: x\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        append_system_rag_collection(_SNIPPET_NEW, name="x", path=path)


def test_append_handles_empty_collections_block(tmp_path) -> None:
    """A literally empty ``collections:`` (no entries) still appends cleanly."""
    path = tmp_path / "retrieval.yaml"
    path.write_text(
        "system_rag:\n  collections:\n\nrouting:\n  x: y\n", encoding="utf-8"
    )
    appended = append_system_rag_collection(
        _SNIPPET_NEW, name="brand_new_en", path=path
    )
    assert appended is True
    after = path.read_text(encoding="utf-8")
    assert "- name: brand_new_en" in after
    # And the routing: sibling key is still on its own column-0 line.
    assert "\nrouting:\n" in after
