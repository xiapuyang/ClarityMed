"""Unit tests for ``scripts/normalize_mesh.py``.

These tests run the normalizer against a hand-built XML fixture that
exercises every type-mapping branch plus permuted-term filtering. The
fixtures stay small (~7 desc records + 2 supp records) so the
expected output can be enumerated inline — when the parser changes,
the diff makes intent obvious instead of hand-counting JSON lines.

CI-safe: no network, no live services, no real MeSH XML required.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

# scripts/ isn't a package; load it the same way init_terminology is
# loaded by tests/e2e/conftest.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
try:
    import normalize_mesh  # type: ignore[import-not-found]
finally:
    sys.path.pop(0)

_FIXTURES = Path(__file__).parent / "fixtures"
_DESC = _FIXTURES / "mesh_sample_desc.xml"
_SUPP = _FIXTURES / "mesh_sample_supp.xml"


def _run(
    *,
    desc: Path | None = None,
    supp: Path | None = None,
    type_filter: set[str] | None = None,
) -> list[dict]:
    """Run the normalizer in-process; parse stdout-equivalent buffer back to dicts.

    Going through the module-level ``_process_file`` (rather than
    ``subprocess`` + ``main``) keeps the test fast and lets pytest's
    capture machinery see assertion failures directly.
    """
    buf = io.StringIO()
    if desc is not None:
        normalize_mesh._process_file(
            desc,
            tag=normalize_mesh._DESC_TAG,
            type_fn=normalize_mesh._descriptor_type,
            out=buf,
            type_filter=type_filter,
            label="desc",
        )
    if supp is not None:
        normalize_mesh._process_file(
            supp,
            tag=normalize_mesh._SUPP_TAG,
            type_fn=normalize_mesh._supplemental_type,
            out=buf,
            type_filter=type_filter,
            label="supp",
        )
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


def _by_id(records: list[dict]) -> dict[str, dict]:
    return {r["concept_id"]: r for r in records}


# --- type mapping --------------------------------------------------------


def test_descriptor_type_mapping_covers_every_branch():
    """Every documented type branch must produce the documented output."""
    by_id = _by_id(_run(desc=_DESC))
    assert by_id["mesh:D000001"]["type"] == "drug"  # D02.* → drug
    assert by_id["mesh:D003920"]["type"] == "disease"  # C18.* → disease (not C23.888)
    assert by_id["mesh:D006470"]["type"] == "disease"  # C23.550.* first, not C23.888
    assert by_id["mesh:D006261"]["type"] == "symptom"  # C23.888.* first
    assert by_id["mesh:D003711"]["type"] == "procedure"  # E06.* → procedure
    assert by_id["mesh:D000715"]["type"] == "other"  # H01.* → other
    assert by_id["mesh:D099999"]["type"] == "other"  # no TreeNumberList → other


def test_supplemental_type_mapping():
    by_id = _by_id(_run(supp=_SUPP))
    assert by_id["mesh:C000002"]["type"] == "drug"  # SCRClass=1 → drug
    assert by_id["mesh:C400001"]["type"] == "other"  # SCRClass=4 → other


# --- alias extraction ----------------------------------------------------


def test_aliases_include_primary_name_first():
    """``DescriptorName`` must appear at index 0 even if Concepts repeat it.

    Downstream lookup doesn't care about order, but humans reading the
    JSONL do — the primary name leading the list is the convention.
    """
    by_id = _by_id(_run(desc=_DESC))
    aspirin = by_id["mesh:D000001"]
    assert aspirin["aliases"][0]["text"] == "Calcimycin"


def test_aliases_collapse_across_concepts():
    """All Concepts' Terms must land in one alias list (B-strategy storage).

    The first descriptor in the fixture has two Concepts; the test
    asserts a Term from each one survives the merge.
    """
    by_id = _by_id(_run(desc=_DESC))
    texts = [a["text"] for a in by_id["mesh:D000001"]["aliases"]]
    assert "Calcimycin" in texts
    assert "A-23187" in texts  # from the secondary Concept


def test_permuted_terms_are_skipped():
    """``IsPermutedTermYN="Y"`` Terms must not appear in aliases.

    Permuted forms ("A 23187", "A23187, Antibiotic") are noise for
    substring / token-level matching and would crowd the expansion
    output without adding semantic recall.
    """
    by_id = _by_id(_run(desc=_DESC))
    texts = {a["text"] for a in by_id["mesh:D000001"]["aliases"]}
    assert "A 23187" not in texts
    assert "A23187, Antibiotic" not in texts


def test_aliases_dedup_case_insensitive():
    """Repeated case variants of the same string collapse to one alias.

    MeSH frequently repeats the preferred name across DescriptorName
    + the first Concept's first Term; emitting it twice would inflate
    the alias list and waste expand_query's dedup-set capacity.
    """
    by_id = _by_id(_run(supp=_SUPP))
    texts = [a["text"] for a in by_id["mesh:C000002"]["aliases"]]
    lowered = [t.lower() for t in texts]
    assert len(lowered) == len(set(lowered))


def test_every_alias_is_tagged_mesh_and_en():
    """Source attribution is the operator's audit trail."""
    for rec in _run(desc=_DESC, supp=_SUPP):
        for alias in rec["aliases"]:
            assert alias["source"] == "mesh"
            assert alias["language"] == "en"


# --- concept_id prefix ---------------------------------------------------


def test_concept_ids_use_mesh_prefix():
    for rec in _run(desc=_DESC, supp=_SUPP):
        assert rec["concept_id"].startswith("mesh:"), rec


# --- type filtering ------------------------------------------------------


def test_type_filter_drops_non_matching_records():
    """``--types drug,symptom`` must drop disease/procedure/other records."""
    records = _run(desc=_DESC, type_filter={"drug", "symptom"})
    by_id = _by_id(records)
    assert "mesh:D000001" in by_id  # drug — kept
    assert "mesh:D006261" in by_id  # symptom — kept
    assert "mesh:D003920" not in by_id  # disease — dropped
    assert "mesh:D003711" not in by_id  # procedure — dropped
    assert "mesh:D000715" not in by_id  # other — dropped


def test_unknown_type_in_filter_exits_loud(monkeypatch):
    """Operator typos in --types must fail loud, not silently drop everything."""
    with pytest.raises(SystemExit):
        normalize_mesh._parse_types("drug,disees")  # typo


# --- output schema matches init_terminology validator -------------------


def test_emitted_records_validate_against_init_terminology_schema():
    """The merge target must be able to read our output unmodified."""
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    try:
        from init_terminology import _validate_record  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    for rec in _run(desc=_DESC, supp=_SUPP):
        _validate_record(rec)  # raises if shape is wrong
