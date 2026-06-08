"""Unit 4: UMLS+CMeKG term service + query expansion."""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.core.rag.terms import (
    Alias,
    ConceptHit,
    NoOpTermService,
    TermService,
    UmlsCmekgLocalService,
    expand_query,
)
from claritymed.core.rag.terms.factory import build_term_service as _build
from claritymed.core.rag.schemas import TermServiceConfig
from claritymed.errors import UnknownTermServiceError

FIXTURE = Path(__file__).parent / "fixtures" / "concepts_sample.jsonl"


@pytest.fixture(scope="module")
def svc() -> UmlsCmekgLocalService:
    return UmlsCmekgLocalService(FIXTURE)


# --- UmlsCmekgLocalService ----------------------------------------------


def test_service_implements_protocol(svc):
    assert isinstance(svc, TermService)


def test_lookup_en_finds_concept(svc):
    hits = svc.lookup("aspirin", "en")
    assert len(hits) == 1
    assert hits[0].concept_id == "C0004057"
    assert hits[0].type == "drug"
    surfaces = {a.text for a in hits[0].aliases}
    assert "aspirin" in surfaces
    assert "acetylsalicylic acid" in surfaces
    assert "ASA" in surfaces
    assert "阿司匹林" in surfaces


def test_lookup_zh_finds_same_concept_as_en(svc):
    hits_en = svc.lookup("aspirin", "en")
    hits_zh = svc.lookup("阿司匹林", "zh")
    assert hits_en[0].concept_id == hits_zh[0].concept_id


def test_lookup_is_case_insensitive_for_english(svc):
    assert svc.lookup("Aspirin", "en")[0].concept_id == "C0004057"
    assert svc.lookup("ASPIRIN", "en")[0].concept_id == "C0004057"


def test_lookup_unknown_surface_returns_empty(svc):
    assert svc.lookup("totally not a drug name", "en") == []


def test_lookup_empty_or_whitespace_returns_empty(svc):
    assert svc.lookup("", "en") == []
    assert svc.lookup("   ", "en") == []


def test_cross_lingual_aliases_returns_both_languages(svc):
    aliases = svc.cross_lingual_aliases("C0004057")
    langs = {a.language for a in aliases}
    assert {"en", "zh"} <= langs


def test_cross_lingual_aliases_unknown_concept_returns_empty(svc):
    assert svc.cross_lingual_aliases("C9999999") == []


def test_missing_file_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError):
        UmlsCmekgLocalService(tmp_path / "missing.jsonl")


def test_bad_jsonl_line_is_skipped(tmp_path, caplog):
    f = tmp_path / "concepts.jsonl"
    f.write_text(
        '{"concept_id":"C1","type":"drug","aliases":[{"text":"x","language":"en","source":"umls"}]}\n'
        "not valid json\n"
        '{"concept_id":"C2","type":"drug","aliases":[{"text":"y","language":"en","source":"umls"}]}\n',
        encoding="utf-8",
    )
    svc = UmlsCmekgLocalService(f)
    assert svc.lookup("x", "en")
    assert svc.lookup("y", "en")


# --- NoOpTermService ----------------------------------------------------


def test_noop_returns_empty():
    no = NoOpTermService()
    assert isinstance(no, TermService)
    assert no.lookup("aspirin", "en") == []
    assert no.cross_lingual_aliases("any") == []


# --- expand_query -------------------------------------------------------


def test_expand_query_adds_english_synonyms(svc):
    out = expand_query("aspirin side effects", "en", svc)
    assert "acetylsalicylic acid" in out
    assert "ASA" in out
    # original query preserved at the front
    assert out.startswith("aspirin side effects")


def test_expand_query_adds_cross_lingual_for_zh(svc):
    out = expand_query("阿司匹林 副作用", "zh", svc)
    assert "aspirin" in out
    assert "acetylsalicylic acid" in out


def test_expand_query_multi_word_term(svc):
    # "acetylsalicylic acid" is a 2-word term; n-gram lookup picks it up.
    out = expand_query("acetylsalicylic acid is what?", "en", svc)
    assert "aspirin" in out
    assert "ASA" in out


def test_expand_query_no_matches_returns_unchanged(svc):
    q = "this query has nothing useful"
    assert expand_query(q, "en", svc) == q


def test_expand_query_empty_input(svc):
    assert expand_query("", "en", svc) == ""
    assert expand_query("   ", "en", svc) == "   "


def test_expand_query_with_noop_is_identity():
    q = "aspirin side effects"
    assert expand_query(q, "en", NoOpTermService()) == q


def test_expand_query_does_not_duplicate_existing_tokens(svc):
    out = expand_query("aspirin acetylsalicylic acid", "en", svc)
    # 'aspirin' and 'acetylsalicylic acid' both already in query; should
    # not be re-appended. ASA should still appear.
    assert out.count("aspirin") == 1
    assert out.count("acetylsalicylic acid") == 1
    assert "ASA" in out


# --- factory ------------------------------------------------------------


def test_build_term_service_noop():
    cfg = TermServiceConfig(
        active="none",
        catalog=[
            {"id": "umls_cmekg_local", "kind": "local", "data_dir": "x"},  # type: ignore[list-item]
            {"id": "none", "kind": "noop"},  # type: ignore[list-item]
        ],
    )
    assert isinstance(_build(cfg), NoOpTermService)


def test_build_term_service_local_requires_data_file(tmp_path):
    cfg = TermServiceConfig(
        active="umls_cmekg_local",
        catalog=[
            {  # type: ignore[list-item]
                "id": "umls_cmekg_local",
                "kind": "local",
                "data_dir": str(tmp_path),
            }
        ],
    )
    with pytest.raises(FileNotFoundError):
        _build(cfg)


def test_build_term_service_local_happy_path(tmp_path):
    # Copy the fixture into tmp_path and load via factory.
    (tmp_path / "concepts.jsonl").write_bytes(FIXTURE.read_bytes())
    cfg = TermServiceConfig(
        active="umls_cmekg_local",
        catalog=[
            {  # type: ignore[list-item]
                "id": "umls_cmekg_local",
                "kind": "local",
                "data_dir": str(tmp_path),
            }
        ],
    )
    svc = _build(cfg)
    assert isinstance(svc, UmlsCmekgLocalService)


def test_build_term_service_unknown_id_raises():
    from claritymed.core.rag.schemas import TermServiceConfig, TermServiceEntry

    entry = TermServiceEntry.model_construct(
        id="umls_remote_api", kind="local", data_dir=None
    )
    cfg = TermServiceConfig.model_construct(active="umls_remote_api", catalog=[entry])
    with pytest.raises(UnknownTermServiceError):
        _build(cfg)


# --- value type roundtrip -----------------------------------------------


def test_concept_hit_and_alias_are_frozen():
    alias = Alias(text="x", language="en", source="umls")
    with pytest.raises(Exception):
        alias.text = "y"  # type: ignore[misc]

    hit = ConceptHit(
        concept_id="C1",
        surface="x",
        language="en",
        score=1.0,
        type="drug",
        aliases=(alias,),
    )
    with pytest.raises(Exception):
        hit.score = 0.5  # type: ignore[misc]
