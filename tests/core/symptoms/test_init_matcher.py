"""Unit coverage for the init-symptom matcher.

Avoids loading SapBERT — every test injects a stub encoder via
``InitMatcherEmbedder._model``. Catalog vectors are constructed
inline as orthogonal unit vectors so cosine scores are deterministic.
"""

from __future__ import annotations

import numpy as np

from claritymed.core.symptoms.datasets.canonical import (
    CanonicalEvidence,
    CanonicalValue,
    InitSymptomCatalog,
)
from claritymed.core.symptoms.init_matcher import (
    InitMatcherEmbedder,
    MatchResult,
    build_catalog,
    candidate_text,
    filter_candidate_evidences,
)
from claritymed.core.symptoms.schemas import InitSymptomFilter


def _ev(
    idx: int,
    *,
    dtype: str = "B",
    is_antecedent: bool = False,
    question: str = "",
) -> CanonicalEvidence:
    return CanonicalEvidence(
        id=f"E_{idx}",
        idx=idx,
        dtype=dtype,
        values=[CanonicalValue(raw="yes", local_idx=0)] if dtype != "B" else [],
        native_question_text={"en": question} if question else {},
        is_antecedent=is_antecedent,
    )


class _StubModel:
    """Drop-in replacement for SentenceTransformer.encode.

    ``vectors`` is a dict ``text → np.array``; missing texts return a
    zero vector. All output is L2-normalized to match the real
    SapBERT behaviour under ``normalize_embeddings=True``.
    """

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def encode(
        self,
        texts,
        *,
        normalize_embeddings=True,  # noqa: ARG002
        convert_to_numpy=True,  # noqa: ARG002
        show_progress_bar=False,  # noqa: ARG002
    ):
        out = []
        for t in texts:
            vec = self._vectors.get(t, np.zeros(4))
            norm = np.linalg.norm(vec)
            out.append(vec / norm if norm > 0 else vec)
        return np.asarray(out, dtype=np.float32)


def _make_embedder(
    model: _StubModel | None, *, threshold: float = 0.55
) -> InitMatcherEmbedder:
    em = InitMatcherEmbedder(model_id="stub", device="cpu", default_threshold=threshold)
    em._model = model
    # Bypass the load path so failures are about logic, not import.
    em._load_failed = model is None
    return em


# --- filter --------------------------------------------------------------


def test_filter_excludes_antecedent_by_default() -> None:
    evs = [
        _ev(0, is_antecedent=False),
        _ev(1, is_antecedent=True),
        _ev(2, is_antecedent=False),
    ]
    spec = InitSymptomFilter()
    kept = filter_candidate_evidences(evs, spec)
    assert [e.idx for e in kept] == [0, 2]


def test_filter_restricts_to_allowed_dtypes() -> None:
    evs = [
        _ev(0, dtype="B"),
        _ev(1, dtype="C"),
        _ev(2, dtype="M"),
        _ev(3, dtype="B"),
    ]
    spec = InitSymptomFilter(allowed_dtypes=["B"])
    kept = filter_candidate_evidences(evs, spec)
    assert [e.idx for e in kept] == [0, 3]


def test_filter_can_be_relaxed_to_include_M() -> None:
    evs = [_ev(0, dtype="B"), _ev(1, dtype="M")]
    spec = InitSymptomFilter(allowed_dtypes=["B", "M"])
    assert [e.idx for e in filter_candidate_evidences(evs, spec)] == [0, 1]


def test_candidate_text_prefers_en_question_then_id() -> None:
    with_q = _ev(0, question="Do you have a fever?")
    no_q = _ev(1)
    assert candidate_text(with_q) == "Do you have a fever?"
    assert candidate_text(no_q) == "E_1"


# --- catalog build -------------------------------------------------------


def test_build_catalog_returns_none_when_no_candidates() -> None:
    evs = [_ev(0, is_antecedent=True)]
    spec = InitSymptomFilter()
    em = _make_embedder(_StubModel({}))
    assert build_catalog(evs, spec, em, threshold=0.5) is None


def test_build_catalog_returns_none_when_encoder_fails() -> None:
    evs = [_ev(0, question="fever")]
    spec = InitSymptomFilter()
    em = _make_embedder(None)
    assert build_catalog(evs, spec, em, threshold=0.5) is None


def test_build_catalog_packs_idx_in_filter_order() -> None:
    evs = [
        _ev(0, question="fever"),
        _ev(1, dtype="C"),  # filtered out
        _ev(2, question="cough"),
    ]
    spec = InitSymptomFilter()
    em = _make_embedder(
        _StubModel(
            {
                "fever": np.array([1.0, 0.0, 0.0, 0.0]),
                "cough": np.array([0.0, 1.0, 0.0, 0.0]),
            }
        )
    )
    cat = build_catalog(evs, spec, em, threshold=0.5)
    assert cat is not None
    assert cat.candidate_idx == [0, 2]
    assert cat.matrix.shape == (2, 4)
    # Rows are L2-normalized.
    norms = np.linalg.norm(cat.matrix, axis=1)
    assert np.allclose(norms, 1.0)


# --- match ---------------------------------------------------------------


def _orthonormal_catalog() -> tuple[InitSymptomCatalog, InitMatcherEmbedder]:
    """Catalog of 2 orthogonal unit vectors at idx 7 and 9."""
    matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    catalog = InitSymptomCatalog(candidate_idx=[7, 9], matrix=matrix, threshold=0.55)
    em = _make_embedder(
        _StubModel(
            {
                "fever today": np.array([1.0, 0.0, 0.0, 0.0]),
                "mild fever": np.array([0.6, 0.4, 0.0, 0.0]),
                "fuzzy": np.array([0.4, 0.4, 0.4, 0.4]),
            }
        )
    )
    return catalog, em


def test_match_returns_top_match_above_threshold() -> None:
    cat, em = _orthonormal_catalog()
    r = em.match("fever today", cat)
    assert r.evidence_idx == 7
    assert r.score > 0.99


def test_match_skips_below_threshold() -> None:
    cat, em = _orthonormal_catalog()
    # "fuzzy" is normalized to ~0.5 on each axis → top score 0.5 < 0.55
    r = em.match("fuzzy", cat)
    assert r.evidence_idx is None
    assert r.score < 0.55


def test_match_picks_higher_of_two_candidates() -> None:
    cat, em = _orthonormal_catalog()
    # mild fever: normalized ≈ (0.832, 0.555); axis-0 wins
    r = em.match("mild fever", cat)
    assert r.evidence_idx == 7


def test_match_returns_none_on_empty_complaint() -> None:
    cat, em = _orthonormal_catalog()
    r = em.match("   ", cat)
    assert r == MatchResult(evidence_idx=None, score=0.0)


def test_match_returns_none_when_encoder_unavailable() -> None:
    cat, _ = _orthonormal_catalog()
    em = _make_embedder(None)
    r = em.match("fever", cat)
    assert r.evidence_idx is None
