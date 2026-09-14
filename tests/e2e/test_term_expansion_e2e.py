"""E2E: ``umls_cmekg_local`` exercised through the live retrieval stack.

What this proves end-to-end (no mocks anywhere):

* The auto-seeded ``shared/terminology/concepts.jsonl`` (see
  ``tests/e2e/conftest.py::_seed_terminology``) is read by
  ``build_hybrid_retriever()`` and produces a real
  ``UmlsCmekgLocalService`` — not the no-op fallback.
* ``HybridRetriever.retrieve`` calls ``expand_query`` against that service
  and surfaces the expanded surface form on ``EvidenceBundle.trace.
  expanded_query``. This is the only public seam where "the synonyms
  actually reached the embedder" is observable.
* The expanded query still produces a non-empty chunk set (i.e. the
  broader concept bag did not push the embedder out of distribution and
  retrieval is genuinely happening against the seeded corpora).

These tests **explicitly** patch ``term_service`` to
``umls_cmekg_local``; ``configs/retrieval.yaml`` currently ships
``active: none`` (see ``test_term_expansion_integration.py::
test_default_yaml_ships_term_service_off``), but the live path must keep
working so a single-line YAML flip is enough for operators.

Run:
    uv run pytest tests/e2e/test_term_expansion_e2e.py -v --no-cov

Required services (skipped otherwise — see ``conftest.py``):
    embedder (:8082), reranker (:8083), Qdrant (:6333).
"""

from __future__ import annotations

import asyncio

import pytest

from claritymed.core.rag.retriever_factory import build_hybrid_retriever
from claritymed.core.rag.schemas import (
    RetrievalConfig,
    TermServiceConfig,
    TermServiceEntry,
    load_retrieval_config,
)
from claritymed.core.rag.terms import UmlsCmekgLocalService


def _retriever_with_umls():
    """Build a HybridRetriever forced to use ``umls_cmekg_local``.

    Mirrors ``test_retriever_factory.test_build_hybrid_retriever_returns_real_retriever``
    in shape — model_copy + explicit ``TermServiceConfig`` — so the
    patching pattern stays consistent across the suite.
    """
    cfg: RetrievalConfig = load_retrieval_config()
    patched = cfg.model_copy(
        update={
            "term_service": TermServiceConfig(
                active="umls_cmekg_local",
                catalog=[
                    TermServiceEntry(id="umls_cmekg_local", kind="local"),
                    TermServiceEntry(id="none", kind="noop"),
                ],
            ),
        }
    )
    return build_hybrid_retriever(patched)


@pytest.mark.local
def test_retriever_uses_real_term_service():
    """Sanity: the wired-up retriever isn't silently NoOp."""
    retriever = _retriever_with_umls()
    # ``_term_service`` is private but the only way to assert composition
    # without re-running the whole factory.
    assert isinstance(retriever._term_service, UmlsCmekgLocalService)


@pytest.mark.local
@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_english_query_expansion_reaches_trace():
    """``aspirin`` must expand to ``acetylsalicylic acid`` / ``阿司匹林``."""
    retriever = _retriever_with_umls()
    bundle = asyncio.run(
        retriever.retrieve(
            "aspirin side effects",
            language="en",
            user_id="e2e_term_expansion",
        )
    )
    expanded = bundle.trace.expanded_query
    assert expanded is not None, (
        "trace.expanded_query is None — TermService produced no aliases for "
        "'aspirin'. Either the seed jsonl was not written or the factory "
        "silently fell back to NoOp."
    )
    assert "acetylsalicylic acid" in expanded
    assert "阿司匹林" in expanded
    # Original surface form must still lead the bag — the embedder weights
    # leading tokens more strongly, so synonym pollution at the front would
    # measurably hurt recall.
    assert expanded.startswith("aspirin side effects")


@pytest.mark.local
@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_chinese_query_expansion_pulls_in_english():
    """zh→en is the cross-lingual recall lift the term service exists for."""
    retriever = _retriever_with_umls()
    bundle = asyncio.run(
        retriever.retrieve(
            "血红蛋白 偏低",
            language="zh",
            user_id="e2e_term_expansion",
        )
    )
    expanded = bundle.trace.expanded_query
    assert expanded is not None
    assert "hemoglobin" in expanded.lower()


@pytest.mark.local
@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_expansion_does_not_break_retrieval():
    """The expanded bag must still pull chunks from statpearls_en.

    Catches the failure mode where synonym pollution drifts the embedding
    away from any relevant cluster — retrieval returns zero chunks even
    though the underlying query is on-topic.
    """
    retriever = _retriever_with_umls()
    bundle = asyncio.run(
        retriever.retrieve(
            "anemia symptoms",
            language="en",
            user_id="e2e_term_expansion",
        )
    )
    assert bundle.trace.expanded_query is not None
    assert len(bundle.chunks) >= 1, (
        f"expansion produced {bundle.trace.expanded_query!r} but retrieval "
        f"returned 0 chunks — synonym bag may be drifting the embedding "
        f"off-topic, or statpearls_en is unpopulated."
    )
