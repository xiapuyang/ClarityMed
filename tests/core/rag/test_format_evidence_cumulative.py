"""Regression tests for cumulative-numbered Evidence formatting.

Locks in the contract introduced for ce:review P1 #6 — when the LLM
calls ``retrieve_medical_literature`` more than once in a turn, the
Evidence block for each call must number against the cumulative
dedup of ``deps.retrieved_chunks``, NOT against just this call's
chunks. Otherwise ``[2]`` from call 1 and ``[2]`` from call 2 refer to
different documents while the final Sources block shows only the
first one — the LLM's citation index would be unmoored from what the
user sees.
"""

from __future__ import annotations

from claritymed.core.rag.retrieval_pipeline import format_evidence
from claritymed.core.schemas.retrieval import RetrievedChunk


def _chunk(doc_id: str, text: str, *, source_uri: str | None = None) -> RetrievedChunk:
    """Minimal chunk factory satisfying RetrievedChunk's required fields."""
    return RetrievedChunk(
        text=text,
        source="system_rag",
        score=1.0,
        doc_id=doc_id,
        source_uri=source_uri,
    )


def test_format_evidence_per_call_numbering_when_no_cumulative_arg():
    """Without ``cumulative=``, numbering stays per-call (back-compat)."""
    chunks = [_chunk("a", "alpha"), _chunk("b", "beta")]
    out = format_evidence(chunks)
    assert "[1]" in out
    assert "[2]" in out
    assert "[3]" not in out


def test_format_evidence_cumulative_offsets_second_call_indices():
    """Second tool call's chunks must be numbered as [3] [4] when the
    first call already filled positions [1] [2]."""
    call1 = [_chunk("a", "alpha"), _chunk("b", "beta")]
    call2 = [_chunk("c", "gamma"), _chunk("d", "delta")]
    cumulative = [*call1, *call2]

    out_call2 = format_evidence(call2, cumulative=cumulative)
    # call2's chunks land at positions 3 and 4 in the cumulative dedup.
    assert "[3] " in out_call2
    assert "[4] " in out_call2
    # And NOT at 1 / 2 — which is the bug the cumulative arg fixes.
    assert "[1] " not in out_call2
    assert "[2] " not in out_call2


def test_format_evidence_cumulative_indices_match_user_sources():
    """The cumulative numbering used in Evidence must match what the
    user finally sees in Sources (which dedupes by doc_id too)."""
    call1 = [_chunk("a", "alpha", source_uri="urn:a")]
    call2 = [_chunk("b", "beta", source_uri="urn:b")]
    cumulative = [*call1, *call2]

    # The Sources block (AskService._format_sources) dedupes by doc_id
    # and assigns [1]=a, [2]=b. Evidence for call 2 must therefore show
    # [2] for doc 'b' — not [1].
    out_call2 = format_evidence(call2, cumulative=cumulative)
    assert "[2] (urn:b)" in out_call2


def test_format_evidence_cumulative_handles_duplicate_doc_across_calls():
    """A chunk for the same doc_id appearing in both calls keeps the
    same index — the cumulative dedup is what supplies the numbering."""
    chunk_a1 = _chunk("a", "alpha first")
    chunk_a2 = _chunk("a", "alpha second")  # same doc_id
    chunk_b = _chunk("b", "beta")
    cumulative = [chunk_a1, chunk_a2, chunk_b]

    out_call1 = format_evidence([chunk_a1], cumulative=cumulative)
    out_call2 = format_evidence([chunk_a2, chunk_b], cumulative=cumulative)

    # Both calls show doc 'a' as [1].
    assert "[1] " in out_call1
    assert "[1] " in out_call2
    # And doc 'b' as [2] in call 2.
    assert "[2] " in out_call2


def test_format_evidence_empty_input_returns_empty_string():
    """No chunks → no Evidence block, regardless of cumulative."""
    assert format_evidence([]) == ""
    assert format_evidence([], cumulative=[_chunk("a", "alpha")]) == ""
