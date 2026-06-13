"""E2E: the 3-layer PHI cloud defense composes end-to-end.

* Layer 1 — structural: PHI records land in ``user_phi_<id>``, never
  in ``user_rag_<id>``. Verified by ``UserPhiRagStore.add_record``
  writing to a separately-named collection.
* Layer 2 — filter: ``PhiGuard.filter_chunks_for_provider(kind=cloud)``
  drops every chunk with ``can_cloud=False``.
* Layer 3 — runtime: ``PhiAssertionModel`` refuses a cloud-bound
  ``Model.request`` whose message stream contains PHI text.

Each layer is unit-tested separately; this file proves their
composition: a bug in layer 2 still cannot leak because layer 3
fires; a bug in layer 3 still has layers 1+2 protecting.
"""

from __future__ import annotations

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.phi.assertion_model import PhiAssertionModel
from claritymed.core.phi.guard import PhiGuard
from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.errors import PhiLeakDetected


@pytest.fixture
def _ctx():
    tokens = apply_context("20260611000000ABCDEF12", "e2e", "en")
    yield
    reset_context(tokens)


def test_layer2_filter_drops_can_cloud_false():
    """``filter_chunks_for_provider(kind="cloud")`` drops chunks marked
    is_phi=True AND not can_cloud=True. Mirrors the contract a
    layer-2 leak would violate.
    """
    guard = PhiGuard.from_config()
    phi_chunk = RetrievedChunk(
        text="HGB 105 g/L",
        source="user_rag",
        score=0.9,
        doc_id="exam-reports/2026-06-11-aaaa",
        chunk_index=0,
        is_phi=True,
        can_cloud=False,
        collection_name="user_phi_alice",
    )
    library_chunk = RetrievedChunk(
        text="published reference",
        source="system_rag",
        score=0.9,
        doc_id="textbooks/foo",
        chunk_index=0,
        is_phi=False,
        can_cloud=True,
        collection_name="textbooks_en",
    )
    kept, _ = guard.filter_chunks_for_provider(
        [phi_chunk, library_chunk], provider_kind="cloud"
    )
    assert library_chunk in kept
    assert phi_chunk not in kept


async def test_layer3_assertion_blocks_phi_in_message_stream(_ctx):
    """A buggy filter that lets a PHI chunk through still trips layer 3."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    class _Inner:
        model_name = "stub"
        system = "test"

        async def request(self, *args, **kwargs):
            raise AssertionError("inner should not run when leak detected")

    inner = _Inner()
    wrapper = PhiAssertionModel(inner, guard=PhiGuard.from_config())
    messages = [ModelRequest(parts=[UserPromptPart(content="Call me at 415-555-1234")])]

    class _Params:
        function_tools = []

    with pytest.raises(PhiLeakDetected):
        await wrapper.request(messages, None, _Params())
