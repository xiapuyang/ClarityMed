"""Tests for ``ClaritymedRagLM`` — AskService → lm-eval LM bridge.

The strategy is: never build a real ``AskService``. The adapter only
talks to its inner service through ``service.run`` (an async generator
of events), so we substitute a stub service that yields a scripted
event sequence per call. Anything below the service layer — the model,
the retriever, the strategy, the PHI guard — is therefore irrelevant
to these unit tests.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import pytest
from lm_eval.api.instance import Instance

from claritymed.core.events import Done, Error, Event, TokenChunk, ToolStarted
from claritymed.core.schemas import ProviderConfig
from claritymed.evals.lm.rag import ClaritymedRagLM

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_PROVIDER_DICT = {
    "id": "ollama",
    "kind": "local",
    "model": "qwen3:14b",
    "base_url": "http://127.0.0.1:11434/v1",
}


def _provider() -> ProviderConfig:
    return ProviderConfig.model_validate(_PROVIDER_DICT)


def _instance(context: str, idx: int = 0, gen_kwargs: dict | None = None) -> Instance:
    return Instance(
        request_type="generate_until",
        doc={},
        arguments=(context, gen_kwargs if gen_kwargs is not None else {}),
        idx=idx,
    )


class _StubService:
    """Replacement for ``AskService`` — records calls + scripts events.

    ``script`` is a list-of-lists: outer index = call number, inner list
    = events that call yields. When the adapter calls ``service.run``
    for the Nth time, it gets ``script[N]`` back.
    """

    def __init__(self, script: list[list[Event]]) -> None:
        self._script = list(script)
        self.calls: list[tuple[str, str]] = []

    async def run(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
        self.calls.append((user_input, user_id))
        if not self._script:
            return
        events = self._script.pop(0)
        for ev in events:
            yield ev


def _make_lm(
    monkeypatch, script: list[list[Event]]
) -> tuple[ClaritymedRagLM, _StubService]:
    """Build a ``ClaritymedRagLM`` whose internal service is the stub."""
    # The adapter's constructor calls ``_build_service`` which in turn
    # calls ``build_model``, ``make_translation_provider``, and
    # ``AskService(...)``. Patching ``_build_service`` directly is the
    # cleanest seam — the adapter never reaches past it in normal use.
    stub = _StubService(script)
    monkeypatch.setattr(
        ClaritymedRagLM,
        "_build_service",
        lambda self: stub,
    )
    # ``_default_strategy`` reads configs/retrieval.yaml + builds a real
    # hybrid retriever; stub to None for hermeticity.
    monkeypatch.setattr(
        "claritymed.evals.lm.rag._default_strategy",
        lambda _p: None,
    )
    lm = ClaritymedRagLM(_provider())
    return lm, stub


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_concatenates_token_chunks_in_arrival_order(monkeypatch):
    script = [
        [
            TokenChunk(text="The answer is "),
            TokenChunk(text="B."),
            Done(final="The answer is B."),
        ]
    ]
    lm, stub = _make_lm(monkeypatch, script)

    out = lm.generate_until([_instance("Q?")])

    assert out == ["The answer is B."]
    assert stub.calls == [("Q?", "eval")]


def test_multi_request_runs_each_through_the_service(monkeypatch):
    script = [
        [TokenChunk(text="A"), Done(final="A")],
        [TokenChunk(text="B"), Done(final="B")],
        [TokenChunk(text="C"), Done(final="C")],
    ]
    lm, stub = _make_lm(monkeypatch, script)

    out = lm.generate_until(
        [_instance("Q1?", idx=0), _instance("Q2?", idx=1), _instance("Q3?", idx=2)]
    )

    assert out == ["A", "B", "C"]
    assert stub.calls == [("Q1?", "eval"), ("Q2?", "eval"), ("Q3?", "eval")]


def test_ignores_non_token_events(monkeypatch):
    """Tool/retrieval events fire for audit purposes; the LM-facing
    completion sees only ``TokenChunk`` text."""
    script = [
        [
            ToolStarted(tool_name="translate.query"),
            TokenChunk(text="X"),
            ToolStarted(tool_name="retrieve_medical_literature"),
            TokenChunk(text="Y"),
            Done(final="XY"),
        ]
    ]
    lm, _ = _make_lm(monkeypatch, script)

    assert lm.generate_until([_instance("Q?")]) == ["XY"]


def test_done_with_no_token_chunks_returns_empty_string(monkeypatch):
    script = [[Done(final="")]]
    lm, _ = _make_lm(monkeypatch, script)

    assert lm.generate_until([_instance("Q?")]) == [""]


def test_provider_id_and_model_name_exposed(monkeypatch):
    script = [[TokenChunk(text="A"), Done(final="A")]]
    lm, _ = _make_lm(monkeypatch, script)

    assert lm.provider_id == "ollama"
    assert lm.model_name == "qwen3:14b"


# ---------------------------------------------------------------------------
# Latency tracking
# ---------------------------------------------------------------------------


def test_latencies_recorded_per_request(monkeypatch):
    script = [
        [TokenChunk(text="A"), Done(final="A")],
        [TokenChunk(text="B"), Done(final="B")],
    ]
    lm, _ = _make_lm(monkeypatch, script)

    lm.generate_until([_instance("Q1?", idx=0), _instance("Q2?", idx=1)])

    assert len(lm.latencies_ms) == 2
    assert all(ms >= 0 for ms in lm.latencies_ms)
    # doc_id is set from the Instance constructor (lm-eval populates it
    # from the dataset). Synthetic Instances here don't have one — but
    # the by-doc dict should remain consistent (empty or populated).
    assert isinstance(lm.latencies_ms_by_doc_id, dict)


def test_latency_recorded_even_when_call_raises(monkeypatch):
    """A mid-stream error still costs wall-clock time; the runner
    should be able to attribute it to a specific request."""
    script = [
        [
            TokenChunk(text="partial "),
            Error(
                error_type="llm_error",
                message="provider exploded",
                retryable=True,
            ),
        ]
    ]
    lm, _ = _make_lm(monkeypatch, script)

    with pytest.raises(RuntimeError, match="provider exploded"):
        lm.generate_until([_instance("Q?")])

    assert len(lm.latencies_ms) == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_error_event_raises_runtime_error(monkeypatch):
    script = [
        [
            Error(
                error_type="phi_violation",
                message="cloud provider blocked PHI",
                retryable=False,
            ),
        ]
    ]
    lm, _ = _make_lm(monkeypatch, script)

    with pytest.raises(RuntimeError, match="phi_violation"):
        lm.generate_until([_instance("Q?")])


def test_empty_args_rejected(monkeypatch):
    lm, _ = _make_lm(monkeypatch, [[Done(final="")]])
    req = Instance(
        request_type="generate_until",
        doc={},
        arguments=(),
        idx=0,
    )
    with pytest.raises(ValueError, match="no args"):
        lm.generate_until([req])


# ---------------------------------------------------------------------------
# Unsupported request types
# ---------------------------------------------------------------------------


def test_loglikelihood_raises_not_implemented(monkeypatch):
    lm, _ = _make_lm(monkeypatch, [[Done(final="")]])

    with pytest.raises(NotImplementedError, match="generate_until"):
        lm.loglikelihood([_instance("Q?")])


def test_loglikelihood_rolling_raises_not_implemented(monkeypatch):
    lm, _ = _make_lm(monkeypatch, [[Done(final="")]])

    with pytest.raises(NotImplementedError):
        lm.loglikelihood_rolling([_instance("Q?")])


# ---------------------------------------------------------------------------
# Synthetic user_id contract
# ---------------------------------------------------------------------------


def test_service_called_with_synthetic_eval_user_id(monkeypatch):
    """Every eval request goes through ``user_id="eval"`` — the
    inject_context wrap in cli/commands/eval.py picks this up for audit rows
    and Phoenix span baggage."""
    script = [[Done(final="")]]
    lm, stub = _make_lm(monkeypatch, script)

    lm.generate_until([_instance("Q?")])

    assert stub.calls == [("Q?", "eval")]


def test_latency_by_doc_id_populated_when_instance_carries_one(monkeypatch):
    """When lm-eval populates ``doc_id`` (real dataset), the adapter
    keys the latency dict by it so the runner can correlate without
    relying on list ordering."""
    script = [[TokenChunk(text="A"), Done(final="A")]]
    lm, _ = _make_lm(monkeypatch, script)

    req = Instance(
        request_type="generate_until",
        doc={},
        arguments=("Q?", {}),
        idx=0,
    )
    req.doc_id = 42  # lm-eval sets this from the dataset row id
    lm.generate_until([req])

    assert 42 in lm.latencies_ms_by_doc_id
    assert lm.latencies_ms_by_doc_id[42] == lm.latencies_ms[0]


# ---------------------------------------------------------------------------
# Default-strategy / default-rag-mode loaders
# ---------------------------------------------------------------------------


def test_default_rag_mode_is_deterministic(monkeypatch):
    """Eval is opinionated about rag_mode — defaults to deterministic
    so retrieval fires on every question. Project-wide default ``tool``
    mode is wrong for MCQA: short clinical vignettes don't trigger the
    LLM's "I should search" reflex, so the with-rag arm collapses to
    baseline (~20% tool-call rate observed on MedQA) and the delta
    signal disappears. Override via the ``rag_mode`` constructor arg
    when measuring agent behaviour rather than retrieval value."""
    monkeypatch.setattr(ClaritymedRagLM, "_build_service", lambda self: None)
    monkeypatch.setattr("claritymed.evals.lm.rag._default_strategy", lambda _p: None)
    lm = ClaritymedRagLM(_provider())
    assert lm._rag_mode == "deterministic"


def test_rag_mode_override_honored(monkeypatch):
    """Caller can ask for tool-mode (production agent behaviour) when
    measuring how often the LLM decides to search vs ignoring the tool."""
    monkeypatch.setattr(ClaritymedRagLM, "_build_service", lambda self: None)
    monkeypatch.setattr("claritymed.evals.lm.rag._default_strategy", lambda _p: None)
    lm = ClaritymedRagLM(_provider(), rag_mode="tool")
    assert lm._rag_mode == "tool"


def test_drain_consumes_to_natural_end_without_breaking(monkeypatch):
    """``AskService.run`` nests context managers (ContextVars + OTel
    span) whose token-reset must run in the Context they entered in.
    Breaking out of the loop early forces the cleanup to run via
    ``GeneratorExit`` injection (from ``aclose`` or asyncio shutdown),
    which lands in the wrong Context and produces "Token was created
    in a different Context" / "Failed to detach context" errors.

    Draining to the generator's natural end lets every ``finally`` /
    ``__exit__`` unwind cleanly. This test verifies that:

    1. We do *not* break on ``Done`` — the stub's ``finally`` runs as
       part of the natural for-loop exit, not as an injected close.
    2. Any events emitted *after* ``Done`` are discarded rather than
       appended (keeps the completion text faithful to the answer).
    """
    closed = {"finally_ran": False, "post_done_yields": 0}

    class _DrainStub:
        async def run(self, user_input: str, user_id: str):
            try:
                yield TokenChunk(text="A")
                yield Done(final="A")
                # A trailing chunk a future plugin might emit. Must NOT
                # land in the completion string but should still be
                # consumed so the generator ends cleanly.
                closed["post_done_yields"] += 1
                yield TokenChunk(text="\n\n**Sources:**\n- [1] foo")
            finally:
                closed["finally_ran"] = True

    stub = _DrainStub()
    monkeypatch.setattr(ClaritymedRagLM, "_build_service", lambda self: stub)
    monkeypatch.setattr("claritymed.evals.lm.rag._default_strategy", lambda _p: None)
    lm = ClaritymedRagLM(_provider())
    out = lm.generate_until([_instance("Q?")])

    # Trailing chunk was consumed but NOT appended to the completion.
    assert out == ["A"]
    assert closed["post_done_yields"] == 1
    # finally ran on natural exit, not on GeneratorExit injection.
    assert closed["finally_ran"] is True


# ---------------------------------------------------------------------------
# Per-question timeout
# ---------------------------------------------------------------------------


def _make_lm_with_timeout(
    monkeypatch,
    stub,
    *,
    timeout_s: float,
) -> ClaritymedRagLM:
    """Build a ClaritymedRagLM with a custom service stub and tight timeout."""
    monkeypatch.setattr(ClaritymedRagLM, "_build_service", lambda self: stub)
    monkeypatch.setattr("claritymed.evals.lm.rag._default_strategy", lambda _p: None)
    return ClaritymedRagLM(_provider(), question_timeout_s=timeout_s)


def test_hanging_question_returns_empty_within_timeout(monkeypatch):
    """A single rabbit-holed LLM call must not sink the run. Slow stub
    that exceeds the per-question timeout returns '' so the downstream
    filter regex misses → exact_match=0 → correct=False."""
    import asyncio as _asyncio

    # The CLI normally sets request_id / user_id / language ContextVars
    # via inject_context; in unit tests we stub audit_event so the
    # missing-context guard doesn't trip during the timeout path.
    monkeypatch.setattr("claritymed.evals.lm.rag.audit_event", lambda *a, **kw: None)

    class _HangingStub:
        async def run(self, user_input, user_id):
            # Long enough to blow past the tight test timeout.
            await _asyncio.sleep(5)
            yield TokenChunk(text="A")
            yield Done(final="A")

    stub = _HangingStub()
    lm = _make_lm_with_timeout(monkeypatch, stub, timeout_s=0.05)

    t0 = time.perf_counter()
    out = lm.generate_until([_instance("Q?")])
    elapsed = time.perf_counter() - t0

    assert out == [""]
    # Completed well before the stub would have produced anything.
    assert elapsed < 2.0, f"timeout did not interrupt promptly ({elapsed:.2f}s)"


def test_timeout_records_doc_id_and_audits(monkeypatch):
    """``timed_out_doc_ids`` lists every question that hit the cap and
    one ``eval.question.timeout`` audit row fires per hit."""
    import asyncio as _asyncio

    audit_payloads: list[dict] = []

    def _capture_audit(kind, payload=None):
        if kind == "eval.question.timeout":
            audit_payloads.append({"kind": kind, "payload": payload})

    monkeypatch.setattr("claritymed.evals.lm.rag.audit_event", _capture_audit)

    class _HangingStub:
        async def run(self, user_input, user_id):
            await _asyncio.sleep(5)
            yield Done(final="")

    stub = _HangingStub()
    lm = _make_lm_with_timeout(monkeypatch, stub, timeout_s=0.05)

    req1 = Instance(request_type="generate_until", doc={}, arguments=("Q1?", {}), idx=0)
    req1.doc_id = 101
    req2 = Instance(request_type="generate_until", doc={}, arguments=("Q2?", {}), idx=1)
    req2.doc_id = 202
    lm.generate_until([req1, req2])

    assert lm.timed_out_doc_ids == [101, 202]
    assert len(audit_payloads) == 2
    assert audit_payloads[0]["payload"]["doc_id"] == 101
    assert audit_payloads[0]["payload"]["timeout_s"] == 0.05
    assert audit_payloads[1]["payload"]["doc_id"] == 202


def test_normal_question_does_not_trigger_timeout_path(monkeypatch):
    """Fast-completing stubs don't get logged as timed-out and the
    audit row doesn't fire — proves the cap is only a backstop."""
    audit_calls: list[str] = []

    def _capture_audit(kind, payload=None):
        if kind == "eval.question.timeout":
            audit_calls.append(kind)

    monkeypatch.setattr("claritymed.evals.lm.rag.audit_event", _capture_audit)

    class _FastStub:
        async def run(self, user_input, user_id):
            yield TokenChunk(text="B")
            yield Done(final="B")

    stub = _FastStub()
    lm = _make_lm_with_timeout(monkeypatch, stub, timeout_s=5.0)

    req = Instance(request_type="generate_until", doc={}, arguments=("Q?", {}), idx=0)
    req.doc_id = 42
    out = lm.generate_until([req])

    assert out == ["B"]
    assert lm.timed_out_doc_ids == []
    assert audit_calls == []


def test_default_strategy_returns_none_when_rag_disabled(monkeypatch):
    """When ``rag.enabled=False`` the adapter runs the service in
    "no strategy" mode — useful for isolating PHI-guard cost from
    retrieval cost."""
    from claritymed.evals.lm import rag as rag_mod

    class _Cfg:
        class rag:  # noqa: D401, N801
            enabled = False
            mode = "tool"
            max_evidence = 5

        strategies = None

    monkeypatch.setattr(
        "claritymed.core.rag.schemas.load_retrieval_config", lambda: _Cfg()
    )
    assert rag_mod._default_strategy(_provider()) is None
