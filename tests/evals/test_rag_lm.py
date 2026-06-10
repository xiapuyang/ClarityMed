"""Tests for ``ClaritymedRagLM`` — AskService → lm-eval LM bridge.

The strategy is: never build a real ``AskService``. The adapter only
talks to its inner service through ``service.run`` (an async generator
of events), so we substitute a stub service that yields a scripted
event sequence per call. Anything below the service layer — the model,
the retriever, the strategy, the PHI guard — is therefore irrelevant
to these unit tests.
"""

from __future__ import annotations

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
    # Defaults can read configs/retrieval.yaml; force stable values to
    # keep tests hermetic.
    monkeypatch.setattr(
        "claritymed.evals.lm.rag._default_rag_mode",
        lambda: "tool",
    )
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
    inject_context wrap in cli/eval.py picks this up for audit rows
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


def test_default_rag_mode_reads_config(monkeypatch):
    """``_default_rag_mode`` returns whatever the retrieval config has."""
    from claritymed.evals.lm import rag as rag_mod

    class _Cfg:
        class rag:  # noqa: D401, N801 — mimicking schema shape
            mode = "deterministic"

    monkeypatch.setattr(
        "claritymed.core.rag.schemas.load_retrieval_config", lambda: _Cfg()
    )
    assert rag_mod._default_rag_mode() == "deterministic"


def test_default_rag_mode_falls_back_when_config_unavailable(monkeypatch):
    """Config-load failures fall back to ``"tool"`` rather than crashing
    the adapter — the runner can still produce a useful "bare LLM"
    completion without retrieval."""
    from claritymed.evals.lm import rag as rag_mod

    def _explode():
        raise RuntimeError("config file missing")

    monkeypatch.setattr("claritymed.core.rag.schemas.load_retrieval_config", _explode)
    assert rag_mod._default_rag_mode() == "tool"


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
    monkeypatch.setattr("claritymed.evals.lm.rag._default_rag_mode", lambda: "tool")
    monkeypatch.setattr("claritymed.evals.lm.rag._default_strategy", lambda _p: None)
    lm = ClaritymedRagLM(_provider())
    out = lm.generate_until([_instance("Q?")])

    # Trailing chunk was consumed but NOT appended to the completion.
    assert out == ["A"]
    assert closed["post_done_yields"] == 1
    # finally ran on natural exit, not on GeneratorExit injection.
    assert closed["finally_ran"] is True


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
