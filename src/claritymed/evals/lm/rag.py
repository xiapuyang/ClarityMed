"""``ClaritymedRagLM`` — AskService wrapped as an lm-eval LM.

Phase 2 counterpart to ``ClaritymedBaselineLM``: same lm-eval-harness
boundary, but each ``generate_until`` request goes through the full
``AskService`` pipeline (PHI guard → RAG retrieval → LLM stream).
Comparing the two adapters on the same MedQA questions is the
"did RAG help or hurt" signal the §9-layer-5 floor demands.

Design notes:

* The adapter drains ``AskService.run`` (an ``AsyncIterator[Event]``)
  and concatenates ``TokenChunk.text`` into a single completion. All
  other events — ``ToolStarted`` / ``RetrievalCompleted`` / etc — are
  observed for audit (they fire ``mode.ask.*`` rows inside the service)
  but contribute nothing to the LM-facing completion. An ``Error``
  event is re-raised so lm-eval-harness sees the failure cleanly.
* ``chat_session=None``: a 1273-question MedQA run with chat session
  persistence would produce a giant session.jsonl with no review value.
  Each eval question is a fresh turn.
* ``user_id="eval"`` — synthetic, never a real account. Matches the
  CLI's ``inject_context(user_id="eval", ...)`` wrapping in
  ``cli/eval.py``. ``AskService`` doesn't enforce a user_exists check,
  so this stays clean.
* Sync interface: lm-eval-harness is synchronous; ``AskService.run`` is
  an async generator. Each request wraps the drain in ``asyncio.run``.
  Serial-only — batching across questions would need a separate harness
  (see plan §Risks).
* Per-call wall-clock latency is captured the same way as the baseline,
  so the runner's JSONL writer fills ``latency_ms`` without branching
  on which adapter produced the row.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from lm_eval.api.model import LM
from tqdm.auto import tqdm

from claritymed.core.events import Done, Error, TokenChunk
from claritymed.core.llm.model import build_model
from claritymed.core.observability.audit import audit_event

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

    from claritymed.core.rag.strategies.base import RagStrategy
    from claritymed.core.schemas import ProviderConfig


# Synthetic user id for every eval request. Matches the CLI's
# ``inject_context(user_id="eval", ...)`` wrap so ``audit.log`` and
# Phoenix span baggage stay consistent across the run.
_EVAL_USER_ID = "eval"

# Per-question wall-clock cap on the full RAG turn (retrieval + LLM
# stream). MedQA empirical budget is ~5–15 s/q on a local 7–35B model;
# 90 s is ~6× that, well past any healthy completion but tight enough
# that one rabbit-holed reasoning chain can't sink a 50-q run. On
# timeout the question scores wrong (empty completion → filter regex
# misses → exact_match=0) and the loop moves on — losing one row beats
# losing the whole batch.
_DEFAULT_QUESTION_TIMEOUT_S = 90.0

# Default RAG mode for eval runs. Deviates from configs/retrieval.yaml
# on purpose: the project default ``tool`` mode hands the retrieval
# decision to the LLM, which on MedQA-style MCQA prompts (short clinical
# vignette + four labelled options + "Answer with one letter") almost
# never fires — observed ~20% tool-call rate. That collapses the RAG
# arm into ~baseline performance and produces a near-zero delta that
# isn't a real signal. ``deterministic`` forces retrieval on every turn
# so the comparison answers the *intended* question — "does prepending
# retrieved evidence to the prompt help?" Tool-mode evaluation is still
# possible by passing ``rag_mode="tool"`` (CLI: ``--rag-mode tool``)
# when measuring agent behaviour rather than retrieval value.
_DEFAULT_RAG_MODE = "deterministic"


class ClaritymedRagLM(LM):
    """``AskService``-backed adapter: PHI guard + retrieval + LLM stream."""

    def __init__(
        self,
        provider: "ProviderConfig",
        *,
        strategy: "RagStrategy | None" = None,
        rag_mode: str | None = None,
        question_timeout_s: float = _DEFAULT_QUESTION_TIMEOUT_S,
    ) -> None:
        super().__init__()
        self.provider_id = provider.id
        self.model_name = provider.model
        self._provider = provider
        self._strategy = (
            strategy if strategy is not None else _default_strategy(provider)
        )
        self._rag_mode = rag_mode if rag_mode is not None else _DEFAULT_RAG_MODE
        self._question_timeout_s = question_timeout_s
        self._service = self._build_service()
        # Per-request wall-clock latency, in input order.
        self.latencies_ms: list[float] = []
        # Same data keyed by ``Instance.doc_id`` so the runner can
        # correlate latencies with samples even when ordering shifts.
        self.latencies_ms_by_doc_id: dict[int, float] = {}
        # Index of every question that hit the per-question timeout,
        # surfaced via the ``timed_out_doc_ids`` property so the runner
        # / tests can flag them in JSONL or summary output.
        self._timed_out_doc_ids: list[int] = []

    # ------------------------------------------------------------------
    # The only request type our task YAMLs use.
    # ------------------------------------------------------------------

    def generate_until(self, requests: list["Instance"]) -> list[str]:
        """Score each request by streaming AskService and joining tokens.

        ``gen_kwargs`` (``until`` / ``max_gen_toks`` / ``temperature``)
        are *not* forwarded — ``AskService`` owns its own prompt
        composition and model settings. Letting the task YAML stomp on
        those would break the RAG arm in ways orthogonal to the
        comparison this adapter exists to enable. The downstream
        ``filter_list`` regex still pulls the answer letter out of the
        joined completion exactly like the baseline arm.
        """
        completions: list[str] = []
        iterable = tqdm(
            requests,
            desc=f"eval[{self.provider_id}+rag]",
            unit="q",
            leave=False,
        )
        for req in iterable:
            context = self._unpack(req)
            t0 = time.perf_counter_ns()
            try:
                text = asyncio.run(self._drain(context, doc_id=req.doc_id))
            finally:
                elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
                self.latencies_ms.append(elapsed_ms)
                if req.doc_id is not None:
                    self.latencies_ms_by_doc_id[req.doc_id] = elapsed_ms

            completions.append(text)
        return completions

    # ------------------------------------------------------------------
    # Unsupported request types — fail loud, not silently.
    # ------------------------------------------------------------------

    def loglikelihood(self, requests):  # type: ignore[override]
        raise NotImplementedError(
            "ClaritymedRagLM uses generate_until scoring only; "
            "loglikelihood is unavailable because the RAG pipeline does "
            "not expose per-token logprobs. Pin output_type: generate_until "
            "in the task YAML."
        )

    def loglikelihood_rolling(self, requests):  # type: ignore[override]
        raise NotImplementedError(
            "ClaritymedRagLM uses generate_until scoring only; "
            "loglikelihood_rolling is unsupported."
        )

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _build_service(self):
        """Construct one ``AskService`` for the lifetime of this LM.

        Plugins are stateless across turns and the model handle is
        thread-safe under pydantic-ai's async stream, so a single service
        instance handles every question. Translation is wired so a
        future zh-MCQA task YAML (CMB-Exam) still works — for MedQA
        (English), the translation provider is a no-op pass-through.
        """
        from claritymed.core.translation import make_translation_provider
        from claritymed.orchestrator.services import AskService

        model = build_model(self._provider)
        return AskService(
            model=model,
            language="en",
            chat_session=None,
            provider_id=self._provider.id,
            model_name=str(self._provider.model),
            strategy=self._strategy,
            provider_config=self._provider,
            translation_service=make_translation_provider(model),
            rag_mode=self._rag_mode,
        )

    async def _drain(self, question: str, *, doc_id: int | None = None) -> str:
        """Drain ``AskService.run`` with a per-question wall-clock cap.

        ``asyncio.wait_for`` cancels the inner coroutine on timeout.
        That cancel propagates ``CancelledError`` through
        ``AskService``'s ``try/finally`` (ContextVar reset) and ``with
        start_as_current_span`` (OTel detach) — and crucially does so
        *inside the same Task/Context that entered them*, so neither
        teardown raises the "Token was created in a different Context"
        / "Failed to detach context" errors that hit when we let
        ``asyncio.run`` shutdown finalize a half-consumed generator.

        On timeout we return an empty string so the downstream
        ``filter_list`` regex misses → ``exact_match=0`` → ``correct``
        is recorded as ``False`` in JSONL. The audit row
        ``eval.question.timeout`` makes the event greppable for
        post-mortem.
        """
        try:
            return await asyncio.wait_for(
                self._drain_inner(question),
                timeout=self._question_timeout_s,
            )
        except asyncio.TimeoutError:
            if doc_id is not None:
                self._timed_out_doc_ids.append(doc_id)
            audit_event(
                "eval.question.timeout",
                payload={
                    "provider_id": self.provider_id,
                    "model_name": str(self.model_name),
                    "timeout_s": self._question_timeout_s,
                    "doc_id": doc_id,
                },
            )
            return ""

    async def _drain_inner(self, question: str) -> str:
        """Iterate the service event stream into a single string completion.

        Critical: never ``break`` or ``aclose`` the generator early.
        ``AskService.run`` wraps each turn in *two* nested
        context-manager pairs that record tokens at enter time and
        ``reset(token)`` at exit:

        * our own ``ContextVar`` tokens via ``apply_context`` /
          ``reset_context`` (in ``AskService.run``); and
        * an OpenTelemetry span via ``with start_as_current_span(...)``
          (in ``AskService._run_inner``), which calls ``context.detach``
          on exit.

        Both rely on the *exact* Context they entered in being current
        at exit. Draining to the generator's natural end lets every
        ``finally`` and ``__exit__`` unwind cleanly. ``Done`` is the
        terminal event in practice; anything emitted after it is
        discarded rather than appended so the completion text stays
        faithful to the model's answer.
        """
        parts: list[str] = []
        done = False
        async for event in self._service.run(question, user_id=_EVAL_USER_ID):
            if done:
                continue
            if isinstance(event, TokenChunk):
                parts.append(event.text)
            elif isinstance(event, Error):
                raise RuntimeError(
                    f"AskService emitted Error({event.error_type}): {event.message}"
                )
            elif isinstance(event, Done):
                done = True
        return "".join(parts)

    @property
    def timed_out_doc_ids(self) -> list[int]:
        """Doc ids that hit ``question_timeout_s``. Empty when none did."""
        return list(self._timed_out_doc_ids)

    @staticmethod
    def _unpack(request: "Instance") -> str:
        """Extract the prompt context from a ``generate_until`` Instance."""
        args = request.args
        if not args:
            raise ValueError(
                "generate_until request has no args; expected (context, gen_kwargs)"
            )
        return str(args[0])


# ---------------------------------------------------------------------------
# Defaults — kept module-level so tests can monkeypatch independently.
# ---------------------------------------------------------------------------


def _default_strategy(provider: "ProviderConfig"):
    """Build a strategy matching the active ``configs/retrieval.yaml``.

    Returns ``None`` when RAG is disabled — the service still runs but
    behaves like a bare LLM call (no retrieval evidence injected).
    Mirrors the cli/main.py ``_maybe_build_strategy`` helper so the
    eval RAG arm uses the same wiring as the ``ask`` command.
    """
    try:
        from claritymed.core.rag.schemas import load_retrieval_config
    except Exception:  # noqa: BLE001
        return None
    cfg = load_retrieval_config()
    if not cfg.rag.enabled:
        return None
    from claritymed.core.rag import build_hybrid_retriever
    from claritymed.core.rag.strategies import build_strategy

    retriever = build_hybrid_retriever(cfg)
    model = build_model(provider)
    return build_strategy(
        retriever,
        config=cfg.strategies,
        max_evidence=cfg.rag.max_evidence,
        model=model,
    )
