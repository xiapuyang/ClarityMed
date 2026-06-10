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

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

    from claritymed.core.rag.strategies.base import RagStrategy
    from claritymed.core.schemas import ProviderConfig


# Synthetic user id for every eval request. Matches the CLI's
# ``inject_context(user_id="eval", ...)`` wrap so ``audit.log`` and
# Phoenix span baggage stay consistent across the run.
_EVAL_USER_ID = "eval"


class ClaritymedRagLM(LM):
    """``AskService``-backed adapter: PHI guard + retrieval + LLM stream."""

    def __init__(
        self,
        provider: "ProviderConfig",
        *,
        strategy: "RagStrategy | None" = None,
        rag_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.provider_id = provider.id
        self.model_name = provider.model
        self._provider = provider
        self._strategy = (
            strategy if strategy is not None else _default_strategy(provider)
        )
        self._rag_mode = rag_mode if rag_mode is not None else _default_rag_mode()
        self._service = self._build_service()
        # Per-request wall-clock latency, in input order.
        self.latencies_ms: list[float] = []
        # Same data keyed by ``Instance.doc_id`` so the runner can
        # correlate latencies with samples even when ordering shifts.
        self.latencies_ms_by_doc_id: dict[int, float] = {}

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
                text = asyncio.run(self._drain(context))
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

    async def _drain(self, question: str) -> str:
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
        at exit. If we inject ``GeneratorExit`` mid-yield (via
        ``aclose`` from our outer ``asyncio.run`` loop), or worse, leak
        the half-consumed generator to ``asyncio.run`` shutdown, the
        cleanup runs in a different Context and the resets raise
        ``ValueError: Token was created in a different Context`` (or
        OTel logs "Failed to detach context"). Draining the generator
        to its natural end lets every ``finally`` and ``__exit__`` run
        in the right Context — no surgery in ``AskService`` required.

        ``Done`` is the terminal event in practice; anything emitted
        after it (extremely unlikely in current ``AskService``, but
        possible if a future plugin adds trailing diagnostics) is
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


def _default_rag_mode() -> str:
    """Read ``rag.mode`` from ``configs/retrieval.yaml``; fall back to ``tool``."""
    try:
        from claritymed.core.rag.schemas import load_retrieval_config

        return load_retrieval_config().rag.mode
    except Exception:  # noqa: BLE001 — config-load failures are non-fatal here
        return "tool"


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
