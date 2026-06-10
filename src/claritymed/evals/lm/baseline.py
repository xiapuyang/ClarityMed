"""``ClaritymedBaselineLM`` — pydantic-ai Agent wrapped as an lm-eval LM.

This is the *baseline* arm of any baseline-vs-RAG MCQA comparison: no PHI
guard, no retrieval, no chat session, no translation. The whole point is
"score the model exactly as it would answer with nothing but the prompt."

Design notes:

* Only ``generate_until`` is implemented. ``loglikelihood`` and
  ``loglikelihood_rolling`` raise ``NotImplementedError`` with an
  explanatory message — the task YAMLs in ``evals/tasks/`` all pin
  ``output_type: generate_until``, so these are never called in normal
  use. Failing loud catches a misconfigured future task YAML.
* Per-call latency is captured in ``self.latencies_ms`` (ordered list,
  one entry per ``generate_until`` request). The runner reads this list
  to populate the ``latency_ms`` JSONL column.
* Sync interface: lm-evaluation-harness is synchronous, and
  ``pydantic_ai.Agent.run`` is async — so each request goes through
  ``Agent.run_sync`` (pydantic-ai's own sync helper). For 1273-question
  MedQA this is fine; batching is deferred per plan §Risks.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from lm_eval.api.model import LM
from pydantic_ai import Agent

from claritymed.core.llm.model import build_model

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

    from claritymed.core.schemas import ProviderConfig

# Rough char-per-token approximation used to enforce ``max_gen_toks`` at
# the string layer. We don't tokenize at the adapter — for MCQA the
# downstream regex filter only needs the first letter, so being a little
# generous on the upper bound is harmless. Used only when ``max_gen_toks``
# is provided in ``gen_kwargs``.
_CHARS_PER_TOKEN_ESTIMATE = 4

# Default ``max_gen_toks`` when the task YAML doesn't specify. Matches
# lm-eval-harness's own default — high enough not to cut off an answer,
# low enough to keep runaway responses bounded.
_DEFAULT_MAX_GEN_TOKS = 256


class ClaritymedBaselineLM(LM):
    """Bare-model adapter: pydantic-ai Agent in, lm-eval LM out."""

    def __init__(self, provider: "ProviderConfig") -> None:
        super().__init__()
        self.provider_id = provider.id
        self.model_name = provider.model
        self._agent: Agent = Agent(build_model(provider), output_type=str)
        # Per-request wall-clock latency in milliseconds, one entry per
        # ``generate_until`` request in call order. Drained by the runner.
        self.latencies_ms: list[float] = []

    # ------------------------------------------------------------------
    # The only request type our task YAMLs use.
    # ------------------------------------------------------------------

    def generate_until(self, requests: list["Instance"]) -> list[str]:
        """Score each request by calling the agent and truncating the reply."""
        completions: list[str] = []
        for req in requests:
            context, gen_kwargs = self._unpack(req)
            until = self._normalize_until(gen_kwargs.get("until"))
            max_gen_toks = int(gen_kwargs.get("max_gen_toks", _DEFAULT_MAX_GEN_TOKS))

            t0 = time.perf_counter_ns()
            try:
                text = self._call_agent(context)
            finally:
                elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
                self.latencies_ms.append(elapsed_ms)

            completions.append(self._truncate(text, until, max_gen_toks))
        return completions

    # ------------------------------------------------------------------
    # Unsupported request types — fail loud, not silently.
    # ------------------------------------------------------------------

    def loglikelihood(self, requests):  # type: ignore[override]
        raise NotImplementedError(
            "ClaritymedBaselineLM uses generate_until scoring only; "
            "loglikelihood is unavailable because Anthropic does not expose "
            "per-token logprobs through pydantic-ai. Pin "
            "output_type: generate_until in the task YAML."
        )

    def loglikelihood_rolling(self, requests):  # type: ignore[override]
        raise NotImplementedError(
            "ClaritymedBaselineLM uses generate_until scoring only; "
            "loglikelihood_rolling is unsupported."
        )

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _call_agent(self, context: str) -> str:
        """Run the agent on a single context, returning the raw completion."""
        result = self._agent.run_sync(context)
        return str(result.output)

    @staticmethod
    def _unpack(request: "Instance") -> tuple[str, dict]:
        """Extract ``(context, gen_kwargs)`` from a ``generate_until`` Instance."""
        args = request.args
        if not args:
            raise ValueError(
                "generate_until request has no args; expected (context, gen_kwargs)"
            )
        context = str(args[0])
        gen_kwargs = args[1] if len(args) > 1 and isinstance(args[1], dict) else {}
        return context, gen_kwargs

    @staticmethod
    def _normalize_until(raw) -> list[str]:
        """Coerce ``gen_kwargs['until']`` into a list of stop strings."""
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        return [str(s) for s in raw if s]

    @staticmethod
    def _truncate(text: str, until: list[str], max_gen_toks: int) -> str:
        """Apply ``until`` substrings then a max-token char cap.

        The lm-eval harness expects the LM to honor the task's ``until``
        sequences itself (the harness doesn't post-process). Cap with the
        char-per-token approximation as a safety net — task YAMLs set
        ``max_gen_toks: 8`` for letter answers, so this stays tight.
        """
        for stop in until:
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx]
        char_cap = max(1, max_gen_toks * _CHARS_PER_TOKEN_ESTIMATE)
        return text[:char_cap]
