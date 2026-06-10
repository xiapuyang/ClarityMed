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
from pydantic_ai.settings import ModelSettings
from tqdm.auto import tqdm

from claritymed.core.llm.model import build_model

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

    from claritymed.core.schemas import ProviderConfig

# Default ``max_gen_toks`` when the task YAML doesn't specify. Matches
# lm-eval-harness's own default — high enough not to cut off an answer
# from a reasoning model, low enough to keep runaway responses bounded.
_DEFAULT_MAX_GEN_TOKS = 256

# Prompt registry name for the MCQA system instruction. The bilingual
# template lives at ``core/prompts/store/eval_mcqa_instructions.yaml``
# and is versioned per CLAUDE.md's prompt-immutability rule. Hardcoding
# the text here is forbidden by project standards.
_MCQA_INSTRUCTIONS_PROMPT = "eval_mcqa_instructions"

# Sentinel for "caller did not pass anything; load the registry default".
# A separate token distinguishes that case from ``instructions=None``,
# which is an explicit request to run the model with no system message
# (useful for vanilla-baseline runs that compare against any tuning).
_USE_DEFAULT_INSTRUCTIONS = object()


def _default_instructions(language: str = "en") -> str:
    """Read the MCQA system instruction from the prompt registry."""
    from claritymed.core.prompts.registry import get_default_registry

    return get_default_registry().get(_MCQA_INSTRUCTIONS_PROMPT, language=language)  # type: ignore[arg-type]


class ClaritymedBaselineLM(LM):
    """Bare-model adapter: pydantic-ai Agent in, lm-eval LM out."""

    def __init__(
        self,
        provider: "ProviderConfig",
        *,
        instructions=_USE_DEFAULT_INSTRUCTIONS,
    ) -> None:
        super().__init__()
        self.provider_id = provider.id
        self.model_name = provider.model
        # Use the registry default unless the caller explicitly passed
        # something (including ``instructions=None`` to request no
        # system message at all).
        if instructions is _USE_DEFAULT_INSTRUCTIONS:
            instructions = _default_instructions()
        self._agent: Agent = Agent(
            build_model(provider),
            output_type=str,
            instructions=instructions,
        )
        # Per-request wall-clock latency in milliseconds, one entry per
        # ``generate_until`` request in call order. Drained by the runner.
        self.latencies_ms: list[float] = []
        # Same data keyed by ``Instance.doc_id`` for stable runner→sample
        # correlation (lm-eval populates ``doc_id`` on each Instance).
        # Falls back to None when doc_id is unavailable (synthetic tests).
        self.latencies_ms_by_doc_id: dict[int, float] = {}

    # ------------------------------------------------------------------
    # The only request type our task YAMLs use.
    # ------------------------------------------------------------------

    def generate_until(self, requests: list["Instance"]) -> list[str]:
        """Score each request by calling the agent and forwarding gen_kwargs.

        The full agent response is returned verbatim — reasoning models
        produce thinking + answer in one content blob, and the downstream
        ``filter_list`` regex is responsible for pulling the answer out.
        Truncating client-side would lose the trailing "Answer: X" that
        the filter needs.

        Progress is reported via tqdm because lm-eval-harness's own bar
        only covers dataset preparation — once it hands off the full
        batch of Instances to us, there's no visible signal until every
        request comes back. For a 1273-question MedQA run at ~5s per
        question that's ~100 minutes of apparent dead silence; the
        per-request bar below gives a real-time q/s + ETA instead.
        """
        completions: list[str] = []
        iterable = tqdm(
            requests,
            desc=f"eval[{self.provider_id}]",
            unit="q",
            leave=False,
        )
        for req in iterable:
            context, gen_kwargs = self._unpack(req)
            settings = self._build_settings(gen_kwargs)

            t0 = time.perf_counter_ns()
            try:
                text = self._call_agent(context, settings)
            finally:
                elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
                self.latencies_ms.append(elapsed_ms)
                if req.doc_id is not None:
                    self.latencies_ms_by_doc_id[req.doc_id] = elapsed_ms

            completions.append(text)
        return completions

    @staticmethod
    def _build_settings(gen_kwargs: dict) -> ModelSettings | None:
        """Translate task YAML ``generation_kwargs`` into pydantic-ai settings.

        Forwarding ``max_gen_toks`` / ``until`` / ``temperature`` to the wire
        is what lets the API enforce the cap, instead of us truncating the
        response after the model already spent latency generating it. Returns
        ``None`` when there's nothing to override.
        """
        max_gen_toks = gen_kwargs.get("max_gen_toks")
        until = ClaritymedBaselineLM._normalize_until(gen_kwargs.get("until"))
        temperature = gen_kwargs.get("temperature")

        kwargs: dict = {}
        if max_gen_toks is not None:
            kwargs["max_tokens"] = int(max_gen_toks)
        if until:
            kwargs["stop_sequences"] = until
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        return ModelSettings(**kwargs) if kwargs else None

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

    def _call_agent(self, context: str, settings: ModelSettings | None) -> str:
        """Run the agent on a single context, returning the raw completion."""
        if settings is None:
            result = self._agent.run_sync(context)
        else:
            result = self._agent.run_sync(context, model_settings=settings)
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
