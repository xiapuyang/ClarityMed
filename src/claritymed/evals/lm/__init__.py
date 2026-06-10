"""``lm_eval.api.model.LM`` adapters for ClarityMed providers.

Each adapter is the seam between lm-evaluation-harness's request/response
protocol and one project subsystem:

* ``ClaritymedBaselineLM`` — wraps ``pydantic_ai.Agent`` directly. No PHI
  guard, no retrieval, no chat session. The "LLM only" arm of any
  baseline-vs-RAG comparison.
* ``ClaritymedRagLM``      — Phase 2; wraps ``AskService.run``.
"""

from claritymed.evals.lm.baseline import ClaritymedBaselineLM

__all__ = ["ClaritymedBaselineLM"]
