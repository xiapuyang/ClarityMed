"""``lm_eval.api.model.LM`` adapters for ClarityMed providers.

Each adapter is the seam between lm-evaluation-harness's request/response
protocol and one project subsystem:

* ``ClaritymedBaselineLM`` — wraps ``pydantic_ai.Agent`` directly. No PHI
  guard, no retrieval, no chat session. The "LLM only" arm of any
  baseline-vs-RAG comparison.
* ``ClaritymedRagLM``      — wraps ``AskService.run``: PHI scrub +
  retrieval + LLM stream. The "LLM + RAG" arm. Same lm-eval boundary,
  same JSONL output shape; the only difference is which subsystem
  produces the completion.
"""

from claritymed.evals.lm.baseline import ClaritymedBaselineLM
from claritymed.evals.lm.rag import ClaritymedRagLM

__all__ = ["ClaritymedBaselineLM", "ClaritymedRagLM"]
