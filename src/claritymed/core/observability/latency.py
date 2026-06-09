"""Latency + per-call usage helpers used by the ask turn.

Lifted out of ``orchestrator/services/chat_session.py`` so it can be
imported by ``core/rag/`` and ``core/features/`` without crossing the
core → orchestrator boundary. ``ChatSession`` (persistence) stays in
orchestrator/services and pulls these helpers down from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.usage import RunUsage

_UNKNOWN_MODEL = "?"


class LatencyTrace(BaseModel):
    """Per-turn latency breakdown for performance triage.

    ``total_ms`` covers the full ``agent.run_stream`` span. ``ttft_ms`` is
    time-to-first-token (perceived UX latency). ``completion_ms`` is the
    streaming span from first token to stream close. For non-streaming or
    single-shot runs, ``ttft_ms`` / ``completion_ms`` may be ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_ms: int
    ttft_ms: int | None = None
    completion_ms: int | None = None

    def to_dict(self) -> dict[str, int]:
        out: dict[str, int] = {"totalMs": self.total_ms}
        if self.ttft_ms is not None:
            out["ttftMs"] = self.ttft_ms
        if self.completion_ms is not None:
            out["completionMs"] = self.completion_ms
        return out


def usage_dict(usage: "RunUsage") -> dict[str, int]:
    """Normalise pydantic-ai usage into a flat dict for audit + JSONL.

    ``input_tokens`` / ``output_tokens`` / ``total_tokens`` are always
    present. ``cache_read_tokens`` / ``cache_write_tokens`` are only
    included when non-zero — they show up on Anthropic / OpenAI when
    prompt caching is in play, and we price them differently from
    regular tokens so they need to be auditable.
    """
    input_tokens = (
        getattr(usage, "input_tokens", None)
        or getattr(usage, "request_tokens", None)
        or 0
    )
    output_tokens = (
        getattr(usage, "output_tokens", None)
        or getattr(usage, "response_tokens", None)
        or 0
    )
    total_tokens = getattr(usage, "total_tokens", None) or (
        input_tokens + output_tokens
    )
    out: dict[str, int] = {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(total_tokens),
    }
    cache_read = getattr(usage, "cache_read_tokens", 0) or 0
    cache_write = getattr(usage, "cache_write_tokens", 0) or 0
    if cache_read:
        out["cache_read_tokens"] = int(cache_read)
    if cache_write:
        out["cache_write_tokens"] = int(cache_write)
    return out


def build_step_records(messages: list["ModelMessage"]) -> list[dict[str, object]]:
    """Extract one record per LLM call from a list of ``ModelMessage``.

    Only ``ModelResponse`` messages carry per-call usage and the model
    name. For the current single-step ask agent this returns one record;
    once tools land it returns one per round-trip so we can see which
    step is slow.
    """
    from pydantic_ai.messages import ModelResponse

    records: list[dict[str, object]] = []
    for msg in messages:
        if not isinstance(msg, ModelResponse):
            continue
        record: dict[str, object] = {
            "model": getattr(msg, "model_name", None) or _UNKNOWN_MODEL,
        }
        ts = getattr(msg, "timestamp", None)
        if ts is not None:
            record["timestamp"] = ts.isoformat()
        usage = getattr(msg, "usage", None)
        if usage is not None:
            record["usage"] = usage_dict(usage)
        records.append(record)
    return records
