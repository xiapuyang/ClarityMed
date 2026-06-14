"""LLM call logger: writes every request/response to logs/llm.log.

Active only when ``CLARITYMED_DEBUG=1``. Wraps a pydantic-ai ``Model``
via ``LoggingModel``; every ``request`` and ``request_stream`` call
produces a REQ/RES block in ``~/.claritymed/logs/llm.log``:

    ==== REQ 2026-06-10T03:27:24 rid=2026...  call#1 ====
    model: claude-sonnet-4-5  system: anthropic
    history: 4 messages  tools: ask_user_question, retrieve_medical_literature
      [1] [sys]   You are ClarityMed ...
      [2] [user]  which health checkup package should I choose
      [3] [asst]  ← tool_call: retrieve_medical_literature({"query": "..."})
      [4] [tool]  retrieve_medical_literature → [{"chunk": "..."}]
    ==== RES  total_ms=7213  ttft_ms=6195  call#1 ====
    finish: stop  tokens: in=1234 out=45 total=1279
      text: Based on your profile ...
    ----

    ==== REQ 2026-06-10T03:27:43 rid=2026...  call#2 ====
    ...
"""

from __future__ import annotations

import datetime
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic_ai.models import Model as _PydanticModel

from claritymed.context import request_id_ctx
from claritymed.core.observability.logging import get_llm_logger

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import ModelRequestParameters, StreamedResponse
    from pydantic_ai.settings import ModelSettings


def _safe(text: str) -> str:
    """Strip newlines and neutralise REQ-boundary lookalikes.

    Model output is logged verbatim with no length cap, so a malicious or
    confused model could otherwise inject a fake ``==== REQ ====`` line
    that downstream log parsers would treat as a real boundary. Stripping
    newlines and collapsing 4+ ``=`` runs to 3 keeps the log safe to grep.
    """
    return text.replace("\n", " ").replace("====", "===")


def _fmt_messages(messages: list[ModelMessage]) -> str:
    """Return a human-readable conversation thread (full content, all roles).

    Format:
        [N] [sys]   <system prompt>
        [N] [user]  <user text>
        [N] [asst]  ← tool_call: name(args)
        [N] [tool]  name → result
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        ThinkingPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )

    lines: list[str] = []
    idx = 0
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                idx += 1
                n = f"[{idx}]"
                if isinstance(part, SystemPromptPart):
                    lines.append(f"  {n} [sys]   {_safe(str(part.content))}")
                elif isinstance(part, UserPromptPart):
                    lines.append(f"  {n} [user]  {_safe(str(part.content))}")
                elif isinstance(part, ToolReturnPart):
                    lines.append(
                        f"  {n} [tool]  {part.tool_name} → {_safe(str(part.content))}"
                    )
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                idx += 1
                n = f"[{idx}]"
                if isinstance(part, TextPart):
                    lines.append(f"  {n} [asst]  {_safe(part.content)}")
                elif isinstance(part, ToolCallPart):
                    lines.append(
                        f"  {n} [asst]  ← tool_call: {part.tool_name}({_safe(str(part.args))})"
                    )
                elif isinstance(part, ThinkingPart):
                    lines.append(f"  {n} [asst/thinking]  {_safe(part.content)}")
    return "\n".join(lines) if lines else "  (no messages)"


def _fmt_response(response: Any) -> str:
    """Format ModelResponse parts for the log (full content, no length cap)."""
    from pydantic_ai.messages import TextPart, ThinkingPart, ToolCallPart

    lines: list[str] = []
    for part in getattr(response, "parts", []):
        if isinstance(part, TextPart):
            lines.append(f"  text: {_safe(part.content)}")
        elif isinstance(part, ToolCallPart):
            lines.append(f"  tool_call: {part.tool_name}({_safe(str(part.args))})")
        elif isinstance(part, ThinkingPart):
            lines.append(f"  thinking: {_safe(part.content)}")
    return "\n".join(lines) if lines else "  (no parts)"


def _fmt_usage(usage: Any) -> str:
    """Return a compact token-usage summary string from a pydantic-ai usage object."""
    if usage is None:
        return "n/a"
    from claritymed.core.observability.latency import usage_dict

    d = usage_dict(usage)
    parts = [
        f"in={d.get('input_tokens', 0)}",
        f"out={d.get('output_tokens', 0)}",
        f"total={d.get('total_tokens', 0)}",
    ]
    if d.get("cache_read_tokens"):
        parts.append(f"cache_read={d['cache_read_tokens']}")
    if d.get("cache_write_tokens"):
        parts.append(f"cache_write={d['cache_write_tokens']}")
    return " ".join(parts)


class _TimedStreamProxy:
    """Proxy for StreamedResponse that fires a callback on the first event."""

    def __init__(self, inner: StreamedResponse, on_first: Any) -> None:
        self._inner = inner
        self._on_first = on_first
        self._fired = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __aiter__(self) -> AsyncIterator:
        return self._tracked()

    async def _tracked(self) -> AsyncIterator:
        async for event in self._inner:
            if not self._fired:
                self._fired = True
                self._on_first(time.perf_counter())
            yield event


class LoggingModel:
    """Transparent ``Model`` wrapper that logs every LLM call to llm.log.

    Registered as a virtual subclass of pydantic-ai's ``Model`` ABC so
    ``isinstance(instance, Model)`` passes — this makes ``infer_model``
    return the wrapper unchanged instead of trying to parse it as a model
    name string. All ``Model`` protocol calls delegate to ``_inner`` via
    ``__getattr__``; ``request`` and ``request_stream`` are overridden
    explicitly to add logging.
    """

    def __init__(self, inner: "_PydanticModel") -> None:
        self._inner = inner
        # Per-request_id call counter so call#N is meaningful per turn.
        # OrderedDict gives insertion-order LRU eviction independent of rid format.
        self._counters: OrderedDict[str, int] = OrderedDict()

    # ---- delegation ---------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Guard against infinite recursion when _inner is missing
        # (e.g. object.__new__ without __init__ during unpickling).
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    async def __aenter__(self) -> "LoggingModel":
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> Any:
        return await self._inner.__aexit__(*args)

    # ---- helpers ------------------------------------------------------------

    def _next_call(self) -> tuple[str, int]:
        rid = request_id_ctx.get() or "-"
        n = self._counters.pop(rid, 0) + 1
        self._counters[rid] = n
        # Evict oldest entries (insertion order) to keep memory bounded.
        while len(self._counters) > 200:
            self._counters.popitem(last=False)
        return rid, n

    def _log_req(
        self,
        rid: str,
        call_n: int,
        messages: list[ModelMessage],
        params: ModelRequestParameters,
    ) -> None:
        tools = [t.name for t in (params.function_tools or [])]
        get_llm_logger().debug(
            "==== REQ %s  rid=%s  call#%d ====\n"
            "model: %s  system: %s\n"
            "history: %d messages  tools: %s\n"
            "%s",
            datetime.datetime.now().isoformat(timespec="seconds"),
            rid,
            call_n,
            self._inner.model_name,
            self._inner.system,
            len(messages),
            ", ".join(tools) or "(none)",
            _fmt_messages(messages),
        )

    def _log_res(
        self,
        rid: str,
        call_n: int,
        total_ms: int,
        ttft_ms: int | None,
        response: Any,
        usage: Any,
    ) -> None:
        timing = f"total_ms={total_ms}"
        if ttft_ms is not None:
            timing += f"  ttft_ms={ttft_ms}"
        get_llm_logger().debug(
            "==== RES  %s  call#%d ====\nfinish: %s  tokens: %s\n%s\n----\n",
            timing,
            call_n,
            getattr(response, "finish_reason", "?"),
            _fmt_usage(usage),
            _fmt_response(response) if response else "  (no response)",
        )

    # ---- pydantic-ai Model interface ----------------------------------------

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        rid, call_n = self._next_call()
        self._log_req(rid, call_n, messages, model_request_parameters)
        t0 = time.perf_counter()
        response = await self._inner.request(
            messages, model_settings, model_request_parameters
        )
        total_ms = int((time.perf_counter() - t0) * 1000)
        self._log_res(
            rid, call_n, total_ms, None, response, getattr(response, "usage", None)
        )
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[_TimedStreamProxy]:
        rid, call_n = self._next_call()
        self._log_req(rid, call_n, messages, model_request_parameters)

        t_start = time.perf_counter()
        t_first: list[float | None] = [None]

        async with self._inner.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as stream:
            yield _TimedStreamProxy(stream, lambda t: t_first.__setitem__(0, t))

        total_ms = int((time.perf_counter() - t_start) * 1000)
        ttft_ms = int((t_first[0] - t_start) * 1000) if t_first[0] is not None else None
        try:
            response = stream.get()
            usage = stream.usage
        except Exception:
            get_llm_logger().warning(
                "llm_logger: post-stream state read failed", exc_info=True
            )
            response = None
            usage = None
        self._log_res(rid, call_n, total_ms, ttft_ms, response, usage)


# Register as virtual subclass so isinstance(instance, Model) passes.
# pydantic-ai's infer_model() checks isinstance first; without this it
# falls through to parse_model_id() which does ':' in model → TypeError.
_PydanticModel.register(LoggingModel)
