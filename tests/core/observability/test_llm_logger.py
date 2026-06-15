"""Tests for the LLM call logger (``core/observability/llm_logger.py``).

Covers the pure helper formatters, the per-request_id call counter, the
``LoggingModel`` request/request_stream paths via a stub inner model, and
the ``_TimedStreamProxy`` first-event timing hook.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
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
from pydantic_ai.models import Model as _PydanticModel

from claritymed.context import request_id_ctx
from claritymed.core.observability.llm_logger import (
    LoggingModel,
    _TimedStreamProxy,
    _fmt_messages,
    _fmt_response,
    _fmt_usage,
    _safe,
)


# ----- pure formatter helpers --------------------------------------------


def test_safe_strips_newlines_and_neutralises_boundary():
    text = "line1\nline2 ==== fake REQ ====\n"
    safe = _safe(text)
    assert "\n" not in safe
    # 4+ equals collapsed to 3.
    assert "====" not in safe
    assert "===" in safe


def test_fmt_messages_renders_each_part_type():
    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart(content="sys text"),
                UserPromptPart(content="user text"),
                ToolReturnPart(
                    tool_name="lookup", content="tool result", tool_call_id="abc"
                ),
            ]
        )
    ]
    out = _fmt_messages(messages)
    assert "[sys]" in out and "sys text" in out
    assert "[user]" in out and "user text" in out
    assert "[tool_result]" in out and "lookup" in out and "tool result" in out


def test_fmt_messages_includes_model_response():
    """ModelResponse (assistant turns) are now included in the thread."""
    resp = ModelResponse(parts=[TextPart(content="hi")])
    out = _fmt_messages([resp])
    assert "[assistant]" in out and "hi" in out


def test_fmt_messages_renders_assistant_tool_call():
    messages = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="get_drug", args={"name": "aspirin"}, tool_call_id="t1"
                )
            ]
        )
    ]
    out = _fmt_messages(messages)
    assert "← tool_call: get_drug" in out


def test_fmt_messages_empty_list_returns_sentinel():
    assert "(no messages)" in _fmt_messages([])


def test_fmt_messages_sequence_numbers_are_monotonic():
    """Each part gets a unique [N] index, incrementing across msg boundaries."""
    messages = [
        ModelRequest(
            parts=[SystemPromptPart(content="s"), UserPromptPart(content="u")]
        ),
        ModelResponse(parts=[TextPart(content="a")]),
    ]
    out = _fmt_messages(messages)
    assert "[1]" in out
    assert "[2]" in out
    assert "[3]" in out


def test_fmt_response_renders_each_part_type():
    resp = ModelResponse(
        parts=[
            TextPart(content="answer text"),
            ToolCallPart(tool_name="lookup", args={"q": "x"}, tool_call_id="abc"),
            ThinkingPart(content="reasoning trace"),
        ]
    )
    out = _fmt_response(resp)
    assert "text: answer text" in out
    assert "tool_call: lookup" in out
    assert "thinking: reasoning trace" in out


def test_fmt_response_handles_no_parts():
    resp = SimpleNamespace(parts=[])
    assert "(no parts)" in _fmt_response(resp)


def test_fmt_response_handles_object_without_parts_attr():
    # `getattr(response, 'parts', [])` falls back to []
    assert "(no parts)" in _fmt_response(object())


def test_fmt_usage_none_returns_na():
    assert _fmt_usage(None) == "n/a"


def test_fmt_usage_minimal():
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)
    out = _fmt_usage(usage)
    assert "in=10" in out
    assert "out=5" in out
    assert "total=15" in out
    # No cache info present.
    assert "cache_read" not in out
    assert "cache_write" not in out


def test_fmt_usage_with_cache_tokens():
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cache_read_tokens=3,
        cache_write_tokens=2,
    )
    out = _fmt_usage(usage)
    assert "cache_read=3" in out
    assert "cache_write=2" in out


# ----- _TimedStreamProxy --------------------------------------------------


class _FakeStream:
    """Async iterable that yields events from a list."""

    def __init__(self, items: list) -> None:
        self._items = items
        self.tag = "inner"  # for __getattr__ delegation test

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for item in self._items:
            yield item


async def test_timed_stream_proxy_delegates_attributes():
    inner = _FakeStream([])
    proxy = _TimedStreamProxy(inner, lambda _t: None)
    # __getattr__ → inner attribute
    assert proxy.tag == "inner"


async def test_timed_stream_proxy_fires_callback_on_first_event_only():
    inner = _FakeStream(["a", "b", "c"])
    captured: list[float] = []

    proxy = _TimedStreamProxy(inner, lambda t: captured.append(t))
    received = []
    async for event in proxy:
        received.append(event)

    assert received == ["a", "b", "c"]
    # Callback fires exactly once (on first event).
    assert len(captured) == 1


# ----- LoggingModel: counters, request, stream ---------------------------


class _StubInnerModel(_PydanticModel):
    """Minimal pydantic-ai Model stand-in.

    Only implements what LoggingModel calls; the rest delegates via __getattr__.
    """

    model_name = "stub-model"
    system = "stub-system"

    def __init__(self) -> None:
        self.request_calls = 0
        self.stream_calls = 0

    async def request(self, messages, model_settings, model_request_parameters):
        self.request_calls += 1
        # Reuse pydantic-ai's real ModelResponse so _fmt_response works.
        return ModelResponse(
            parts=[TextPart(content="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3),
            finish_reason="stop",
        )

    @asynccontextmanager
    async def request_stream(
        self, messages, model_settings, model_request_parameters, run_context=None
    ):
        self.stream_calls += 1
        yield _FakeStream(["chunk1", "chunk2"])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def model_name_for_request(self, *args, **kwargs):  # pragma: no cover - unused
        return self.model_name

    @property
    def base_url(self) -> str:  # required abstract property
        return ""


def _params(tools: list[str] | None = None):
    """Minimal stand-in for ModelRequestParameters — only `.function_tools` is read."""
    tool_defs = [SimpleNamespace(name=t) for t in (tools or [])]
    return SimpleNamespace(function_tools=tool_defs)


def test_logging_model_registered_as_virtual_subclass():
    """isinstance(LoggingModel(...), Model) must pass — critical for infer_model."""
    inner = _StubInnerModel()
    lm = LoggingModel(inner)
    assert isinstance(lm, _PydanticModel)


def test_logging_model_delegates_attribute_access():
    inner = _StubInnerModel()
    lm = LoggingModel(inner)
    # model_name forwards to inner
    assert lm.model_name == "stub-model"


def test_logging_model_inner_missing_raises_attributeerror():
    """Guard against infinite recursion when _inner is missing (unpickling case)."""
    lm = LoggingModel.__new__(LoggingModel)
    # _inner not set; __getattr__ for `_inner` must raise, not recurse.
    with pytest.raises(AttributeError):
        lm._inner  # noqa: B018


def test_next_call_counter_per_request_id():
    inner = _StubInnerModel()
    lm = LoggingModel(inner)
    token = request_id_ctx.set("req-1")
    try:
        assert lm._next_call() == ("req-1", 1)
        assert lm._next_call() == ("req-1", 2)
    finally:
        request_id_ctx.reset(token)

    token2 = request_id_ctx.set("req-2")
    try:
        assert lm._next_call() == ("req-2", 1)
    finally:
        request_id_ctx.reset(token2)


def test_next_call_uses_dash_when_no_request_id():
    inner = _StubInnerModel()
    lm = LoggingModel(inner)
    rid, n = lm._next_call()
    assert rid == "-"
    assert n == 1


def test_next_call_evicts_oldest_when_over_limit():
    """Counter map is capped at 200 entries — oldest gets evicted."""
    inner = _StubInnerModel()
    lm = LoggingModel(inner)
    for i in range(250):
        token = request_id_ctx.set(f"req-{i}")
        try:
            lm._next_call()
        finally:
            request_id_ctx.reset(token)
    assert len(lm._counters) == 200
    # The first ones should have been evicted.
    assert "req-0" not in lm._counters
    assert "req-249" in lm._counters


async def test_request_logs_req_and_res(caplog):
    inner = _StubInnerModel()
    lm = LoggingModel(inner)
    messages = [ModelRequest(parts=[UserPromptPart(content="hello")])]
    params = _params(tools=["tool_a"])

    with caplog.at_level(logging.DEBUG, logger="claritymed.llm"):
        result = await lm.request(messages, None, params)

    assert inner.request_calls == 1
    assert isinstance(result, ModelResponse)
    log_text = "\n".join(rec.message for rec in caplog.records)
    assert "==== REQ" in log_text
    assert "==== RES" in log_text
    assert "----" in log_text
    assert "tool_a" in log_text
    assert "[user]" in log_text and "hello" in log_text
    # response part rendered
    assert "text: ok" in log_text
    # finish reason and tokens captured
    assert "finish: stop" in log_text
    assert "in=1" in log_text


async def test_request_logs_no_tools_as_none(caplog):
    inner = _StubInnerModel()
    lm = LoggingModel(inner)
    messages = [ModelRequest(parts=[UserPromptPart(content="hi")])]
    params = _params(tools=None)
    with caplog.at_level(logging.DEBUG, logger="claritymed.llm"):
        await lm.request(messages, None, params)
    log_text = "\n".join(rec.message for rec in caplog.records)
    assert "tools: (none)" in log_text


async def test_request_stream_logs_req_and_res(caplog):
    inner = _StubInnerModel()
    lm = LoggingModel(inner)
    messages = [ModelRequest(parts=[UserPromptPart(content="streaming")])]
    params = _params(tools=[])

    with caplog.at_level(logging.DEBUG, logger="claritymed.llm"):
        async with lm.request_stream(messages, None, params, None) as proxy:
            collected = [event async for event in proxy]

    assert inner.stream_calls == 1
    assert collected == ["chunk1", "chunk2"]
    log_text = "\n".join(rec.message for rec in caplog.records)
    assert "==== REQ" in log_text
    assert "==== RES" in log_text


async def test_request_stream_handles_post_state_read_failure(caplog):
    """When stream.get()/stream.usage raise after the context exits, the
    logger must warn but not propagate."""

    class _PostFailStream(_FakeStream):
        def get(self):
            raise RuntimeError("nope")

        @property
        def usage(self):
            raise RuntimeError("nope")

    inner = _StubInnerModel()

    @asynccontextmanager
    async def _bad_request_stream(*args, **kwargs):
        yield _PostFailStream(["x"])

    # Monkey-patch the inner method.
    inner.request_stream = _bad_request_stream  # type: ignore[method-assign]
    lm = LoggingModel(inner)
    messages = [ModelRequest(parts=[UserPromptPart(content="x")])]
    params = _params(tools=[])

    with caplog.at_level(logging.WARNING, logger="claritymed.llm"):
        async with lm.request_stream(messages, None, params, None) as proxy:
            _ = [e async for e in proxy]

    # No exception leaked; warning was emitted.
    assert any("post-stream state read failed" in rec.message for rec in caplog.records)


async def test_aenter_aexit_delegate_to_inner():
    class _ContextInner(_StubInnerModel):
        def __init__(self):
            super().__init__()
            self.entered = False
            self.exited = False

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, *args):
            self.exited = True
            return False

    inner: Any = _ContextInner()
    lm = LoggingModel(inner)
    async with lm as same:
        assert same is lm
    assert inner.entered is True
    assert inner.exited is True
