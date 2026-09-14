"""Unit tests for ``AskService._resolve_approvals``.

The deferred-tool loop has two halves: the framework-driven agent run
that materializes a ``DeferredToolRequests`` and the AskService-driven
resolver that walks each call through the channel and builds the
``DeferredToolResults``. The agent-run half is exercised by the
integration test suite; this file pins the resolver behavior on its
own so a channel regression surfaces without needing a live LLM.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, ToolApproved, ToolDenied

from claritymed.context import apply_context, reset_context
from claritymed.core.events import ToolCompleted, ToolStarted
from claritymed.core.interaction import (
    ApprovalDecision,
    InteractiveChannelUnavailable,
    ToolApprovalChannel,
)
from claritymed.core.schemas.models import ProviderConfig
from claritymed.orchestrator.services import AskService
from claritymed.stores.settings_store import SettingsStore


@pytest.fixture(autouse=True)
def _isolated_user_dir(monkeypatch, tmp_path: Path):
    """Redirect data/ writes into a tmp dir so SettingsStore tests don't
    pollute the developer's real user dir."""
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    yield


@pytest.fixture
def _ctx():
    tokens = apply_context("20260612000000ABCDEF12", "test", "en")
    yield
    reset_context(tokens)


class _StubChannel:
    """Records request() calls and returns a pre-seeded decision queue."""

    def __init__(self, *decisions: ApprovalDecision | Exception) -> None:
        self.decisions: list = list(decisions)
        self.calls: list[tuple[str, dict, str | None]] = []

    async def request(self, tool_name, args, *, breadcrumb=None):
        self.calls.append((tool_name, dict(args), breadcrumb))
        if not self.decisions:
            raise AssertionError("StubChannel ran out of decisions")
        out = self.decisions.pop(0)
        # CancelledError is a BaseException, not Exception, in 3.8+.
        if isinstance(out, BaseException):
            raise out
        return out


def _call_part(name: str, args: dict, call_id: str = "c1") -> ToolCallPart:
    return ToolCallPart(tool_name=name, args=args, tool_call_id=call_id)


def _make_service(channel: ToolApprovalChannel | None) -> AskService:
    """Build a barebones AskService — no chat_session, just the channel.

    The deferred-tool flow doesn't need any of the AskService state
    other than the channel reference, so we can skip the heavier
    wiring (RAG strategy, ChatSession) for these unit tests.
    """
    return AskService(
        model=TestModel(custom_output_text="noop"),
        provider_config=ProviderConfig(id="t", kind="local", model="openai:gpt-4o"),
        features=[],
        tool_approval_channel=channel,
    )


async def test_resolve_once_returns_tool_approved(_ctx):
    channel = _StubChannel(ApprovalDecision(decision="once"))
    service = _make_service(channel)
    deferred = DeferredToolRequests(
        approvals=[_call_part("save_allergy", {"substance": "penicillin"})]
    )
    out: asyncio.Queue = asyncio.Queue()
    results = await service._resolve_approvals(deferred, "test", out)

    assert "c1" in results.approvals
    decision = results.approvals["c1"]
    assert isinstance(decision, ToolApproved)
    assert decision.override_args is None
    assert channel.calls[0][0] == "save_allergy"
    assert channel.calls[0][2] == "Tool 1/1"


async def test_resolve_deny_returns_tool_denied(_ctx):
    channel = _StubChannel(ApprovalDecision(decision="deny"))
    service = _make_service(channel)
    deferred = DeferredToolRequests(approvals=[_call_part("save_allergy", {})])
    out: asyncio.Queue = asyncio.Queue()
    results = await service._resolve_approvals(deferred, "test", out)
    assert isinstance(results.approvals["c1"], ToolDenied)


async def test_resolve_always_tool_persists_rule(_ctx):
    channel = _StubChannel(ApprovalDecision(decision="always_tool"))
    service = _make_service(channel)
    deferred = DeferredToolRequests(
        approvals=[_call_part("save_allergy", {"substance": "penicillin"})]
    )
    out: asyncio.Queue = asyncio.Queue()
    await service._resolve_approvals(deferred, "test", out)

    rules = SettingsStore("test").list_rules()
    assert any(r.tool == "save_allergy" and r.action == "allow" for r in rules)


async def test_resolve_existing_deny_rule_short_circuits(_ctx):
    """A pre-existing deny rule must beat the modal — the user already
    said no for this shape and the channel must not be consulted."""
    SettingsStore("test").add_rule(
        "save_allergy", {"substance": "penicillin"}, action="deny"
    )
    channel = _StubChannel()  # No decisions queued — channel must not be called.
    service = _make_service(channel)
    deferred = DeferredToolRequests(
        approvals=[_call_part("save_allergy", {"substance": "penicillin"})]
    )
    out: asyncio.Queue = asyncio.Queue()
    results = await service._resolve_approvals(deferred, "test", out)
    assert isinstance(results.approvals["c1"], ToolDenied)
    assert channel.calls == []


async def test_resolve_channel_unavailable_returns_denied(_ctx):
    channel = _StubChannel(InteractiveChannelUnavailable("no TUI"))
    service = _make_service(channel)
    deferred = DeferredToolRequests(approvals=[_call_part("save_allergy", {})])
    out: asyncio.Queue = asyncio.Queue()
    results = await service._resolve_approvals(deferred, "test", out)
    assert isinstance(results.approvals["c1"], ToolDenied)


async def test_resolve_cancelled_mid_batch_denies_remaining(_ctx):
    """``asyncio.CancelledError`` mid-batch propagates a deny for the
    current call and short-circuits the rest of the batch without
    re-opening the modal — the host UI is going away."""
    channel = _StubChannel(
        ApprovalDecision(decision="once"),
        asyncio.CancelledError(),
    )
    service = _make_service(channel)
    deferred = DeferredToolRequests(
        approvals=[
            _call_part("save_allergy", {}, call_id="c1"),
            _call_part("save_condition", {}, call_id="c2"),
            _call_part("save_record", {}, call_id="c3"),
        ]
    )
    out: asyncio.Queue = asyncio.Queue()
    results = await service._resolve_approvals(deferred, "test", out)
    assert isinstance(results.approvals["c1"], ToolApproved)
    assert isinstance(results.approvals["c2"], ToolDenied)
    assert isinstance(results.approvals["c3"], ToolDenied)
    # The channel was consulted twice (the third was short-circuited).
    assert len(channel.calls) == 2


async def test_resolve_emits_tool_events(_ctx):
    channel = _StubChannel(ApprovalDecision(decision="once"))
    service = _make_service(channel)
    deferred = DeferredToolRequests(approvals=[_call_part("save_allergy", {})])
    out: asyncio.Queue = asyncio.Queue()
    await service._resolve_approvals(deferred, "test", out)
    events = []
    while not out.empty():
        events.append(out.get_nowait())
    assert any(isinstance(e, ToolStarted) for e in events)
    assert any(isinstance(e, ToolCompleted) for e in events)
