"""``AskService._run_post_process_hooks`` invocation contract.

The method itself is exercised directly here without standing up a full
``AskService`` — its job is small (walk plugins, call hook, swallow
errors) and a real AskService.run() smoke is covered by Unit 18 e2e.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


from claritymed.core.features.base import PostProcessHook
from claritymed.orchestrator.services.ask_service import AskService


class _AuditPlugin(PostProcessHook):
    name = "audit_plugin"
    mode = "tool"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def post_process(self, text: str, tool_result: dict) -> str:
        self.calls.append((text, tool_result))
        return text  # audit-only — no mutation


class _MutatingPlugin(PostProcessHook):
    name = "mutating_plugin"
    mode = "tool"

    async def post_process(self, text: str, tool_result: dict) -> str:
        return f"[mutated] {text}"


class _BoomPlugin(PostProcessHook):
    name = "boom"
    mode = "tool"

    async def post_process(self, text: str, tool_result: dict) -> str:
        raise RuntimeError("boom")


class _NoHookPlugin:
    name = "no_hook"
    mode = "tool"


def _service_with_features(features: list[Any]) -> AskService:
    """Construct an AskService bypassing the heavy init via __new__.

    The hook method only reads ``self._features``; everything else on
    AskService stays uninitialised. This keeps the test focused on the
    contract without dragging the full ask pipeline in.
    """
    service = AskService.__new__(AskService)
    service._features = features
    return service


def _deps_with_tool_calls(calls: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(tool_calls=calls)


async def test_hook_invoked_when_plugin_implements_protocol() -> None:
    plugin = _AuditPlugin()
    service = _service_with_features([plugin])
    result = {"final_text": "hello"}
    await service._run_post_process_hooks(
        _deps_with_tool_calls({"audit_plugin": 1}), result
    )
    assert plugin.calls == [("hello", result)]
    # audit-only plugin returns text unchanged
    assert result["final_text"] == "hello"


async def test_hook_skipped_when_plugin_lacks_protocol() -> None:
    """A plugin not implementing ``PostProcessHook`` is silently passed
    over — the contract is opt-in."""
    service = _service_with_features([_NoHookPlugin()])
    result = {"final_text": "hello"}
    await service._run_post_process_hooks(_deps_with_tool_calls({}), result)
    assert result["final_text"] == "hello"


async def test_mutating_hook_overwrites_final_text() -> None:
    """The Protocol allows mutation; the orchestrator honours it.
    Symptoms plugin is audit-only by KTD-2 choice, but the wiring must
    not preclude a future mutating implementation."""
    service = _service_with_features([_MutatingPlugin()])
    result = {"final_text": "hello"}
    await service._run_post_process_hooks(
        _deps_with_tool_calls({"mutating_plugin": 1}), result
    )
    assert result["final_text"] == "[mutated] hello"


async def test_hook_raising_does_not_break_reply(caplog) -> None:
    """A buggy hook must not break the user-visible reply — the
    orchestrator swallows the exception and logs a warning."""
    plugin = _BoomPlugin()
    service = _service_with_features([plugin])
    result = {"final_text": "hello"}
    with caplog.at_level("WARNING"):
        await service._run_post_process_hooks(
            _deps_with_tool_calls({"boom": 1}), result
        )
    assert result["final_text"] == "hello"
    assert any("post_process hook" in rec.message for rec in caplog.records)


async def test_hooks_chain_in_plugin_order() -> None:
    """When two plugins implement the protocol, both fire in the order
    they were registered — the second sees the first's output."""

    class _AppendOne(PostProcessHook):
        name = "p1"
        mode = "tool"

        async def post_process(self, text: str, tool_result: dict) -> str:
            return text + " A"

    class _AppendTwo(PostProcessHook):
        name = "p2"
        mode = "tool"

        async def post_process(self, text: str, tool_result: dict) -> str:
            return text + " B"

    service = _service_with_features([_AppendOne(), _AppendTwo()])
    result = {"final_text": "start"}
    await service._run_post_process_hooks(
        _deps_with_tool_calls({"p1": 1, "p2": 1}), result
    )
    assert result["final_text"] == "start A B"
