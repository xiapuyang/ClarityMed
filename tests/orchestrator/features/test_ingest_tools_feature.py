"""Tests for ``IngestToolsFeature`` plugin wrapper + factory wiring."""

from __future__ import annotations

from claritymed.core.features import build_features
from claritymed.orchestrator.features.ingest_tools_plugin import (
    IngestToolsFeature,
)
from claritymed.orchestrator.services.tool_dispatcher import ToolDispatcher


def test_ingest_feature_metadata():
    feature = IngestToolsFeature(ToolDispatcher(), approval_required_func=None)
    assert feature.name == "ingest_tools"
    assert feature.mode == "tool"


def test_ingest_feature_as_tool_returns_none():
    feature = IngestToolsFeature(ToolDispatcher(), approval_required_func=None)
    assert feature.as_tool() is None


def test_ingest_feature_as_toolset_plain_when_no_approval_func():
    """Without an approval_required_func, the bundled toolset is the raw
    ``FunctionToolset`` — used by ``cli tool --auto-approve`` and tests."""
    from pydantic_ai.toolsets import FunctionToolset

    feature = IngestToolsFeature(ToolDispatcher(), approval_required_func=None)
    toolset = feature.as_toolset()
    assert isinstance(toolset, FunctionToolset)


def test_ingest_feature_as_toolset_wraps_with_approval():
    """With an approval_required_func, the toolset is wrapped in
    ``ApprovalRequiredToolset`` so every call routes through the gate."""
    from pydantic_ai.toolsets import ApprovalRequiredToolset

    def _always_required(_ctx, _tool_def, _args) -> bool:
        return True

    feature = IngestToolsFeature(
        ToolDispatcher(), approval_required_func=_always_required
    )
    toolset = feature.as_toolset()
    assert isinstance(toolset, ApprovalRequiredToolset)


def test_build_features_includes_ingest_when_factory_provided():
    """``ingest_factory`` is the build_features hook AskService uses to
    add the plugin when the TUI exposes an approval channel."""

    def _factory():
        return IngestToolsFeature(ToolDispatcher(), approval_required_func=None)

    plugins = build_features(
        rag_mode="tool",
        rag_strategy=None,
        get_session_id=lambda: "sess-x",
        ingest_factory=_factory,
    )
    assert any(isinstance(p, IngestToolsFeature) for p in plugins)


def test_build_features_omits_ingest_by_default():
    """Without ``ingest_factory``, the plugin must not be auto-added —
    headless / one-shot CLI paths have no approval channel and would
    deny every call."""
    plugins = build_features(
        rag_mode="tool",
        rag_strategy=None,
        get_session_id=lambda: "sess-x",
    )
    assert not any(isinstance(p, IngestToolsFeature) for p in plugins)
