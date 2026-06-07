"""Tests for the ask agent module."""

from __future__ import annotations

from claritymed.core.prompts.registry import PromptRegistry
from claritymed.orchestrator.agents import make_ask_agent
from claritymed.orchestrator.agents.ask_agent import ASK_TOOL_NAMES


def test_tool_names_match_modes_config():
    from claritymed import config as _cfg

    modes = _cfg.load_modes_config()
    assert set(modes.get("ask").tools) == set(ASK_TOOL_NAMES)


def test_make_ask_agent_streaming_text_shape():
    from pydantic_ai.models.test import TestModel

    registry = PromptRegistry()
    agent = make_ask_agent(TestModel(), registry=registry, language="en")
    from pydantic_ai import Agent

    assert isinstance(agent, Agent)


def test_ask_agent_does_not_have_ingest_save_tools():
    """Tool whitelist isolation: ask must not expose ingest save tools.

    The pydantic-ai Agent built by ``make_ask_agent`` registers no tools at
    all in Phase 1 (the service layer does the retrieval), so the test is
    that the tool name lists across modes have no overlap with destructive
    save tools — preventing future drift where someone wires save_to_profile
    onto the ask agent by accident.
    """
    from claritymed.orchestrator.agents.ingest_agent import INGEST_TOOL_NAMES

    save_tools = {
        "save_to_profile",
        "save_to_lab_record",
        "save_to_vision_record",
    }
    for tool in ASK_TOOL_NAMES:
        assert tool not in save_tools
    # And the ingest mode does include these saves.
    for tool in save_tools:
        assert tool in INGEST_TOOL_NAMES
