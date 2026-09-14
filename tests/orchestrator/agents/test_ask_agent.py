"""Tests for the ask agent module."""

from __future__ import annotations

from claritymed.core.prompts.registry import PromptRegistry
from claritymed.orchestrator.agents import make_ask_agent
from claritymed.orchestrator.agents.ask_agent import ASK_TOOL_NAMES


def test_make_ask_agent_streaming_text_shape():
    from pydantic_ai.models.test import TestModel

    registry = PromptRegistry()
    agent = make_ask_agent(TestModel(), registry=registry, language="en")
    from pydantic_ai import Agent

    assert isinstance(agent, Agent)


def test_ask_agent_does_not_have_ingest_save_tools():
    """Tool whitelist isolation: ask must not expose ingest save tools.

    The pydantic-ai Agent built by ``make_ask_agent`` registers no tools at
    all in Phase 1 (the service layer does the retrieval). The guardrail
    prevents future drift where someone wires save_to_profile (or any other
    ingest-side write tool) onto the ask agent by accident.
    """
    save_tools = {
        "save_to_profile",
        "save_to_lab_record",
        "save_to_vision_record",
    }
    for tool in ASK_TOOL_NAMES:
        assert tool not in save_tools
