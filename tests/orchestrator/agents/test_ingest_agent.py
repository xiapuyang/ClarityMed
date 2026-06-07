"""Tests for the ingest agent module: stubs, real save tools, factory."""

from __future__ import annotations

from claritymed.core.prompts.registry import PromptRegistry
from claritymed.orchestrator.agents import (
    INGEST_TOOL_NAMES,
    save_to_lab_record,
    save_to_profile,
    save_to_vision_record,
)
from claritymed.orchestrator.agents.ingest_agent import (
    extract_lab_fields_stub,
    make_ingest_agent,
    normalize_to_loinc_stub,
)


def test_tool_names_match_modes_config():
    from claritymed import config as _cfg

    modes = _cfg.load_modes_config()
    assert set(modes.get("ingest").tools) == set(INGEST_TOOL_NAMES)


def test_extract_lab_fields_stub_marks_self_as_stub():
    result = extract_lab_fields_stub("some report text")
    assert result["_stub"] is True


def test_normalize_to_loinc_stub_preserves_fields():
    result = normalize_to_loinc_stub([{"name": "Hb", "value": 12.0}])
    assert result[0]["name"] == "Hb"
    assert result[0]["loinc"] is None


def test_save_to_profile_returns_unique_record_id():
    a = save_to_profile("alice", "allergy", "penicillin")
    b = save_to_profile("alice", "allergy", "penicillin")
    assert a != b  # uuid suffix
    assert a.startswith("profile-alice-allergy-")


def test_save_to_lab_record_id_shape():
    rid = save_to_lab_record("alice", "doc1")
    assert rid.startswith("lab-alice-doc1-")


def test_save_to_vision_record_id_shape():
    rid = save_to_vision_record("alice", "doc1")
    assert rid.startswith("vision-alice-doc1-")


def test_make_ingest_agent_builds_with_real_model():
    """Smoke test: factory builds an Agent using a real provider, no LLM call."""
    from pydantic_ai.models.test import TestModel

    registry = PromptRegistry()
    agent = make_ingest_agent(TestModel(), registry=registry, language="en")
    # Factory returns a working Pydantic AI Agent without making a call.
    from pydantic_ai import Agent

    assert isinstance(agent, Agent)
