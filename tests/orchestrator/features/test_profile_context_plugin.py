"""Tests for ``ProfileContextFeature`` plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from claritymed.core.features import ProfileContextFeature, build_features
from claritymed.core.features.base import TurnContext
from claritymed.core.schemas import Allergy, Condition, Medication, Profile
from claritymed.orchestrator.features.profile_context_plugin import (
    _format_profile_block,
)
from claritymed.stores.profile import ProfileStore

_USER_ID = "test"


@dataclass
class _FakeDeps:
    """Minimal TurnState stand-in for tests."""

    user_id: str = _USER_ID
    language: str = "en"
    session_id: str | None = None
    strategy: Any = None


def _make_ctx(user_id: str = _USER_ID) -> TurnContext:
    return TurnContext(scrubbed="hello", deps=_FakeDeps(user_id=user_id))


# ---------------------------------------------------------------------------
# Construction / metadata
# ---------------------------------------------------------------------------


def test_default_mode_is_deterministic():
    p = ProfileContextFeature()
    assert p.mode == "deterministic"
    assert p.name == "profile_context"


def test_tool_mode_attrs():
    p = ProfileContextFeature(mode="tool")
    assert p.mode == "tool"
    tool = p.as_tool()
    assert tool is not None
    assert tool.__name__ == "retrieve_profile"
    assert p.as_toolset() is None


def test_deterministic_mode_as_tool_returns_none():
    p = ProfileContextFeature(mode="deterministic")
    assert p.as_tool() is None
    assert p.as_toolset() is None


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        ProfileContextFeature(mode="agentic")


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


def test_factory_deterministic_includes_plugin():
    features = build_features(profile_context_mode="deterministic")
    names = [f.name for f in features]
    assert "profile_context" in names


def test_factory_tool_includes_plugin():
    features = build_features(profile_context_mode="tool")
    names = [f.name for f in features]
    assert "profile_context" in names


def test_factory_off_excludes_plugin():
    features = build_features(profile_context_mode="off")
    names = [f.name for f in features]
    assert "profile_context" not in names


# ---------------------------------------------------------------------------
# __init__ lazy export
# ---------------------------------------------------------------------------


def test_init_lazy_export():
    from claritymed.core.features import ProfileContextFeature as PCF

    assert PCF is ProfileContextFeature


# ---------------------------------------------------------------------------
# _format_profile_block (synchronous formatting logic)
# ---------------------------------------------------------------------------


def test_format_block_empty_user(tmp_path, monkeypatch):
    """User with no data at all returns empty string."""
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    result = _format_profile_block("test")
    assert result == ""


def test_format_block_with_profile(tmp_path, monkeypatch):
    """Biometric fields appear in the header lines."""
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    store = ProfileStore("test")
    store.upsert_profile(
        Profile(sex="male", weight_kg=72.0, height_cm=178.0),
        owner_user_id="test",
    )
    result = _format_profile_block("test")
    assert "[Patient profile]" in result
    assert "sex: male" in result
    assert "weight_kg: 72.0" in result


def test_format_block_allergy_active(tmp_path, monkeypatch):
    """Active allergies appear; end_date-set allergies are excluded."""
    from datetime import date

    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    store = ProfileStore("test")
    store.add_allergy(
        Allergy(substance="penicillin", severity="severe", source="self_report"),
        owner_user_id="test",
    )
    # Resolved allergy — should not be counted
    store.add_allergy(
        Allergy(
            substance="aspirin",
            severity="mild",
            source="self_report",
            end_date=date(2020, 1, 1),
        ),
        owner_user_id="test",
    )
    result = _format_profile_block("test")
    assert "Allergies (1):" in result
    assert "penicillin" in result
    assert "aspirin" not in result


def test_format_block_condition_resolved_suffix(tmp_path, monkeypatch):
    """Resolved conditions get '(resolved)' suffix."""
    from datetime import date

    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    store = ProfileStore("test")
    store.add_condition(
        Condition(display="hypertension"),
        owner_user_id="test",
    )
    store.add_condition(
        Condition(display="old issue", end_date=date(2019, 6, 1)),
        owner_user_id="test",
    )
    result = _format_profile_block("test")
    assert "Active conditions (2):" in result
    assert "old issue (resolved)" in result


def test_format_block_medication_active_only(tmp_path, monkeypatch):
    """Only active medications (end_date=None) are listed."""
    from datetime import date

    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    store = ProfileStore("test")
    store.add_medication(
        Medication(display="metformin", dose="500mg", frequency="twice daily"),
        owner_user_id="test",
    )
    store.add_medication(
        Medication(display="old drug", end_date=date(2021, 1, 1)),
        owner_user_id="test",
    )
    result = _format_profile_block("test")
    assert "Current medications (1):" in result
    assert "metformin" in result
    assert "old drug" not in result


# ---------------------------------------------------------------------------
# pre_invoke async interface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_invoke_deterministic_empty_user(tmp_path, monkeypatch):
    """deterministic pre_invoke returns '' for user with no data."""
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    p = ProfileContextFeature(mode="deterministic")
    result = await p.pre_invoke(_make_ctx())
    assert result == ""


@pytest.mark.asyncio
async def test_pre_invoke_tool_mode_always_empty(tmp_path, monkeypatch):
    """tool mode pre_invoke always returns '' regardless of profile data."""
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    store = ProfileStore("test")
    store.upsert_profile(Profile(sex="female"), owner_user_id="test")

    p = ProfileContextFeature(mode="tool")
    result = await p.pre_invoke(_make_ctx())
    assert result == ""


@pytest.mark.asyncio
async def test_pre_invoke_deterministic_with_data(tmp_path, monkeypatch):
    """deterministic pre_invoke includes profile block when data exists."""
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    store = ProfileStore("test")
    store.upsert_profile(Profile(sex="female", weight_kg=60.0), owner_user_id="test")

    p = ProfileContextFeature(mode="deterministic")
    result = await p.pre_invoke(_make_ctx())
    assert "[Patient profile]" in result
    assert "sex: female" in result
