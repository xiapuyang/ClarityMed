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
    features = build_features(
        profile_context_factory=lambda: ProfileContextFeature(mode="deterministic")
    )
    names = [f.name for f in features]
    assert "profile_context" in names


def test_factory_tool_includes_plugin():
    features = build_features(
        profile_context_factory=lambda: ProfileContextFeature(mode="tool")
    )
    names = [f.name for f in features]
    assert "profile_context" in names


def test_factory_off_excludes_plugin():
    features = build_features(profile_context_factory=None)
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


# ---------------------------------------------------------------------------
# _format_profile_block — branches not covered above
# ---------------------------------------------------------------------------


def test_format_block_all_biographic_fields(tmp_path, monkeypatch):
    """birth_date, residence, birthplace, marital_status, has_children,
    current_occupation all appear when set."""
    from datetime import date

    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    store = ProfileStore("test")
    store.upsert_profile(
        Profile(
            birth_date=date(1989, 1, 1),
            residence="Shanghai",
            birthplace="Beijing",
            marital_status="married",
            has_children=True,
            current_occupation="engineer",
        ),
        owner_user_id="test",
    )
    result = _format_profile_block("test")
    assert "birth_date: 1989-01-01" in result
    assert "residence: Shanghai" in result
    assert "birthplace: Beijing" in result
    assert "marital_status: married" in result
    assert "has_children: true" in result
    assert "current_occupation: engineer" in result


def test_format_block_condition_with_onset_date(tmp_path, monkeypatch):
    """Active condition with onset_date renders as 'display (since ONSET)'."""
    from datetime import date

    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    store = ProfileStore("test")
    store.add_condition(
        Condition(display="type 2 diabetes", onset_date=date(2020, 1, 1)),
        owner_user_id="test",
    )
    result = _format_profile_block("test")
    assert "type 2 diabetes (since 2020-01-01)" in result


def test_format_block_records_summary(tmp_path, monkeypatch):
    """Records section lists category/slug and metadata for recent manifests."""
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    from claritymed.stores.manifest_store import ManifestStore

    ms = ManifestStore("test", "records")
    # Four manifests so the "..." suffix is exercised end-to-end.
    for i in range(3):
        ms.create(
            "labs",
            f"2026-05-1{i}-lipid",
            {
                "title": f"Lipid panel 1{i}",
                "kind": "lab_report",
                "date": f"2026-05-1{i}",
            },
        )
    ms.create(
        "labs",
        "2026-05-20-lipid",
        {
            "title": "Lipid panel 20",
            "kind": "lab_report",
            "date": "2026-05-20",
        },
    )
    result = _format_profile_block("test")
    assert "Records (4 total):" in result
    assert "labs/2026-05-20-lipid" in result
    assert "..." in result


# ---------------------------------------------------------------------------
# retrieve_profile tool callable
# ---------------------------------------------------------------------------


async def test_retrieve_profile_tool_returns_formatted_block(tmp_path, monkeypatch):
    """The tool closure reads user_id from RunContext deps at call time."""
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path))
    store = ProfileStore("test")
    store.upsert_profile(Profile(sex="male"), owner_user_id="test")

    p = ProfileContextFeature(mode="tool")
    tool = p.as_tool()
    assert tool is not None

    class _Ctx:
        deps = _FakeDeps(user_id="test")

    result = await tool(_Ctx())
    assert "[Patient profile]" in result
    assert "sex: male" in result
