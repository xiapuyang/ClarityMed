"""Live full-stack chest-CT e2e — vision-server + medical-clip-server + real LLM.

Mirror of ``test_vision_e2e.py`` for the ``lung_cancer_chest_ct`` disease.
Skips cleanly when any of these gates fail:

1. ``http://127.0.0.1:8085/health`` reachable with non-empty
   ``models_loaded``.
2. ``http://127.0.0.1:8086/health`` reachable.
3. A promoted chest-CT artifact at
   ``~/.claritymed/models/vision/lung_cancer_chest_ct/lung_chest_ct_resnet50_v1/``.
4. ``tests/fixtures/vision/chest_ct/`` contains at least one real CT
   slice (placeholders filtered out).
5. A reachable LLM provider (shared ``e2e_provider_id`` fixture).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.context import apply_context, reset_context

from tests.e2e._vision_helpers import (
    USER_ID,
    assert_tool_invoked,
    pick_fixture_or_skip,
    require_promoted_artifact,
    require_vision_servers,
)

_PROMPT = (
    "Here is a chest CT axial slice I just had taken. Could this nodule "
    "be concerning? Please call the detect_disease_from_image tool with "
    "disease_id=lung_cancer_chest_ct on this image — do not answer in "
    "free text. [Image sha:{sha8}]"
)


@pytest.fixture(scope="module", autouse=True)
def _require_servers() -> None:
    require_vision_servers()


@pytest.fixture(scope="module", autouse=True)
def _require_promoted_chest_ct_model() -> None:
    require_promoted_artifact("lung_cancer_chest_ct", "lung_chest_ct_resnet50_v1")


@pytest.fixture(scope="module")
def _chest_ct_fixture_path() -> Path:
    return pick_fixture_or_skip("chest_ct")


@pytest.fixture
def _ctx():
    tokens = apply_context("20260616e2echesct0000000", USER_ID, "en")
    yield
    reset_context(tokens)


async def test_chest_ct_full_round_trip(
    e2e_provider_id: str,
    _chest_ct_fixture_path: Path,
    _ctx,
) -> None:
    """LLM picks the tool → confirm answers yes → server returns → reply branches."""
    await assert_tool_invoked(
        provider_id=e2e_provider_id,
        fixture_path=_chest_ct_fixture_path,
        modality="ct",
        prompt=_PROMPT,
        label="chest_ct",
    )
