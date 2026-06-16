"""Live full-stack skin-lesion e2e — vision-server + medical-clip-server + real LLM.

Mirror of ``test_vision_e2e.py`` for the ``skin_cancer_dermoscopy`` disease.
Skips cleanly when any of these gates fail:

1. ``http://127.0.0.1:8085/health`` reachable with non-empty
   ``models_loaded``.
2. ``http://127.0.0.1:8086/health`` reachable.
3. A promoted skin-lesion artifact at
   ``~/.claritymed/models/vision/skin_cancer_dermoscopy/skin_isic_resnet50_v1/``.
4. ``tests/fixtures/vision/skin/`` contains at least one real
   dermoscopy image (placeholders filtered out).
5. ``configs/vision.yaml`` has ``skin_cancer_dermoscopy`` flipped to
   ``enabled: true`` after the operator promotes weights.
6. A reachable LLM provider (shared ``e2e_provider_id`` fixture).
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
    "Here is a dermoscopy image of a mole on my arm. Could this lesion "
    "be concerning? Please call the detect_disease_from_image tool with "
    "disease_id=skin_cancer_dermoscopy on this image — do not answer "
    "in free text. [Image sha:{sha8}]"
)


@pytest.fixture(scope="module", autouse=True)
def _require_servers() -> None:
    require_vision_servers()


@pytest.fixture(scope="module", autouse=True)
def _require_promoted_skin_model() -> None:
    require_promoted_artifact("skin_cancer_dermoscopy", "skin_isic_resnet50_v1")


@pytest.fixture(scope="module")
def _skin_fixture_path() -> Path:
    return pick_fixture_or_skip("skin")


@pytest.fixture
def _ctx():
    tokens = apply_context("20260616e2eskin000000000", USER_ID, "en")
    yield
    reset_context(tokens)


async def test_skin_full_round_trip(
    e2e_provider_id: str,
    _skin_fixture_path: Path,
    _ctx,
) -> None:
    """LLM picks the tool → confirm answers yes → server returns → reply branches."""
    await assert_tool_invoked(
        provider_id=e2e_provider_id,
        fixture_path=_skin_fixture_path,
        modality="dermoscopy",
        prompt=_PROMPT,
        label="skin",
    )
