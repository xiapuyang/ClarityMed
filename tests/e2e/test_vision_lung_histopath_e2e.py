"""Live full-stack lung-histopath e2e — vision-server + medical-clip-server + real LLM.

Mirror of ``test_vision_skin_e2e.py`` for the
``lung_cancer_histopathology`` disease. Skips cleanly when any of these
gates fail:

1. ``http://127.0.0.1:8085/health`` reachable with non-empty
   ``models_loaded``.
2. ``http://127.0.0.1:8086/health`` reachable.
3. A promoted lung-histopath artifact at
   ``~/.claritymed/models/vision/lung_cancer_histopathology/lung_histopath_resnet50_v1/``.
4. ``tests/fixtures/vision/histopath/`` contains at least one real
   histopathology image (placeholders filtered out).
5. ``configs/vision.yaml`` has ``lung_cancer_histopathology`` flipped to
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
    "Here's a lung histopathology slide from a recent biopsy. Could "
    "this tissue be concerning? Please call the "
    "detect_disease_from_image tool with "
    "disease_id=lung_cancer_histopathology on this image — do not "
    "answer in free text. [Image sha:{sha8}]"
)


@pytest.fixture(scope="module", autouse=True)
def _require_servers() -> None:
    require_vision_servers()


@pytest.fixture(scope="module", autouse=True)
def _require_promoted_lung_histopath_model() -> None:
    require_promoted_artifact(
        "lung_cancer_histopathology", "lung_histopath_resnet50_v1"
    )


@pytest.fixture(scope="module")
def _histopath_fixture_path() -> Path:
    return pick_fixture_or_skip("histopath")


@pytest.fixture
def _ctx():
    tokens = apply_context("20260616e2elunghistop0000", USER_ID, "en")
    yield
    reset_context(tokens)


async def test_lung_histopath_full_round_trip(
    e2e_provider_id: str,
    _histopath_fixture_path: Path,
    _ctx,
) -> None:
    """LLM picks the tool → confirm answers yes → server returns → reply branches."""
    await assert_tool_invoked(
        provider_id=e2e_provider_id,
        fixture_path=_histopath_fixture_path,
        modality="histopathology",
        prompt=_PROMPT,
        label="lung_histopath",
    )
