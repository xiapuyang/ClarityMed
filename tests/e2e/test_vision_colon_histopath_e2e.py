"""Live full-stack colon-histopath e2e — vision-server + medical-clip-server + real LLM.

Sister of ``test_vision_lung_histopath_e2e.py`` for the
``colon_cancer_histopathology`` disease. Same skip gates — both rely on
the same vision-server + medical-clip-server processes and the same
``histopath`` fixture directory; only the disease_id and the prompt
text differ.
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
    "Here's a colon histopathology slide from a recent biopsy. Could "
    "this tissue be concerning? Please call the "
    "detect_disease_from_image tool with "
    "disease_id=colon_cancer_histopathology on this image — do not "
    "answer in free text. [Image sha:{sha8}]"
)


@pytest.fixture(scope="module", autouse=True)
def _require_servers() -> None:
    require_vision_servers()


@pytest.fixture(scope="module", autouse=True)
def _require_promoted_colon_histopath_model() -> None:
    require_promoted_artifact(
        "colon_cancer_histopathology", "colon_histopath_resnet50_v1"
    )


@pytest.fixture(scope="module")
def _histopath_fixture_path() -> Path:
    return pick_fixture_or_skip("histopath")


@pytest.fixture
def _ctx():
    tokens = apply_context("20260616e2ecolonhistop000", USER_ID, "en")
    yield
    reset_context(tokens)


async def test_colon_histopath_full_round_trip(
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
        label="colon_histopath",
    )
