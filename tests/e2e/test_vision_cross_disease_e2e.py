"""KNOWN GAP canary — same-modality wrong-anatomy passes through end-to-end.

Companion to :mod:`test_vision_modality_gate_e2e`, which covers the
*different-modality* case (image tagged ``ct`` submitted under a
breast-ultrasound disease → :class:`ModalityMismatchResult`, no HTTP
fires).

This file covers the *same-modality, wrong-anatomy* permutation: pick
two diseases that share a modality (``lung_cancer_histopathology`` and
``colon_cancer_histopathology`` both accept ``histopathology``) and
submit a generic histopath fixture under the "wrong" disease_id. The
modality hard gate does not fire, the chosen model runs to completion,
and the system returns whatever verdict the model produces. There is
no anatomy / cross-disease check today.

When anatomy gating ships, update the assertion at the bottom of
:func:`test_lung_image_submitted_under_colon_disease_runs_inference`
to match the new short-circuit kind.

Skip gates mirror the other histopath e2e tests:

1. vision-server + medical-clip-server reachable.
2. Promoted artifact for the *target* disease (the one we route to).
3. Real histopath fixture present (placeholders skip).
4. Target disease flipped to ``enabled: true`` in ``configs/vision.yaml``.
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

# Route the histopath image to the COLON disease. The fixture directory
# is a generic ``histopath/`` bucket — we don't know whether the image
# is lung tissue or colon tissue. Either way, the colon model runs and
# returns a verdict without flagging the mismatch.
_PROMPT_LUNG_IMAGE_AS_COLON = (
    "Here's a histopathology slide from a recent biopsy. Please call "
    "the detect_disease_from_image tool with "
    "disease_id=colon_cancer_histopathology on this image — do not "
    "answer in free text. [Image sha:{sha8}]"
)


@pytest.fixture(scope="module", autouse=True)
def _require_servers() -> None:
    require_vision_servers()


@pytest.fixture(scope="module", autouse=True)
def _require_promoted_colon_histopath_model() -> None:
    # We route to the colon model, so it must be promoted. The "lung"
    # half is whatever the histopath fixture actually contains — no
    # separate artifact required.
    require_promoted_artifact(
        "colon_cancer_histopathology", "colon_histopath_resnet50_v1"
    )


@pytest.fixture(scope="module")
def _histopath_fixture_path() -> Path:
    return pick_fixture_or_skip("histopath")


@pytest.fixture
def _ctx():
    tokens = apply_context("20260616e2ecrossdisease0", USER_ID, "en")
    yield
    reset_context(tokens)


async def test_lung_image_submitted_under_colon_disease_runs_inference(
    e2e_provider_id: str,
    _histopath_fixture_path: Path,
    _ctx,
) -> None:
    """Cross-disease canary: confirm modal fires, model runs, no gate trips.

    The system has no anatomy check today, so this is structurally
    identical to ``test_colon_histopath_full_round_trip`` from the
    happy-path test. The value is the docstring + the file name: when
    a future PR adds anatomy gating, the maintainer updating this
    canary will see exactly what behavior change is expected.
    """
    await assert_tool_invoked(
        provider_id=e2e_provider_id,
        fixture_path=_histopath_fixture_path,
        modality="histopathology",
        prompt=_PROMPT_LUNG_IMAGE_AS_COLON,
        label="cross_disease_histopath",
    )
