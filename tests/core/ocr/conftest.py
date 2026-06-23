"""Shared OCR-test fixtures.

The MineRU env gate (``CLARITYMED_ALLOW_MINERU``) is part of the v1 PHI
defense — provider construction outside of explicit env opt-in raises
``MinerUNotAllowed``. The existing OCR test suite legitimately
constructs the provider to test its API; opt them all in here once so
the per-test code stays unchanged. Tests that exercise the env gate
itself override this via ``monkeypatch.delenv``.
"""

from __future__ import annotations


import pytest


@pytest.fixture(autouse=True)
def _allow_mineru_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set ``CLARITYMED_ALLOW_MINERU=1`` for every test in this dir.

    Tests that need to verify the gate raises can ``monkeypatch.delenv``
    explicitly to override.
    """
    monkeypatch.setenv("CLARITYMED_ALLOW_MINERU", "1")


@pytest.fixture(autouse=True)
def _permissive_vision_image_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a wide-open :class:`ImageLimits` for every OCR test.

    The shared OCR suite seeds tiny synthetic blobs (e.g. ``b"binary"``)
    to exercise provider behavior without dragging in real fixtures.
    Production ``vision.image_limits`` rejects those at the byte floor,
    which is correct for runtime PHI safety but irrelevant to OCR-layer
    semantics. Dedicated coverage of the guard lives in
    ``tests/core/vision/test_image_guard.py``.
    """
    from claritymed.core.vision.image_guard import ImageLimits

    monkeypatch.setattr(
        "claritymed.config.vision_image_limits",
        lambda: ImageLimits(
            max_bytes=10**12,
            max_dimension=10**6,
            max_pixels=10**12,
            min_bytes=0,
            min_dimension=0,
            min_pixels=0,
        ),
    )
