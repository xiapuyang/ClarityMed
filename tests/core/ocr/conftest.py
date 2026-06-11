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
