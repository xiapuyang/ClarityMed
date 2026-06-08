"""Verify the ``rag.enabled`` flag gates strategy construction in CLI.

``_maybe_build_strategy`` is the seam between the master switch in
``configs/retrieval.yaml`` and the ``AskService(strategy=...)`` kwarg.
These tests don't invoke the full ``ask`` command (that needs a live
LLM); they exercise the helper directly so the gating contract has a
unit-level guard against regressions.
"""

from __future__ import annotations

from unittest.mock import patch

import claritymed.cli.main as cli_main
from claritymed.core.rag import load_retrieval_config
from claritymed.core.rag.schemas import TermServiceConfig
from claritymed.core.rag.strategies.naive_hybrid import NaiveHybridStrategy


def test_maybe_build_strategy_returns_none_when_disabled():
    """``rag.enabled=false`` — no strategy built."""
    real = load_retrieval_config()
    disabled = real.model_copy(
        update={"rag": real.rag.model_copy(update={"enabled": False})}
    )
    with patch(
        "claritymed.core.rag.load_retrieval_config",
        return_value=disabled,
    ):
        assert cli_main._maybe_build_strategy() is None


def test_maybe_build_strategy_builds_when_enabled():
    """Flipping the flag returns a real NaiveHybridStrategy.

    Patches ``load_retrieval_config`` at the module path the helper
    imports it from. Swaps term_service to ``none`` so the test avoids
    the optional UMLS data dependency.
    """
    real = load_retrieval_config()
    none_entry = next(e for e in real.term_service.catalog if e.id == "none")
    enabled = real.model_copy(
        update={
            "rag": real.rag.model_copy(update={"enabled": True}),
            "term_service": TermServiceConfig(active="none", catalog=[none_entry]),
        }
    )
    with patch(
        "claritymed.core.rag.load_retrieval_config",
        return_value=enabled,
    ):
        strategy = cli_main._maybe_build_strategy()
    assert isinstance(strategy, NaiveHybridStrategy)
