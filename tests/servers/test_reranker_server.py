"""Unit tests for src/claritymed/servers/reranker.py helpers.

The reranker server module pulls torch / transformers / fastapi at
import time (it cannot start without them). When the ``rag-server``
extra is not installed those imports raise ``SystemExit`` from the
top-level guard, so this whole test file is skipped — same posture as
``servers/embedder.py``, which has no separate test file at all.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")

from claritymed.servers.reranker import _apply_query_instruction  # noqa: E402


def test_apply_query_instruction_empty_is_noop():
    """v2-m3 path: no prefix configured → query unchanged."""
    assert _apply_query_instruction("ferritin", "") == "ferritin"


def test_apply_query_instruction_prepends_prefix():
    """v2-gemma path: prefix configured → prepended verbatim to query."""
    assert _apply_query_instruction("ferritin", "Query: ") == "Query: ferritin"


def test_apply_query_instruction_does_not_strip_or_trim():
    """Whitespace in the prefix is the operator's choice; do not touch it."""
    assert _apply_query_instruction("q", "A:") == "A:q"
    assert _apply_query_instruction("q", "A: ") == "A: q"


def test_apply_query_instruction_handles_empty_query():
    """Empty query + prefix → just the prefix; do not raise."""
    assert _apply_query_instruction("", "Prefix: ") == "Prefix: "
