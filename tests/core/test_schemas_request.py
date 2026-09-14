"""Tests for ``RequestContext`` and ContextVar round-trip."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claritymed.context import (
    apply_context,
    new_request_id,
    request_id_ctx,
    reset_context,
)
from claritymed.core.schemas import RequestContext


def test_happy_construction():
    rc = RequestContext(
        request_id="20260606222522A1B2C3D4",
        user_id="alice",
        language="en",
    )
    assert rc.entry == "cli"


def test_old_short_hex_rejected():
    with pytest.raises(ValidationError):
        RequestContext(
            request_id="a1b2c3d4",
            user_id="alice",
            language="en",
        )


def test_lowercase_hex_in_suffix_rejected():
    with pytest.raises(ValidationError):
        RequestContext(
            request_id="20260606222522a1b2c3d4",
            user_id="alice",
            language="en",
        )


def test_user_id_path_traversal_rejected():
    with pytest.raises(ValidationError):
        RequestContext(
            request_id="20260606222522A1B2C3D4",
            user_id="../etc/passwd",
            language="en",
        )


def test_from_and_apply_to_context_round_trips():
    tokens = apply_context("20260606222522DEADBEEF", "test", "zh")
    try:
        rc = RequestContext.from_context_vars(entry="api")
        assert rc.user_id == "test"
        assert rc.language == "zh"
        assert rc.entry == "api"
    finally:
        reset_context(tokens)
    assert request_id_ctx.get() is None


def test_fuzz_new_request_id_matches_schema():
    """new_request_id() always produces a string that satisfies the schema."""
    for _ in range(100):
        rid = new_request_id()
        rc = RequestContext(request_id=rid, user_id="alice", language="en")
        assert rc.request_id == rid
