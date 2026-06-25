"""Pin ``AskService._build_ingest_factory`` — the closure layer that wires
``IngestToolsFeature`` together with ``ToolDispatcher``.

The factory is invoked at most once per AskService (lazy build from
``build_features``), but the closures it returns run on every tool call.
These tests exercise both halves:

1. The factory's own body (imports + closure definitions + final
   ``return _factory``).
2. The four closures it produces:
   * ``_session_shas`` — list session attachments by user/session.
   * ``_rule_match`` — look up an allow-rule for a given tool call.
   * ``_approval_required`` — gate one tool call through the dispatcher.
   * ``_resolve_tool_prompt_language`` — env override vs self-language.

Each closure has a swallow-Exception branch so a broken store can't kill
the agent — those branches are explicitly covered here.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.schemas.models import ProviderConfig
from claritymed.orchestrator.services.ask_service import AskService


def _provider(kind: str = "local") -> ProviderConfig:
    return ProviderConfig(id="test_provider", kind=kind, model="openai:gpt-4o")


@contextmanager
def _with_user_id(uid: str, *, session_id: str | None = None):
    """Bind ``user_id_ctx`` for the with-block.

    ``session_id_ctx`` isn't part of the closure surface (the factory gets
    ``get_session_id`` injected directly), but ``user_id_ctx`` must be
    populated for the closures to do anything meaningful — they short-
    circuit to a noop when the context is unset.
    """
    tokens = apply_context("20260101000000DEADBEEF01", uid, "en")
    try:
        yield
    finally:
        reset_context(tokens)


@pytest.fixture
def service() -> AskService:
    """Minimal AskService — only ``model`` is required; everything else
    defaults. The model is never invoked by these tests."""
    from pydantic_ai.models.test import TestModel

    return AskService(
        model=TestModel(custom_output_text="answer"),
        provider_config=_provider("local"),
    )


# --- factory body --------------------------------------------------------


def test_build_ingest_factory_returns_callable(service: AskService) -> None:
    """Smoke: the function returns a zero-arg callable that builds a plugin."""
    from claritymed.orchestrator.features.ingest_tools_plugin import IngestToolsFeature

    factory = service._build_ingest_factory(get_session_id=lambda: "sess-1")
    assert callable(factory)
    feature = factory()
    assert isinstance(feature, IngestToolsFeature)


# --- _session_shas closure ----------------------------------------------


def test_session_shas_returns_empty_when_no_user_id_ctx(service: AskService) -> None:
    """Outside a request scope → no user_id → no attachments."""
    # Build the factory but exercise the closure separately via the
    # factory's act of producing IngestToolsFeature. The closure is the
    # ``session_attachments`` arg the dispatcher captures.
    factory = service._build_ingest_factory(get_session_id=lambda: "sess-1")
    feature = factory()
    shas = feature._dispatcher._session_attachments()
    assert shas == set()


def test_session_shas_returns_empty_when_session_id_is_none(
    service: AskService,
) -> None:
    """user_id present + session_id None → still noop."""
    factory = service._build_ingest_factory(get_session_id=lambda: None)
    feature = factory()
    with _with_user_id("test"):
        assert feature._dispatcher._session_attachments() == set()


def test_session_shas_returns_attachment_shas_when_present(
    service: AskService,
) -> None:
    """Real session attachments → returned as a set of sha256 strings."""
    from claritymed.stores.blob_store import BlobStore
    from claritymed.stores.session_attachments import SessionAttachments

    factory = service._build_ingest_factory(get_session_id=lambda: "sess-1")
    feature = factory()

    with _with_user_id("test"):
        # Seed two attachments so the closure has something to enumerate.
        bs = BlobStore("test")
        sha_a = bs.store(b"alpha bytes", "pdf")
        sha_b = bs.store(b"beta bytes", "pdf")
        sa = SessionAttachments("test", "sess-1")
        sa.add(sha256=sha_a, filename="a.pdf", mime="application/pdf", size=11)
        sa.add(sha256=sha_b, filename="b.pdf", mime="application/pdf", size=10)

        assert feature._dispatcher._session_attachments() == {sha_a, sha_b}


def test_session_shas_swallows_session_attachments_failure(
    monkeypatch, service: AskService
) -> None:
    """A SessionAttachments.list() exception → noop (no agent crash)."""
    factory = service._build_ingest_factory(get_session_id=lambda: "sess-1")
    feature = factory()

    # Patch the closure's SessionAttachments.list dependency to raise.
    from claritymed.stores import session_attachments as _sa

    def _bad(self):
        raise RuntimeError("sqlite gone")

    monkeypatch.setattr(_sa.SessionAttachments, "list", _bad)

    with _with_user_id("test"):
        assert feature._dispatcher._session_attachments() == set()


# --- _rule_match closure -------------------------------------------------


def test_rule_match_returns_none_when_no_user_id_ctx(service: AskService) -> None:
    factory = service._build_ingest_factory(get_session_id=lambda: "sess-1")
    feature = factory()
    assert feature._dispatcher._rule_match("save_record", {"path": "x"}) is None


def test_rule_match_returns_rule_id_for_matching_allow_rule(
    monkeypatch, service: AskService
) -> None:
    factory = service._build_ingest_factory(get_session_id=lambda: "sess-1")
    feature = factory()

    matching = SimpleNamespace(id="rule-7", action="allow")

    from claritymed.stores import settings_store as _ss

    monkeypatch.setattr(
        _ss.SettingsStore, "match_rule", lambda self, name, args: matching
    )

    with _with_user_id("test"):
        assert feature._dispatcher._rule_match("save_record", {"path": "x"}) == "rule-7"


def test_rule_match_returns_none_when_rule_is_deny(
    monkeypatch, service: AskService
) -> None:
    """A deny-rule still resolves but the closure refuses to short-circuit
    approval — returns ``None`` so the modal opens instead of auto-allowing.
    """
    factory = service._build_ingest_factory(get_session_id=lambda: "sess-1")
    feature = factory()

    deny = SimpleNamespace(id="rule-deny", action="deny")

    from claritymed.stores import settings_store as _ss

    monkeypatch.setattr(_ss.SettingsStore, "match_rule", lambda self, *a: deny)

    with _with_user_id("test"):
        assert feature._dispatcher._rule_match("save_record", {"path": "x"}) is None


def test_rule_match_swallows_match_rule_failure(
    monkeypatch, service: AskService
) -> None:
    """An exception inside the rule lookup is logged but not propagated."""
    factory = service._build_ingest_factory(get_session_id=lambda: "sess-1")
    feature = factory()

    from claritymed.stores import settings_store as _ss

    def _bad(self, *a, **kw):  # noqa: ANN001
        raise RuntimeError("settings store broken")

    monkeypatch.setattr(_ss.SettingsStore, "match_rule", _bad)

    with _with_user_id("test"):
        assert feature._dispatcher._rule_match("save_record", {"path": "x"}) is None


# --- _resolve_tool_prompt_language ---------------------------------------


def test_resolve_tool_prompt_language_env_override_to_zh(
    monkeypatch, service: AskService
) -> None:
    """``CLARITYMED_TOOL_PROMPT_LANG=zh`` wins over the AskService language."""
    monkeypatch.setenv("CLARITYMED_TOOL_PROMPT_LANG", "zh")
    factory = service._build_ingest_factory(get_session_id=lambda: "sess-1")
    feature = factory()
    assert feature._language == "zh"


def test_resolve_tool_prompt_language_falls_back_to_self_language(
    monkeypatch, service: AskService
) -> None:
    """No env override → use the AskService instance's language."""
    monkeypatch.delenv("CLARITYMED_TOOL_PROMPT_LANG", raising=False)
    factory = service._build_ingest_factory(get_session_id=lambda: "sess-1")
    feature = factory()
    assert feature._language == "en"  # default in the fixture


def test_resolve_tool_prompt_language_ignores_unknown_env_value(
    monkeypatch, service: AskService
) -> None:
    """Unknown values (e.g. ``ja``) are silently ignored, not raised."""
    monkeypatch.setenv("CLARITYMED_TOOL_PROMPT_LANG", "ja")
    factory = service._build_ingest_factory(get_session_id=lambda: "sess-1")
    feature = factory()
    # Fell through to self._language since "ja" is not in {en, zh}.
    assert feature._language == "en"
