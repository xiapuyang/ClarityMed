"""Phase 3 tests: extractor, critical short-circuit composer, AskService
critical-path streamer, dynamic [SAFETY CONTEXT] system prompt.

These tests pin the Phase 3 wiring without hitting a real LLM:

* ``LLMExtractor`` / ``LLMComposer`` / ``CriticalReplyComposer`` are
  exercised with TestModel from ``pydantic_ai.models.test`` so the
  pipeline shape is validated end-to-end (prompt loaded, structured
  output produced, Agent constructed) without provider dependencies.
* ``AskService._stream_critical_short_circuit`` is tested directly
  against a manufactured ``EmergencyAssessment`` so the streamer's
  event sequence + result-dict population is locked.
* The ``_triage_system_prompt`` callable is a pure function and gets
  golden-string assertions for each level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from claritymed.core.emergency import (
    EmergencyAssessment,
    ExtractedSymptoms,
    LLMComposer,
    LLMExtractor,
    MatchedRule,
)
from claritymed.core.emergency.critical_reply import (
    CriticalReplyComposer,
    CriticalReplyResult,
    build_default_critical_reply,
)
from claritymed.core.emergency.extractor import (
    _format_history_for_extractor,
    build_default_extractor,
)
from claritymed.core.emergency.composer import build_default_composer


# --- prompt loading ---------------------------------------------------


def test_emergency_extractor_prompt_loads_both_languages():
    from claritymed.core.prompts.registry import PromptRegistry

    reg = PromptRegistry()
    en = reg.get("emergency_extractor", language="en")
    zh = reg.get("emergency_extractor", language="zh")
    assert "primary_complaint" in en
    assert "primary_complaint" in zh
    # Canonical vocabulary must be referenced.
    assert "radiation_left_arm" in en
    assert "radiation_left_arm" in zh


def test_emergency_reply_prompt_loads_both_languages():
    from claritymed.core.prompts.registry import PromptRegistry

    reg = PromptRegistry()
    en = reg.get("emergency_reply", language="en")
    zh = reg.get("emergency_reply", language="zh")
    assert "matched_rules" in en and "matched_rules" in zh
    # Prompt must explicitly forbid the boilerplate that defeats critical messaging.
    assert "I am an AI" in en or "AI" in en
    assert "AI" in zh


# --- history formatting ----------------------------------------------


def test_format_history_no_history():
    text = _format_history_for_extractor("我头疼", None)
    assert text == "user: 我头疼"


def test_format_history_with_prior_turns():
    """Verify both helpers handle real ModelRequest/ModelResponse shapes."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    history = [
        ModelRequest(parts=[UserPromptPart(content="I have a headache")]),
        ModelResponse(parts=[TextPart(content="How long has it lasted?")]),
    ]
    text = _format_history_for_extractor("It's been 2 hours", history)
    assert "user: I have a headache" in text
    assert "assistant: How long has it lasted?" in text
    assert text.endswith("user: It's been 2 hours")


def test_format_history_skips_partless_messages():
    """Messages without ``parts`` (unexpected shapes) are silently skipped."""

    class _Empty:
        pass

    text = _format_history_for_extractor("hi", [_Empty()])
    assert text == "user: hi"


def test_format_history_skips_non_string_parts():
    """Tool-call / structured parts that are not plain strings are skipped."""

    @dataclass
    class _Part:
        content: Any

    @dataclass
    class _Msg:
        parts: list

    msg = _Msg(parts=[_Part(content={"tool": "call"}), _Part(content="real text")])
    text = _format_history_for_extractor("q", [msg])
    # The non-string part is filtered; the string part comes through with
    # role "system" (neither ModelRequest nor ModelResponse class name).
    assert "system: real text" in text


# --- build_default_extractor under unhealthy local provider --------


# --- extractor / composer / critical_reply factory: no local provider -


def test_build_default_extractor_returns_none_when_no_local(monkeypatch):
    """No ``kind: local`` provider in models.yaml → factory returns None."""
    from claritymed.core.emergency import _provider as provider_mod
    from claritymed.core.schemas import ModelsConfig, ProviderConfig

    def fake_load_models():
        return ModelsConfig(
            providers=[
                ProviderConfig(id="openai", kind="cloud", model="openai:gpt-4o"),
            ],
            default_provider="openai",
        )

    monkeypatch.setattr(
        provider_mod, "build_local_gate_model", provider_mod.build_local_gate_model
    )
    # patch the load_models the factory imports
    import claritymed.stores.models as models_mod

    monkeypatch.setattr(models_mod, "load_models", fake_load_models)
    assert build_default_extractor() is None
    assert build_default_composer() is None
    assert build_default_critical_reply() is None


def test_build_default_extractor_returns_instance_when_local(monkeypatch):
    """A ``kind: local`` provider without ``api_key_env`` → factory wires it."""
    from claritymed.core.schemas import ModelsConfig, ProviderConfig

    def fake_load_models():
        return ModelsConfig(
            providers=[
                ProviderConfig(
                    id="ollama",
                    kind="local",
                    model="qwen3:14b",
                    base_url="http://127.0.0.1:11434/v1",
                ),
            ],
            default_provider="ollama",
        )

    import claritymed.stores.models as models_mod

    monkeypatch.setattr(models_mod, "load_models", fake_load_models)
    extractor = build_default_extractor()
    composer = build_default_composer()
    critical = build_default_critical_reply()
    assert extractor is not None
    assert composer is not None
    assert critical is not None


def test_build_default_extractor_skips_provider_with_unset_api_key_env(monkeypatch):
    """A local provider declaring api_key_env but with env unset is skipped."""
    from claritymed.core.schemas import ModelsConfig, ProviderConfig

    def fake_load_models():
        return ModelsConfig(
            providers=[
                ProviderConfig(
                    id="omlx",
                    kind="local",
                    model="some-model",
                    base_url="http://127.0.0.1:8000/v1",
                    api_key_env="OMLX_API_KEY",
                ),
            ],
            default_provider="omlx",
        )

    import claritymed.stores.models as models_mod

    monkeypatch.setattr(models_mod, "load_models", fake_load_models)
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    # All three factories should fail soft.
    assert build_default_extractor() is None
    assert build_default_composer() is None
    assert build_default_critical_reply() is None


# --- LLMExtractor / LLMComposer / CriticalReplyComposer with TestModel


@pytest.mark.asyncio
async def test_llm_extractor_runs_against_test_model():
    from pydantic_ai.models.test import TestModel

    extractor = LLMExtractor(TestModel(), language="en")
    out = await extractor.extract("I have severe chest pain", history=None)
    assert isinstance(out, ExtractedSymptoms)
    # TestModel emits canned data for every field — the exact value
    # doesn't matter, only that the structured-output path produces
    # a valid ExtractedSymptoms (proves pydantic-ai's output_type
    # contract is wired correctly).


@pytest.mark.asyncio
async def test_llm_composer_runs_against_test_model():
    from pydantic_ai.models.test import TestModel

    composer = LLMComposer(TestModel())
    text = await composer.compose(
        [
            MatchedRule(
                rule_id="acs",
                level="critical",
                suggested_action_i18n_key="emergency.action.call_ems_cardiac",
                citations=["AHA/ACC 2021"],
                matched_qualifiers=["radiation_left_arm"],
            ),
        ],
        ExtractedSymptoms(primary_complaint="chest_pain"),
        language="en",
    )
    assert isinstance(text, str)


@pytest.mark.asyncio
async def test_critical_reply_composer_returns_messages_json():
    from pydantic_ai.models.test import TestModel

    composer = CriticalReplyComposer(TestModel())
    result = await composer.compose(
        [
            MatchedRule(
                rule_id="acs",
                level="critical",
                suggested_action_i18n_key="emergency.action.call_ems_cardiac",
                citations=["AHA/ACC 2021"],
                matched_qualifiers=["radiation_left_arm"],
            ),
        ],
        ExtractedSymptoms(primary_complaint="chest_pain"),
        language="en",
    )
    assert isinstance(result, CriticalReplyResult)
    assert isinstance(result.text, str)
    assert isinstance(result.messages_json, bytes)
    assert result.messages_json  # non-empty


# --- _triage_system_prompt dynamic prompt -----------------------------


@dataclass
class _FakeDeps:
    triage: Any
    language: str = "en"


@dataclass
class _FakeCtx:
    deps: Any


def _make_triage(
    level,
    action_key="emergency.action.call_ems_cardiac",
    rules=None,
    missing=None,
    reasoning="",
):
    return EmergencyAssessment(
        level=level,
        matched_rules=rules
        or [
            MatchedRule(
                rule_id="acs",
                level=level if level != "routine" else "routine",
                suggested_action_i18n_key=action_key,
                citations=["x"],
                matched_qualifiers=[],
            )
        ],
        suggested_action_i18n_key=action_key,
        missing_qualifiers=missing or [],
        reasoning=reasoning,
    )


def test_triage_system_prompt_empty_when_no_triage():
    from claritymed.orchestrator.services.ask_service import _triage_system_prompt

    out = _triage_system_prompt(_FakeCtx(deps=_FakeDeps(triage=None)))
    assert out == ""


def test_triage_system_prompt_empty_when_routine():
    from claritymed.orchestrator.services.ask_service import _triage_system_prompt

    triage = EmergencyAssessment.routine_noop()
    out = _triage_system_prompt(_FakeCtx(deps=_FakeDeps(triage=triage)))
    assert out == ""


def test_triage_system_prompt_empty_when_critical():
    """Defensive: critical never reaches here (short-circuit fires first)."""
    from claritymed.orchestrator.services.ask_service import _triage_system_prompt

    triage = _make_triage("critical")
    out = _triage_system_prompt(_FakeCtx(deps=_FakeDeps(triage=triage)))
    assert out == ""


def test_triage_system_prompt_urgent_injects_safety_context():
    from claritymed.orchestrator.services.ask_service import _triage_system_prompt

    triage = _make_triage(
        "urgent",
        action_key="emergency.action.urgent_eval_chest_pain",
        missing=["radiation", "diaphoresis"],
        reasoning="chest pain pattern needs work-up",
    )
    out = _triage_system_prompt(_FakeCtx(deps=_FakeDeps(triage=triage)))
    assert "[SAFETY CONTEXT]" in out
    assert "level: urgent" in out
    assert "Chest pain needs evaluation today" in out
    assert "radiation, diaphoresis" in out
    assert "chest pain pattern needs work-up" in out


def test_triage_system_prompt_skips_missing_action_key():
    """A triage without an action key still emits level + reasoning."""
    from claritymed.orchestrator.services.ask_service import _triage_system_prompt

    triage = EmergencyAssessment(
        level="urgent",
        matched_rules=[
            MatchedRule(
                rule_id="x",
                level="urgent",
                suggested_action_i18n_key="not.a.real.key",
                citations=["c"],
            )
        ],
        suggested_action_i18n_key=None,
        reasoning="something",
    )
    out = _triage_system_prompt(_FakeCtx(deps=_FakeDeps(triage=triage)))
    assert "[SAFETY CONTEXT]" in out
    assert "level: urgent" in out


# --- AskService._stream_critical_short_circuit -----------------------


class _StubCriticalReply:
    def __init__(self, text="Stub supporting prose."):
        self._text = text

    async def compose(self, matched_rules, symptoms, *, language):
        return CriticalReplyResult(
            text=self._text,
            messages_json=b'[{"stub": true}]',
            usage=None,
        )


def _make_service(critical_reply=None):
    """Construct a minimally-initialized AskService bypassing __init__.

    The short-circuit is a self-contained method — bypass __init__'s
    heavy plugin / model wiring so the test focuses on the streamer
    contract rather than the surrounding service.
    """
    from claritymed.orchestrator.services.ask_service import AskService

    svc = AskService.__new__(AskService)
    svc._language = "en"
    svc._model_name = "test-model"
    svc._provider_id = "test"
    svc._critical_reply = critical_reply
    return svc


@pytest.fixture
def _stub_audit(monkeypatch):
    """Replace audit_event in ask_service with a list-collector.

    AskService.* methods that call audit_event would otherwise raise
    MissingContextError when no request/user/language ContextVars are
    set (unit tests don't bootstrap the FastAPI/CLI middleware).
    """
    from claritymed.orchestrator.services import ask_service as svc_mod

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        svc_mod,
        "audit_event",
        lambda kind, payload=None: calls.append((kind, payload or {})),
    )
    return calls


@pytest.mark.asyncio
async def test_critical_short_circuit_with_composer(_stub_audit):
    from claritymed.core.events import Done
    from claritymed.orchestrator.agents.ask_deps import AskDeps

    svc = _make_service(critical_reply=_StubCriticalReply("This sounds frightening."))
    triage = _make_triage("critical")
    deps = AskDeps(language="en", triage=triage, effective_sensitivity="balanced")
    result: dict = {
        "final_text": "",
        "messages_json": None,
        "usage": None,
        "latency": None,
        "steps": [],
        "had_error": False,
    }

    events = []
    async for ev in svc._stream_critical_short_circuit(deps, "test", result):
        events.append(ev)

    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "LlmCallStarted"
    assert "LlmFirstToken" in kinds
    assert "TokenChunk" in kinds
    assert isinstance(events[-1], Done)

    assert "Call your local emergency number" in result["final_text"]
    assert "This sounds frightening." in result["final_text"]
    assert result["messages_json"] == b'[{"stub": true}]'
    assert result["latency"] is not None


@pytest.mark.asyncio
async def test_critical_short_circuit_without_composer_uses_action_only(_stub_audit):
    from claritymed.orchestrator.agents.ask_deps import AskDeps

    svc = _make_service(critical_reply=None)
    triage = _make_triage("critical")
    deps = AskDeps(language="en", triage=triage, effective_sensitivity="balanced")
    result: dict = {
        "final_text": "",
        "messages_json": None,
        "usage": None,
        "latency": None,
        "steps": [],
        "had_error": False,
    }

    async for _ in svc._stream_critical_short_circuit(deps, "test", result):
        pass

    assert "Call your local emergency number" in result["final_text"]
    # No composer means no supporting paragraph appended.
    assert result["messages_json"] == b""


@pytest.mark.asyncio
async def test_critical_short_circuit_composer_failure_falls_through(_stub_audit):
    """Composer raising should not break the short-circuit."""
    from claritymed.orchestrator.agents.ask_deps import AskDeps

    class _BoomComposer:
        async def compose(self, *_a, **_kw):
            raise RuntimeError("model down")

    svc = _make_service(critical_reply=_BoomComposer())
    triage = _make_triage("critical")
    deps = AskDeps(language="en", triage=triage, effective_sensitivity="balanced")
    result: dict = {
        "final_text": "",
        "messages_json": None,
        "usage": None,
        "latency": None,
        "steps": [],
        "had_error": False,
    }

    async for _ in svc._stream_critical_short_circuit(deps, "test", result):
        pass

    # Action text still present even when composer fails.
    assert "Call your local emergency number" in result["final_text"]


@pytest.mark.asyncio
async def test_critical_short_circuit_emits_audit_event(monkeypatch):
    from claritymed.orchestrator.agents.ask_deps import AskDeps
    from claritymed.orchestrator.services import ask_service as svc_mod

    calls: list[tuple[str, dict]] = []

    def fake_audit(kind, payload=None):
        calls.append((kind, payload or {}))

    monkeypatch.setattr(svc_mod, "audit_event", fake_audit)
    svc = _make_service(critical_reply=None)
    triage = _make_triage("critical")
    deps = AskDeps(language="en", triage=triage, effective_sensitivity="balanced")
    result: dict = {
        "final_text": "",
        "messages_json": None,
        "usage": None,
        "latency": None,
        "steps": [],
        "had_error": False,
    }

    async for _ in svc._stream_critical_short_circuit(deps, "user-1", result):
        pass

    assert any(c[0] == "redflag.critical_short_circuit" for c in calls)
    payload = next(c[1] for c in calls if c[0] == "redflag.critical_short_circuit")
    assert payload["user_id"] == "user-1"
    assert payload["rule_ids"] == ["acs"]
    assert payload["had_composer"] is False
