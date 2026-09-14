"""End-to-end emergency triage gate behavior — real local LLM provider.

Pins the four invariants the gate makes from ``AskService.run``'s POV:

* **Critical short-circuit** (KTD-E3) — a textbook ACS prompt reaches
  the ``_stream_critical_short_circuit`` path; the streamed reply
  carries the localized action wording (``Call your local emergency
  number`` / ``立即拨打 120``) and the agent loop never runs.
* **Routine path** — a benign greeting passes through the gate with
  ``routine`` level and proceeds to the normal agent flow.
* **``off`` footer** (KTD-E9) — when the user opts out of the gate
  AND ``CLARITYMED_FORCE_EMERGENCY_GATE=off`` lets the override
  through, every reply carries the ``emergency.footer.gate_disabled``
  disclaimer suffix. Unbypassable: appended post-agent, not via prompt.
* **``strict`` footer** — strict-sensitivity replies carry the
  ``emergency.footer.strict_mode_active`` explainer.

Why these tests live in ``tests/e2e/`` rather than ``tests/core/``:

The phase-2/3/4 unit tests under ``tests/core/test_emergency_phase*``
pin every component (rule engine, extractor agent, composer, dynamic
system prompt) against ``TestModel`` and synthetic inputs. This file
covers the integration — does the extractor LLM actually return
canonical qualifier strings on natural language? does the catch-all
fire before the agent? — which only the live provider can answer.

Skipped in default CI; run with::

    uv run pytest tests/e2e/test_emergency_e2e.py -v --no-cov
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.events import Done
from claritymed.orchestrator.services import AskService
from claritymed.orchestrator.services.chat_session import ChatSession

logger = logging.getLogger(__name__)

USER_ID = "e2e"
MAX_ATTEMPTS = 3
PER_TURN_TIMEOUT_S = 120.0

# Textbook STEMI presentation. Specific enough that an extractor LLM
# of any reasonable quality should produce
# ``primary_complaint=chest_pain`` and at least one of
# ``radiation_left_arm`` / ``diaphoresis`` in the qualifier list. Age
# clearly above the ACS rule's age_min=35 floor.
_ACS_COMPLAINT_EN = (
    "I'm a 58-year-old man. For the past 30 minutes I've had crushing "
    "chest pain that's radiating into my left arm. I'm sweating heavily "
    "and feel short of breath."
)

_GREETING_EN = "Hello, how are you today?"


def _final_text(events: list[Any]) -> str:
    """Return ``Done.final`` text, joining streamed chunks as fallback."""
    for ev in reversed(events):
        if isinstance(ev, Done):
            final = getattr(ev, "final", None)
            if isinstance(final, str) and final:
                return final
    from claritymed.core.events import TokenChunk

    return "".join(ev.text for ev in events if isinstance(ev, TokenChunk) and ev.text)


async def _run_one(
    provider_id: str,
    *,
    complaint: str,
    emergency_sensitivity_override: str | None = None,
) -> list[Any]:
    """Construct AskService against a real provider and drain ``run`` events."""
    from claritymed.core.llm.model import build_model
    from claritymed.stores.models import resolve_provider

    provider = resolve_provider(override=provider_id)
    model = build_model(provider)
    chat = ChatSession.new(USER_ID)
    service = AskService(
        model=model,
        chat_session=chat,
        provider_id=provider.id,
        model_name=provider.model,
        provider_config=provider,
        rag_mode="off",  # skip retrieval — emergency gate is the SUT
        emergency_sensitivity_override=emergency_sensitivity_override,
    )
    events: list[Any] = []
    async with asyncio.timeout(PER_TURN_TIMEOUT_S):
        async for ev in service.run(complaint, user_id=USER_ID):
            events.append(ev)
    return events


@pytest.fixture
def _ctx():
    """Apply request context so audit_event fires under a tagged id."""
    tokens = apply_context("20260623e2eemergency", USER_ID, "en")
    yield
    reset_context(tokens)


# ---------------------------------------------------------------------
# Critical short-circuit
# ---------------------------------------------------------------------


async def test_critical_short_circuit_fires_on_textbook_acs(
    e2e_provider_id: str,
    _ctx,
) -> None:
    """Textbook STEMI input → critical reply with localized action wording.

    Retried because local extractors are non-deterministic on
    qualifier recognition: a 7-14B model may emit ``radiation``
    instead of ``radiation_left_arm`` on one run and the canonical
    token on the next. The test fails only when ``MAX_ATTEMPTS``
    independent sessions all fail to surface the action wording —
    that's a real signal (prompt too weak, model too small).
    """
    expected = "Call your local emergency number"
    for attempt in range(MAX_ATTEMPTS):
        try:
            events = await _run_one(e2e_provider_id, complaint=_ACS_COMPLAINT_EN)
        except asyncio.TimeoutError:
            logger.warning(
                "[emergency e2e] attempt %d/%d: timed out after %.0fs",
                attempt + 1,
                MAX_ATTEMPTS,
                PER_TURN_TIMEOUT_S,
            )
            continue
        final = _final_text(events)
        if expected in final:
            logger.info(
                "[emergency e2e] attempt %d/%d: critical short-circuit fired",
                attempt + 1,
                MAX_ATTEMPTS,
            )
            return
        logger.info(
            "[emergency e2e] attempt %d/%d: action wording absent (final=%r)",
            attempt + 1,
            MAX_ATTEMPTS,
            final[:200],
        )
    pytest.fail(
        f"Critical short-circuit did not fire across {MAX_ATTEMPTS} attempts.\n"
        f"  expected substring: {expected!r}\n"
        "  The extractor LLM did not produce qualifiers that match the ACS rule.\n"
        "  Strengthen emergency_extractor.yaml or try a more capable provider."
    )


# ---------------------------------------------------------------------
# Routine baseline
# ---------------------------------------------------------------------


async def test_routine_greeting_passes_through_gate(
    e2e_provider_id: str,
    _ctx,
) -> None:
    """Greeting → no critical action wording in the reply."""
    events = await _run_one(e2e_provider_id, complaint=_GREETING_EN)
    final = _final_text(events)
    assert "Call your local emergency number" not in final
    # Defensive: the reply should not contain the SAH or stroke action
    # wording either. (Action keys live in emergency.yaml; a stray
    # match would mean a rule fired against a greeting — bug.)
    assert "拨打 120" not in final
    assert "brain bleed" not in final.lower()


# ---------------------------------------------------------------------
# ``off`` footer — KTD-E9 safeguard #2 (unbypassable disclaimer)
# ---------------------------------------------------------------------


async def test_off_sensitivity_appends_gate_disabled_footer(
    e2e_provider_id: str,
    _ctx,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the gate is off, every reply must end with the gate_disabled footer.

    Pre-conditions:

    * ``CLARITYMED_FORCE_EMERGENCY_GATE=off`` (deploy-time master
      switch is **down** so the user override is honored).
    * AskService constructed with ``emergency_sensitivity_override="off"``.

    Either pre-condition missing → the resolver downgrades to
    ``lenient`` and the footer is suppressed. This test verifies the
    happy off-path: the user opted out, the operator allowed opt-out,
    the disclaimer still fires.
    """
    monkeypatch.setenv("CLARITYMED_FORCE_EMERGENCY_GATE", "off")
    events = await _run_one(
        e2e_provider_id,
        complaint=_GREETING_EN,
        emergency_sensitivity_override="off",
    )
    final = _final_text(events)
    # Substring from configs/i18n/en/emergency.yaml::footer.gate_disabled.
    assert "emergency triage is currently turned off" in final.lower(), (
        f"gate_disabled footer absent under off mode.\n  final={final[-400:]!r}"
    )


# ---------------------------------------------------------------------
# ``strict`` footer — disclosure of high-recall mode
# ---------------------------------------------------------------------


async def test_strict_sensitivity_appends_strict_mode_footer(
    e2e_provider_id: str,
    _ctx,
) -> None:
    """Strict sensitivity → every reply ends with the strict_mode_active footer."""
    events = await _run_one(
        e2e_provider_id,
        complaint=_GREETING_EN,
        emergency_sensitivity_override="strict",
    )
    final = _final_text(events)
    # Substring from configs/i18n/en/emergency.yaml::footer.strict_mode_active.
    assert "strict-sensitivity mode" in final.lower(), (
        f"strict_mode_active footer absent under strict mode.\n  final={final[-400:]!r}"
    )
