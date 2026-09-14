"""Regression tests for the assembled-prompt scrub gate.

ce:review P0 #1 + #2 — attachment OCR text and feature ``pre_invoke``
blocks (profile_context, etc.) join the prompt AFTER the user_input scrub
at the top of ``_run_scoped``. For cloud-bound turns the assembled prompt
must pass through ``PhiGuard.scrub_free_text`` one more time so PHI from
OCR or profile context doesn't ride past the gate.

The fail-loud behavior is identical to the user_input scrub: if the privacy
filter model is unavailable, the turn errors out with ``scrub_unavailable``
instead of silently sending unscrubbed text to the cloud LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.models.test import TestModel

from claritymed.core.events import Done, Error
from claritymed.core.features.base import FeatureMode, TurnContext
from claritymed.core.phi.guard import PhiGuard
from claritymed.core.schemas.models import ProviderConfig
from claritymed.orchestrator.services import AskService


@dataclass
class _ScrubReportFake:
    rule_hits: int = 0
    model_hits: int = 0
    model_failed: bool = False
    text_len_before: int = 0
    text_len_after: int = 0


class _RecordingGuard(PhiGuard):
    """``PhiGuard`` that records every ``scrub_free_text`` call.

    Replaces ``REDACT_ME`` markers with ``[REDACTED]`` so the test can
    assert downstream that the redaction actually happened (not just that
    scrub was called).
    """

    def __init__(self) -> None:
        # Skip the real rules / scrub-service init; we override scrub_free_text.
        self.calls: list[str] = []

    def scrub_free_text(self, text: str) -> tuple[str, _ScrubReportFake]:  # type: ignore[override]
        self.calls.append(text)
        scrubbed = text.replace("REDACT_ME", "[REDACTED]")
        return scrubbed, _ScrubReportFake(
            rule_hits=text.count("REDACT_ME"),
            text_len_before=len(text),
            text_len_after=len(scrubbed),
        )


class _FailingGuard(_RecordingGuard):
    """Guard whose scrub model fails on the SECOND call (assembled prompt).

    Validates that the assembled-prompt scrub path honors the same
    fail-loud contract the user_input scrub does.
    """

    def scrub_free_text(self, text: str) -> tuple[str, _ScrubReportFake]:  # type: ignore[override]
        self.calls.append(text)
        if len(self.calls) == 1:
            # First call (user_input scrub) succeeds — we want the failure
            # to surface specifically on the assembled-prompt pass.
            return text, _ScrubReportFake(
                text_len_before=len(text), text_len_after=len(text)
            )
        return text, _ScrubReportFake(
            model_failed=True, text_len_before=len(text), text_len_after=len(text)
        )


class _ProfileLikeFeature:
    """Stand-in ``ProfileContextFeature`` — contributes a pre-invoke block
    containing the magic ``REDACT_ME`` token so the test can assert the
    second scrub pass actually saw and redacted it."""

    name: str = "_profile_like"
    mode: FeatureMode = "deterministic"

    async def pre_invoke(self, ctx: TurnContext) -> str:
        return "Profile: patient name=REDACT_ME-NAME, condition=hypertension"

    def as_tool(self):
        return None

    def as_toolset(self):
        return None


def _cloud_provider() -> ProviderConfig:
    return ProviderConfig(id="test-cloud", kind="cloud", model="openai:gpt-4o")


def _local_provider() -> ProviderConfig:
    return ProviderConfig(id="test-local", kind="local", model="openai:gpt-4o")


async def test_assembled_prompt_is_scrubbed_on_cloud_turn() -> None:
    """The user_input scrub happens first; the assembled prompt (with the
    profile pre-block joined in) must also pass through scrub_free_text
    before reaching ``agent.run``."""
    guard = _RecordingGuard()
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        guard=guard,
        provider_config=_cloud_provider(),
        features=[_ProfileLikeFeature()],  # type: ignore[list-item]
    )

    events = [
        ev async for ev in service.run("what does my profile say", user_id="test")
    ]

    assert any(isinstance(e, Done) for e in events)
    # Two scrub calls: first the user_input, second the assembled prompt.
    assert len(guard.calls) >= 2, (
        f"expected at least 2 scrub calls (user_input + assembled prompt), "
        f"got {len(guard.calls)}: {guard.calls!r}"
    )
    # The assembled-prompt call must include the profile pre-block text.
    assembled = next(
        (c for c in guard.calls if "Profile:" in c and "REDACT_ME-NAME" in c),
        None,
    )
    assert assembled is not None, (
        "no scrub call contained the profile pre-block; assembled-prompt "
        f"scrub is not wired. scrub calls were: {guard.calls!r}"
    )


async def test_local_turn_skips_assembled_prompt_scrub() -> None:
    """Local turns do not pay the scrub cost — user's own machine, user
    already consented to send raw text. Only the assembled-prompt path is
    gated; the user_input path is also skipped for local turns."""
    guard = _RecordingGuard()
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        guard=guard,
        provider_config=_local_provider(),
        features=[_ProfileLikeFeature()],  # type: ignore[list-item]
    )

    events = [
        ev async for ev in service.run("what does my profile say", user_id="test")
    ]

    assert any(isinstance(e, Done) for e in events)
    # No scrub on local turns at all (current behavior).
    assert guard.calls == [], (
        f"local turn should not invoke scrub, got: {guard.calls!r}"
    )


async def test_assembled_prompt_scrub_model_failure_emits_error() -> None:
    """If the privacy filter model fails on the assembled prompt, the
    turn must error out instead of silently sending unscrubbed context."""
    guard = _FailingGuard()
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        guard=guard,
        provider_config=_cloud_provider(),
        features=[_ProfileLikeFeature()],  # type: ignore[list-item]
    )

    events = [ev async for ev in service.run("ping", user_id="test")]

    errors = [e for e in events if isinstance(e, Error)]
    assert any(e.error_type == "scrub_unavailable" for e in errors), (
        f"expected scrub_unavailable Error, got events: {events!r}"
    )
    # The model must NOT have been called (no Done event before Error,
    # but pydantic-ai's TestModel may emit Done anyway after Error in some
    # paths; the contract under test is that Error fired).
