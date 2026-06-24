"""``EmergencyTriage`` — pre-step service facade.

Wired into :class:`~claritymed.orchestrator.services.ask_service.AskService`
between message-history retrieval and ``Agent.run``. Always invoked on
every clinical turn, gated only by the per-user / CLI sensitivity
profile.

Pipeline (when the gate is enabled):

    extractor(query, history) → ExtractedSymptoms
        primary_complaint is None? → routine_noop
    profile.apply_to(rules) → effective_rules
    rule_engine.match(symptoms, effective_rules) → matched_rules[]
        empty? → routine_noop with missing_qualifiers from
                 ambiguous-rule hints (Phase 5+)
    composer.compose(matched, symptoms) → reasoning
    return EmergencyAssessment(...)

Phase 2 ships the rule engine, profile application, and composer
plumbing. The extractor LLM lands in Phase 3 — until then ``assess()``
either returns ``routine_noop`` (when no extractor is wired) or runs
the post-extraction pipeline against a caller-supplied
:class:`ExtractedSymptoms` via :meth:`assess_from_symptoms` (used by
tests + the Phase 3 extractor).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from claritymed.core.emergency.composer import build_assessment
from claritymed.core.emergency.rule_engine import match as match_rules
from claritymed.core.emergency.schemas import (
    EmergencyAssessment,
    EmergencyLevel,
    ExtractedSymptoms,
    MatchedRule,
    SensitivityName,
)
from claritymed.core.observability.audit import audit_event

if TYPE_CHECKING:
    from claritymed.core.emergency.composer import Composer
    from claritymed.core.emergency.config import EmergencyConfig
    from claritymed.core.emergency.rules import Rule

logger = logging.getLogger(__name__)


class EmergencyTriage:
    """Pre-step triage facade.

    Construction is cheap (no model load, no file read). The expensive
    components (extractor LLM, composer LLM, rule pack + profile config)
    are injected so the same class powers both production (rules
    loaded from YAML, real LLMs) and tests (synthetic rules, stub
    composer).
    """

    def __init__(
        self,
        *,
        rules: list["Rule"] | None = None,
        config: "EmergencyConfig | None" = None,
        composer: "Composer | None" = None,
        extractor: Any | None = None,
    ) -> None:
        self._rules = rules or []
        self._config = config
        self._composer = composer
        # Extractor lands in Phase 3; held as Any so the import does
        # not pull pydantic-ai into Phase 2 test paths that don't need
        # an LLM.
        self._extractor = extractor

    async def assess(
        self,
        query: str,
        history: list[Any] | None,
        *,
        sensitivity: SensitivityName,
        language: str = "en",
    ) -> EmergencyAssessment:
        """Run the gate. Always returns an assessment — never raises.

        ``sensitivity == 'off'`` short-circuits and emits the
        ``redflag.gate_disabled`` audit event before returning
        ``routine_noop``. Other sensitivities require an extractor to
        produce :class:`ExtractedSymptoms`; when no extractor is wired
        (Phase 2 default), the call is a no-op routine — preserving
        the Phase 1 contract while Phase 3 wires the real extractor.
        """
        if sensitivity == "off":
            try:
                audit_event(
                    "redflag.gate_disabled",
                    payload={"requested": "off", "effective": "off"},
                )
            except Exception:  # noqa: BLE001
                logger.exception("redflag.gate_disabled audit emit failed")
            return EmergencyAssessment.routine_noop()

        if self._extractor is None:
            # Phase 2: extractor not yet wired. The rule engine +
            # composer plumbing is exercised by tests via
            # ``assess_from_symptoms``; production callers pass
            # through this branch as a no-op until Phase 3 lands.
            return EmergencyAssessment.routine_noop()

        try:
            symptoms: ExtractedSymptoms = await self._extractor.extract(query, history)
        except Exception:  # noqa: BLE001
            logger.exception("emergency extractor failed; falling open")
            return EmergencyAssessment.routine_noop()
        return await self.assess_from_symptoms(
            symptoms, sensitivity=sensitivity, language=language
        )

    async def assess_from_symptoms(
        self,
        symptoms: ExtractedSymptoms,
        *,
        sensitivity: SensitivityName,
        language: str = "en",
    ) -> EmergencyAssessment:
        """Run the post-extraction pipeline against ready symptoms.

        Used by:

        * Tests — bypass the extractor LLM with synthetic symptoms.
        * The Phase 3 extractor agent — once it produces symptoms
          inside :meth:`assess`, the rest of the pipeline funnels here.

        Pipeline:

        1. Non-clinical input (``primary_complaint is None``) →
           ``routine_noop``. The early exit keeps the gate's p95
           latency under the 150 ms budget for non-clinical turns
           the plan calls for.
        2. Resolve the effective rule list via the profile's
           ``apply_to`` (config-load validator has already enforced
           floor compliance).
        3. Run the deterministic matcher.
        4. No matches → ``routine_noop``.
        5. Compose the user-facing reasoning via the LLM composer
           (or skip + leave empty when no composer is wired — keeps
           the test surface small).
        """
        if sensitivity == "off":
            try:
                audit_event(
                    "redflag.gate_disabled",
                    payload={"requested": "off", "effective": "off"},
                )
            except Exception:  # noqa: BLE001
                logger.exception("redflag.gate_disabled audit emit failed")
            return EmergencyAssessment.routine_noop()

        if symptoms.primary_complaint is None and not symptoms.qualifiers:
            self._log_decision(
                symptoms, [], "routine", sensitivity, reason="non_clinical_input"
            )
            return EmergencyAssessment.routine_noop()

        effective_rules = self._effective_rules_for(sensitivity)
        matched = match_rules(symptoms, effective_rules)
        if not matched:
            self._log_decision(
                symptoms, [], "routine", sensitivity, reason="no_rules_matched"
            )
            return EmergencyAssessment.routine_noop()

        reasoning = ""
        if self._composer is not None:
            try:
                reasoning = await self._composer.compose(
                    matched, symptoms, language=language
                )
            except Exception:  # noqa: BLE001
                # Composer downtime should not deny the user the
                # action wording — the rule's i18n action key already
                # carries the load-bearing safety advice. Log loud
                # and continue with empty prose.
                logger.exception("emergency composer failed; emitting empty reasoning")

        assessment = build_assessment(matched, symptoms, reasoning)
        self._log_decision(symptoms, matched, assessment.level, sensitivity)
        return assessment

    # --- helpers ----------------------------------------------------

    def _log_decision(
        self,
        symptoms: ExtractedSymptoms,
        matched: list[MatchedRule],
        level: EmergencyLevel,
        sensitivity: SensitivityName,
        *,
        reason: str = "",
    ) -> None:
        """One INFO line per gate decision so request_id grep surfaces it.

        Without this, a successful gate run leaves no log trail at all
        — only failure paths (extractor/composer crash, off-mode audit,
        critical short-circuit audit) emit anything. An operator
        debugging "why didn't the gate fire on this clearly-critical
        prose" had no way to see what the extractor produced or which
        rules ran. This closes that gap with one structured line per
        ``assess_from_symptoms`` exit, covering routine_noop early-exits
        and real assessments alike.

        Off-mode short-circuit (returns at top of assess_from_symptoms)
        and extractor-failure (``assess`` exception branch) log via
        their own paths and are intentionally NOT covered here — they
        already emit something operators can grep.
        """
        suffix = f" reason={reason}" if reason else ""
        logger.info(
            "emergency gate: pc=%s qualifiers=%s matched=%s level=%s sensitivity=%s%s",
            symptoms.primary_complaint,
            list(symptoms.qualifiers),
            [m.rule_id for m in matched],
            level,
            sensitivity,
            suffix,
        )

    def _effective_rules_for(self, sensitivity: SensitivityName) -> list["Rule"]:
        """Apply the profile's overrides + ambiguous-filter to ``self._rules``.

        Returns a fresh list; never mutates the canonical rule pack.
        When no config is wired (Phase 2 tests pass rules directly),
        returns ``self._rules`` unchanged — the matcher then sees the
        rule defaults, which is the ``balanced`` profile behavior.
        """
        if self._config is None:
            return list(self._rules)
        profile = self._config.sensitivity_profiles.get(sensitivity)
        if profile is None:
            return list(self._rules)
        from claritymed.core.emergency.rules import apply_profile_to_rules

        return apply_profile_to_rules(self._rules, profile, sensitivity)
