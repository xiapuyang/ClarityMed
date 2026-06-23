"""``EmergencyTriage`` — pre-step service facade.

Wired into :class:`~claritymed.orchestrator.services.ask_service.AskService`
between message-history retrieval and ``Agent.run``. Always invoked on
every clinical turn, gated only by the per-user / CLI sensitivity
profile.

Phase 1 ships the facade + the ``sensitivity == 'off'`` short-circuit
path. The extractor + rule engine + composer wire in Phases 2-3. Until
then, non-``off`` calls return :meth:`EmergencyAssessment.routine_noop`
so the call site is exercised end-to-end without false positives.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from claritymed.core.emergency.schemas import (
    EmergencyAssessment,
    SensitivityName,
)
from claritymed.core.observability.audit import audit_event

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EmergencyTriage:
    """Pre-step triage facade.

    Construction is cheap (no model load, no file read). The expensive
    components (extractor LLM agent, composer LLM agent, rule pack) are
    injected by the factory in Phase 3 to keep ``__init__`` reachable
    from unit tests without provider setup.
    """

    def __init__(
        self,
        *,
        rules: list[Any] | None = None,
        extractor: Any | None = None,
        composer: Any | None = None,
    ) -> None:
        self._rules = rules or []
        self._extractor = extractor
        self._composer = composer

    async def assess(
        self,
        query: str,
        history: list[Any] | None,
        *,
        sensitivity: SensitivityName,
    ) -> EmergencyAssessment:
        """Run the gate. Always returns an assessment — never raises.

        Phase 1 behavior:

        * ``sensitivity == 'off'`` → emit ``redflag.gate_disabled`` and
          return ``routine_noop`` immediately. Operators get one audit
          line per off-path turn so post-hoc "who was unprotected" is
          a one-grep.
        * Any other sensitivity → return ``routine_noop``. The
          extractor / rule engine / composer land in Phases 2-3; this
          stub keeps the call site live so the disclaimer / footer
          plumbing can be exercised end-to-end now.
        """
        if sensitivity == "off":
            try:
                audit_event(
                    "redflag.gate_disabled",
                    payload={"requested": "off", "effective": "off"},
                )
            except Exception:  # noqa: BLE001
                # Gate downtime must never deny the user their answer.
                # Audit failure is logged loud and the path continues.
                logger.exception("redflag.gate_disabled audit emit failed")
            return EmergencyAssessment.routine_noop()
        # Phase 2-3 land extractor + rule_engine + composer here.
        # Until then the gate is a no-op so the wiring + disclaimer
        # paths can be exercised without false-positive alarms.
        del query, history  # not used yet — keep signature stable
        return EmergencyAssessment.routine_noop()
