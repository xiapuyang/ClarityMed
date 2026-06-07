"""Hybrid mode router: rules first, LLM fallback when ambiguous.

The deterministic path covers the common cases (slash commands, imperative
verbs, question patterns) at zero LLM cost. When rules are not confident
(``low_threshold <= conf < high_threshold``), a small local model is asked
to classify. Anything below ``low_threshold`` returns ``ambiguous`` so the
caller can surface a confirmation modal.

Boundaries use ``>=`` and ``<`` consistently so the bands never overlap.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from claritymed.core.schemas.modes import ModeName, ModesConfig

DecisionSource = Literal["rule", "llm", "explicit", "ambiguous"]


class RoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: ModeName | Literal["ambiguous"]
    confidence: float = Field(ge=0.0, le=1.0)
    source: DecisionSource
    reason: str = ""


class ModeRouter:
    """Classify user input into ingest / ask / rag."""

    def __init__(self, modes: ModesConfig) -> None:
        self._modes = modes
        rules = modes.router.rules
        self._question_patterns = [
            re.compile(p, re.IGNORECASE) for p in rules.question_patterns
        ]

    def classify_rule_only(
        self, user_input: str, has_attachment: bool = False
    ) -> RoutingDecision:
        """Synchronous rule-only pass. Returns the best-effort decision —
        never makes an LLM call. Used by Phase 1 callers that prefer
        determinism over coverage.
        """
        text = user_input.strip()
        rules = self._modes.router.rules

        # Explicit prefixes win — confidence 1.0
        for spec in rules.explicit_prefixes:
            prefix = spec.get("prefix")
            if not prefix:
                continue
            if text.lower().startswith(prefix.lower()):
                target = spec.get("target", "ask")
                if target == "derive":
                    # /mode <name> — parse remainder
                    rest = text[len(prefix) :].strip().split(maxsplit=1)
                    if rest and rest[0] in ("ingest", "ask", "rag"):
                        return RoutingDecision(
                            mode=rest[0],  # type: ignore[arg-type]
                            confidence=1.0,
                            source="explicit",
                            reason=f"prefix {prefix!r}",
                        )
                else:
                    return RoutingDecision(
                        mode=target,
                        confidence=1.0,
                        source="explicit",
                        reason=f"prefix {prefix!r}",
                    )

        # Imperative verbs + attachment → ingest
        lowered = text.lower()
        for verb in rules.imperative_verbs_ingest:
            if verb.lower() in lowered:
                conf = 0.95 if has_attachment else 0.85
                return RoutingDecision(
                    mode="ingest",
                    confidence=conf,
                    source="rule",
                    reason=f"verb {verb!r}",
                )

        # Imperative verbs → rag (library / reference)
        for verb in rules.imperative_verbs_rag:
            if verb.lower() in lowered:
                return RoutingDecision(
                    mode="rag",
                    confidence=0.9,
                    source="rule",
                    reason=f"verb {verb!r}",
                )

        # Question patterns → ask
        for pattern in self._question_patterns:
            if pattern.search(text):
                return RoutingDecision(
                    mode="ask",
                    confidence=0.92,
                    source="rule",
                    reason=f"pattern {pattern.pattern!r}",
                )

        return RoutingDecision(
            mode="ambiguous",
            confidence=0.0,
            source="ambiguous",
            reason="no rule matched",
        )

    async def classify(
        self,
        user_input: str,
        has_attachment: bool = False,
        llm_classify=None,
    ) -> RoutingDecision:
        """Hybrid path: rules first, then optional LLM fallback.

        ``llm_classify`` is an awaitable ``(text) -> RoutingDecision`` —
        injected by the caller so tests can swap a real local Qwen model
        for a stub. Returning ``None`` skips the LLM step entirely.
        """
        decision = self.classify_rule_only(user_input, has_attachment)

        if decision.confidence >= self._modes.router.high_threshold:
            return decision
        if decision.confidence < self._modes.router.low_threshold:
            # Rules failed — try LLM if available; else surface ambiguous.
            if llm_classify is None:
                return decision
            llm_decision = await llm_classify(user_input)
            if llm_decision.confidence < self._modes.router.low_threshold:
                return RoutingDecision(
                    mode="ambiguous",
                    confidence=llm_decision.confidence,
                    source="llm",
                    reason="LLM low-confidence",
                )
            return llm_decision

        # Mid-band: rules produced something but not confident enough.
        if llm_classify is None:
            return decision
        return await llm_classify(user_input)
