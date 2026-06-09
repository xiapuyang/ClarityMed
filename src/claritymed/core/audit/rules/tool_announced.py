"""Rule: per-provider/model rate of "announced retrieval but skipped it".

Pairs ``mode.ask`` (denominator) with ``mode.ask.tool_announced_but_skipped``
(numerator) to produce a per-``provider_id/model`` skip rate. Findings
recommend a concrete action based on rate thresholds calibrated against
the prompt-v3 rollout:

* ``< 2%`` — prompt is doing its job, monitor only.
* ``2 - 5%`` — emit a "watch" finding; tune patterns or wait for more data.
* ``5 - 10%`` — recommend tightening the system prompt (v4).
* ``>= 10%`` — recommend switching that provider to ``rag.mode=deterministic``.

Sample snippets (up to 10) ride along so an operator can eyeball
matches and grow the announcement regex from real data.
"""

from __future__ import annotations

from claritymed.core.audit.rules.base import RuleReport


class ToolAnnouncedButSkippedRule:
    """Per-(provider_id, model) tool-skip rate."""

    name = "tool_announced_but_skipped"
    description = (
        "Rate at which each provider says it will retrieve but never fires the "
        "retrieve_medical_literature tool. Tracks mode.ask (denominator) vs "
        "mode.ask.tool_announced_but_skipped (numerator)."
    )

    # Severity thresholds (skip rate, percent). Tunable per project as
    # we gather more real-world data.
    _RECOMMEND_DETERMINISTIC = 10.0
    _RECOMMEND_PROMPT_TUNE = 5.0
    _WATCH = 2.0
    _MAX_SAMPLES = 10

    def __init__(self) -> None:
        self._totals: dict[tuple[str, str], int] = {}
        self._skipped: dict[tuple[str, str], int] = {}
        self._samples: list[dict] = []

    def accept(self, event: dict) -> None:
        kind = event.get("kind")
        payload = event.get("payload") or {}
        if kind == "mode.ask":
            key = (payload.get("provider_id", "?"), payload.get("model", "?"))
            self._totals[key] = self._totals.get(key, 0) + 1
        elif kind == "mode.ask.tool_announced_but_skipped":
            key = (payload.get("provider_id", "?"), payload.get("model", "?"))
            self._skipped[key] = self._skipped.get(key, 0) + 1
            if len(self._samples) < self._MAX_SAMPLES:
                self._samples.append(
                    {
                        "model": payload.get("model"),
                        "provider_id": payload.get("provider_id"),
                        "snippet": payload.get("snippet"),
                        "created_at": event.get("created_at"),
                    }
                )

    def report(self) -> RuleReport:
        counts: dict[str, int] = {}
        findings: list[str] = []
        total_relevant = sum(self._totals.values())

        # Sort by skip rate descending so the worst offenders sit at the
        # top of the human-readable report.
        rated: list[tuple[str, int, int, float]] = []
        for key, total in self._totals.items():
            skipped = self._skipped.get(key, 0)
            label = f"{key[0]}/{key[1]}"
            counts[label] = skipped
            rate = (skipped / total * 100.0) if total else 0.0
            rated.append((label, skipped, total, rate))
        rated.sort(key=lambda row: row[3], reverse=True)

        for label, skipped, total, rate in rated:
            if rate >= self._RECOMMEND_DETERMINISTIC:
                findings.append(
                    f"{label}: {skipped}/{total} ({rate:.1f}%) — "
                    f"switch to rag.mode=deterministic for this provider"
                )
            elif rate >= self._RECOMMEND_PROMPT_TUNE:
                findings.append(
                    f"{label}: {skipped}/{total} ({rate:.1f}%) — "
                    f"tighten prompt (consider v4 with stronger anti-narration)"
                )
            elif rate >= self._WATCH:
                findings.append(
                    f"{label}: {skipped}/{total} ({rate:.1f}%) — watch; "
                    f"may indicate model drift"
                )

        return RuleReport(
            name=self.name,
            description=self.description,
            total_relevant=total_relevant,
            counts=counts,
            findings=findings,
            samples=self._samples,
        )
