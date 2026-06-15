"""Severity → tier mapping helpers used by the plugin + post_process audit.

Mirrors :class:`~claritymed.core.symptoms.schemas.SeverityTier`. The
mapping is the standard 5-tier compression:

* severity ``1`` → Critical
* severity ``2`` → Urgent
* severity ``3`` / ``4`` → Moderate
* severity ``5`` → Mild

The "rule out the bad thing" reflex (:func:`tier_for_differential`)
takes the most urgent severity present in the differential — a single
Critical row dominates the overall tier even when downstream rows are
Mild. The plugin uses this to pick the audit allow-list and to set
the LLM's opening-paragraph tone instructions.
"""

from __future__ import annotations

from typing import Iterable

from claritymed.core.symptoms.schemas import SeverityTier


def tier_for_severity(severity: int) -> SeverityTier:
    """Return the tier name for a 1-5 severity integer.

    Raises:
        ValueError: ``severity`` outside the 1-5 range. The 5-tier
            mapping is the only contract — surfacing a 0 or 6 here is
            a data bug (the corpus loader's ``strict_severity`` guard
            should have caught it earlier).
    """
    if severity == 1:
        return "Critical"
    if severity == 2:
        return "Urgent"
    if severity in (3, 4):
        return "Moderate"
    if severity == 5:
        return "Mild"
    raise ValueError(
        f"severity must be 1-5, got {severity!r}; "
        f"check the dataset's strict_severity load path."
    )


def tier_for_differential(severities: Iterable[int]) -> SeverityTier:
    """Return the most urgent tier present across ``severities``.

    Empty input maps to ``Mild`` — no signal means no urgency
    instruction, which is the safest default (the LLM still composes
    a normal answer from whatever RAG / free-text path it falls back
    to upstream).
    """
    sev_list = list(severities)
    if not sev_list:
        return "Mild"
    return tier_for_severity(min(sev_list))
