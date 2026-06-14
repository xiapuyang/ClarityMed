"""Eligibility strategies — pluggable filters that decide whether a
complaint is in-scope for a symptoms dataset before the sub-session runs.

Public surface re-exports the Protocol, result type, factory entry, and
the concrete strategies so plugin / test code never reaches into the
strategy modules directly.
"""

from __future__ import annotations

from claritymed.core.symptoms.eligibility.base import (
    EligibilityReason,
    EligibilityResult,
    EligibilityStrategy,
)
from claritymed.core.symptoms.eligibility.direct import DirectEligibility
from claritymed.core.symptoms.eligibility.factory import build_eligibility_strategy
from claritymed.core.symptoms.eligibility.term_service import TermServiceEligibility
from claritymed.core.symptoms.eligibility.translation import TranslationEligibility

__all__ = [
    "DirectEligibility",
    "EligibilityReason",
    "EligibilityResult",
    "EligibilityStrategy",
    "TermServiceEligibility",
    "TranslationEligibility",
    "build_eligibility_strategy",
]
