"""Server-side hydration for the multi-card renderer.

Takes the raw ``predict_disease_from_symptoms`` tool result (rows of
:class:`DifferentialRow`, session flags) and produces a fully
hydrated :class:`DifferentialReady` event the frontend renders
directly. Everything the client shows — confidence chip, directional
headline, curated report / suggestion, per-card citations, and the
above-cards banner key — is decided here, not by the LLM and not by
the frontend.

Why "not the LLM": per-condition medical text is exactly the class
of content that must not hallucinate. The static catalog wins on
safety, i18n, auditability, and local-small-model stability.

Why "not the frontend": banner selection depends on a session-flag
combination (``cancelled`` × ``meets_confidence_threshold`` ×
``severity_override`` × ``hit_cap``) that would otherwise be
duplicated in every UI surface. Server-computed ``banner_key`` keeps
the client to a single dict lookup.
"""

from __future__ import annotations

import logging
from typing import Iterable

from claritymed.core.events import (
    ConfidenceBucket,
    DifferentialCard,
    DifferentialReady,
    DifferentialSessionMeta,
)
from claritymed.core.i18n.loader import t
from claritymed.core.schemas.answer import Language
from claritymed.core.symptoms.conditions_catalog import (
    ConditionEntry,
    SymptomsConditionsCatalog,
)
from claritymed.core.symptoms.schemas import SeverityTier
from claritymed.core.symptoms.severity import tier_for_severity
from claritymed.core.symptoms.wire import DifferentialRow

logger = logging.getLogger(__name__)

# Directional-headline probability breakpoints for rank 1. Chosen to
# match the spec's "Likely / Probably / Consider" ladder; kept as
# module-level constants because the numbers are not translation
# strings (unlike confidence bucket thresholds, which the spec
# explicitly puts in i18n so operators can tune label + threshold
# together).
_HEADLINE_LIKELY = 0.60
_HEADLINE_PROBABLY = 0.40

# "Unlikely (but rule out)" rank-2+ variant fires only when the
# severity is high AND the probability is very low — surfaces
# low-probability-but-consequential findings so the user doesn't
# dismiss them.
_UNLIKELY_RULE_OUT_PROB = 0.15
_UNLIKELY_RULE_OUT_TIERS: frozenset[SeverityTier] = frozenset({"Critical", "Urgent"})

# Confidence bucket i18n key roots. Values under
# ``symptoms.confidence.thresholds.*`` are floats stored as strings by
# the i18n loader; we coerce at read time.
_CONFIDENCE_THRESHOLD_KEY = "symptoms.confidence.thresholds"
_CONFIDENCE_LABEL_KEY = "symptoms.confidence.labels"
_HEADLINE_KEY = "symptoms.headline"
_BANNER_KEY_ROOT = "symptoms.card.banner"

# Fallback thresholds when i18n is missing or malformed. Matches the
# EN defaults in configs/i18n/en/symptoms.yaml — kept in code as a
# defensive default so a stripped-down i18n file cannot silently
# produce always-"low" buckets.
_FALLBACK_THRESHOLDS: dict[ConfidenceBucket, float] = {
    "very_high": 0.75,
    "high": 0.55,
    "moderate": 0.30,
}


def _threshold(bucket: ConfidenceBucket, language: Language) -> float:
    """Resolve one confidence-bucket threshold from i18n, with fallback."""
    key = f"{_CONFIDENCE_THRESHOLD_KEY}.{bucket}"
    raw = t(key, lang=language)
    if raw == key:
        return _FALLBACK_THRESHOLDS[bucket]
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "symptoms card builder: threshold %r=%r not float; using fallback",
            key,
            raw,
        )
        return _FALLBACK_THRESHOLDS[bucket]


def confidence_bucket_for(probability: float, language: Language) -> ConfidenceBucket:
    """Bucket a raw probability using i18n-tuned thresholds."""
    if probability >= _threshold("very_high", language):
        return "very_high"
    if probability >= _threshold("high", language):
        return "high"
    if probability >= _threshold("moderate", language):
        return "moderate"
    return "low"


def confidence_label_for(bucket: ConfidenceBucket, language: Language) -> str:
    """Resolve the localized confidence-bucket label."""
    return t(f"{_CONFIDENCE_LABEL_KEY}.{bucket}", lang=language)


def directional_headline(
    rank: int,
    probability: float,
    severity_tier: SeverityTier,
    condition_name: str,
    language: Language,
) -> str:
    """Build the tier-appropriate directional headline for one card.

    Rank 1 picks from ``symptoms.headline.top.{likely|probably|consider}``
    by probability. Rank 2+ picks
    ``symptoms.headline.also.unlikely_but_rule_out`` when severity is
    Critical/Urgent AND probability is very low (surfaces the
    "unlikely but consequences are severe if missed" case); otherwise
    ``symptoms.headline.also.standard``.
    """
    if rank == 1:
        if probability >= _HEADLINE_LIKELY:
            key = f"{_HEADLINE_KEY}.top.likely"
        elif probability >= _HEADLINE_PROBABLY:
            key = f"{_HEADLINE_KEY}.top.probably"
        else:
            key = f"{_HEADLINE_KEY}.top.consider"
    else:
        if (
            severity_tier in _UNLIKELY_RULE_OUT_TIERS
            and probability < _UNLIKELY_RULE_OUT_PROB
        ):
            key = f"{_HEADLINE_KEY}.also.unlikely_but_rule_out"
        else:
            key = f"{_HEADLINE_KEY}.also.standard"
    return t(key, lang=language, condition=condition_name)


def _build_card(
    row: DifferentialRow,
    rank: int,
    entry: ConditionEntry,
    language: Language,
) -> DifferentialCard:
    tier = tier_for_severity(row.severity)
    bucket = confidence_bucket_for(row.probability, language)
    return DifferentialCard(
        condition_id=row.condition_id,
        condition_name=entry.display_name,
        probability=row.probability,
        severity_tier=tier,
        confidence_bucket=bucket,
        confidence_label=confidence_label_for(bucket, language),
        headline=directional_headline(
            rank=rank,
            probability=row.probability,
            severity_tier=tier,
            condition_name=entry.display_name,
            language=language,
        ),
        report=entry.report,
    )


def _pick_rows_for_cards(result: dict, top_n_cap: int) -> list[DifferentialRow]:
    """Pick which raw rows the cards will render.

    Prefers ``differential`` (normal completion). Falls back to
    ``partial_differential`` when the sub-session ended on cap or on
    cancel-with-meaningful-partial or with a severity override. Rows
    with zero probability are dropped. Clipped to ``top_n_cap`` after
    probability sort. The caller's dict is not mutated.

    NB: the plugin's ``_format_differential`` drops ``condition_id``
    from the LLM-facing dict — we re-hydrate here from the raw
    ``DifferentialRow`` list stashed alongside the result.
    """
    raw: list[DifferentialRow] | None = result.get("_raw_differential")
    if not raw:
        raw = result.get("_raw_partial_differential") or []
    keep = [r for r in raw if r.probability > 0]
    keep.sort(key=lambda r: -r.probability)
    return keep[:top_n_cap]


def _compute_banner_key(result: dict, cards_empty: bool) -> str | None:
    """Server-computed ``banner_key`` — the frontend's one lookup.

    Mapping mirrors the "Session-Flag → UI Branching" table in the
    feat-symptoms-multi-card-render spec. ``None`` means normal
    completion — no banner.
    """
    cancelled = bool(result.get("cancelled"))
    hit_cap = bool(result.get("hit_cap"))
    severity_override = bool(result.get("severity_override"))
    meets_threshold = bool(result.get("meets_confidence_threshold", True))
    session_expired = bool(result.get("session_expired"))

    if session_expired:
        return f"{_BANNER_KEY_ROOT}.session_expired"
    if hit_cap:
        return f"{_BANNER_KEY_ROOT}.hit_cap"
    if cancelled and severity_override:
        return f"{_BANNER_KEY_ROOT}.severity_override"
    if cancelled and meets_threshold:
        return f"{_BANNER_KEY_ROOT}.cancelled_early_meaningful"
    if cancelled and not meets_threshold and cards_empty:
        return f"{_BANNER_KEY_ROOT}.cancelled_no_result"
    return None


def build_session_meta(result: dict, cards_empty: bool) -> DifferentialSessionMeta:
    """Assemble :class:`DifferentialSessionMeta` from the raw result dict."""
    return DifferentialSessionMeta(
        cancelled=bool(result.get("cancelled")),
        hit_cap=bool(result.get("hit_cap")),
        meets_confidence_threshold=bool(result.get("meets_confidence_threshold", True)),
        severity_override=bool(result.get("severity_override")),
        max_low_severity_seen=result.get("max_low_severity_seen"),
        banner_key=_compute_banner_key(result, cards_empty),
    )


def build_differential_ready(
    result: dict,
    *,
    catalog: SymptomsConditionsCatalog,
    language: Language,
    top_n_cap: int,
) -> DifferentialReady | None:
    """Hydrate the sidecar event from the tool result dict.

    Returns ``None`` for the fall-through branches: ``user_declined``,
    ``eligible=False``, ``server_error``, and ``session_expired``
    without any cards — those cases skip the cards UI and let the
    normal free-text reply flow.

    Callers stash the raw ``list[DifferentialRow]`` (before the
    plugin's ``_format_differential`` strips ``condition_id``) under
    ``result["_raw_differential"]`` / ``result["_raw_partial_differential"]``
    so this function can join to the catalog by slug. Missing catalog
    entries are skipped with a warning — the plugin's construct-time
    validator (:func:`validate_conditions_catalog`) should have already
    caught them; this is a runtime safety net.
    """
    if not result.get("eligible", False):
        return None
    if result.get("user_declined") or result.get("server_error"):
        return None
    rows = _pick_rows_for_cards(result, top_n_cap)
    cards: list[DifferentialCard] = []
    for rank, row in enumerate(rows, start=1):
        entry = catalog.get(row.condition_id, language)
        if entry is None:
            logger.warning(
                "symptoms card builder: no catalog entry for %r in %s; "
                "skipping card (should have failed at construct time)",
                row.condition_id,
                language,
            )
            continue
        cards.append(_build_card(row, rank, entry, language))
    session = build_session_meta(result, cards_empty=not cards)
    # Session-expired with no cards has already returned above only when
    # the result has no eligibility flag; the "eligible + session_expired
    # + empty cards" branch still emits the event so the banner renders.
    return DifferentialReady(cards=cards, session=session, language=language)


def collect_raw_rows(
    from_completed: Iterable[DifferentialRow] | None = None,
    from_partial: Iterable[DifferentialRow] | None = None,
) -> tuple[list[DifferentialRow], list[DifferentialRow]]:
    """Return ``(completed, partial)`` as fresh lists, both possibly empty.

    Trivial helper the plugin uses to stash raw rows on the result
    dict without the caller having to reason about ``None`` vs list.
    """
    return list(from_completed or []), list(from_partial or [])
