"""Composite-recall E2E — real local extractor + rule engine.

Why this file exists, in one sentence: the structured-mode baseline
under ``evals/emergency/sources/{ddxplus_subset,public_vignettes,
synthetic_adversarial}.yaml`` measures **rule-layer** recall (the
extractor is bypassed, symptoms are pre-canonicalized), so it cannot
catch a regression where the extractor LLM stops producing the
qualifier tokens the rule pack expects. This test runs the **real
extractor** over ``text_mode_critical.yaml`` and reports the
*composite* recall — the metric the user actually experiences.

How regressions get attributed:

The artifact JSON dump (``logs/e2e/emergency_composite_<pid>.json``)
contains, per case, the full extracted ``ExtractedSymptoms`` plus the
``matched_rule_ids`` under each of the four sensitivity profiles. So
when composite recall drops, an operator can open the artifact and
see whether the extractor failed to surface the load-bearing
qualifier (extractor regression — fix
``core/prompts/store/emergency_extractor.yaml``) or surfaced the
right qualifier but the rule did not fire (rule-pack regression —
fix ``configs/emergency_rules.yaml``).

Provider matrix:

This test honors the same ``CLARITYMED_E2E_PROVIDERS=omlx,ollama,...``
selector as the rest of ``tests/e2e/``, but **only ``kind=local``
providers run** — KTD-E1 mandates that the extractor LLM sees raw
patient prose, so it must stay on the box. Cloud entries in the
matrix skip cleanly with a clear reason rather than getting silently
rerouted to a local provider, which would make the per-provider
metrics meaningless.

Cost discipline:

The extractor LLM is invoked **once per case** — extracted symptoms
are then scored against all four sensitivity profiles via
``assess_from_symptoms``, not via ``triage.assess`` (which would
re-extract each profile). Composer is unwired (``composer=None``)
because the composite-recall metric reads ``level`` and
``matched_rule_ids``, neither of which depend on composer prose.
This keeps a 24-case run at ~75-120 s on a 7-14B-class local model
instead of 5-10 min if we naively looped ``assess()``.

Run with::

    CLARITYMED_E2E_PROVIDERS=omlx uv run pytest \\
        tests/e2e/test_emergency_composite_e2e.py -v --no-cov

To compare providers in one invocation::

    CLARITYMED_E2E_PROVIDERS=omlx,ollama uv run pytest \\
        tests/e2e/test_emergency_composite_e2e.py -v --no-cov
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

from claritymed.context import apply_context, reset_context

logger = logging.getLogger(__name__)

USER_ID = "e2e"
# Bare-minimum floor — "is the extractor + rule pack fundamentally
# wired correctly". Tuning targets live in the dumped artifact (Phase
# 5 metrics doc), not as test assertions; pinning the floor too high
# turns local-model jitter into CI flakes.
BALANCED_RECALL_FLOOR = 0.5
# Same idea on the adversarial side: a perfectly-tuned gate would
# hold this at 0. The floor here is "the gate is not gleefully firing
# critical on every benign-looking input"; finer tuning is artifact
# territory.
BALANCED_ADV_FPR_CEILING = 0.50

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "logs" / "e2e"


@pytest.fixture
def _ctx():
    """Apply request context so off-profile audit emits succeed cleanly."""
    tokens = apply_context("20260623e2ecomposite", USER_ID, "en")
    yield
    reset_context(tokens)


async def test_composite_recall_against_text_mode_cases(
    e2e_provider_id: str,
    _ctx,
) -> None:
    """Drive the gate end-to-end on text-mode YAML cases.

    Steps:
    1. Skip if the selected provider is not ``kind=local`` (PHI floor).
    2. Build :class:`LLMExtractor` against that provider's model.
    3. Load text-mode cases from ``evals/emergency/sources/`` (the
       structured-mode cases are filtered out — this test only
       exercises the extractor path).
    4. For each case: extract once, then score under each of the four
       sensitivity profiles via ``assess_from_symptoms``.
    5. Aggregate per-profile metrics; dump artifact for inspection.
    6. Soft-assert ``balanced`` critical recall floor — anything below
       means the extractor is fundamentally broken, not that one rule
       could use prompt tuning.
    """
    from claritymed.core.emergency import EmergencyTriage
    from claritymed.core.emergency.extractor import LLMExtractor
    from claritymed.core.emergency.rules import load_validated_emergency_config
    from claritymed.core.llm.model import build_model
    from claritymed.stores.models import load_models

    from evals.emergency.metrics import aggregate
    from evals.emergency.runner import (
        PROFILES,
        _history_from_turns,
        discover_case_files,
        load_cases,
    )
    from evals.emergency.schemas import Prediction

    catalog = {p.id: p for p in load_models().providers}
    provider = catalog[e2e_provider_id]
    if provider.kind != "local":
        pytest.skip(
            f"provider {e2e_provider_id!r} is kind={provider.kind!r}; the "
            "emergency extractor must run on kind=local (KTD-E1 — PHI floor)."
        )

    cases = [c for c in load_cases(discover_case_files()) if c.turns]
    if not cases:
        pytest.skip(
            "No text-mode cases found under evals/emergency/sources/. "
            "Author at least one ``turns:`` case before running this test."
        )

    model = build_model(provider)
    # English system prompt regardless of case language — the extractor
    # prompt instructs the LLM to read mixed-language input and emit
    # English canonical tokens. See CLAUDE.md
    # CLARITYMED_TOOL_PROMPT_LANG discussion.
    extractor = LLMExtractor(model, language="en")
    cfg, rules = load_validated_emergency_config()
    # composer=None: we only score ``level`` + ``matched_rule_ids``;
    # composer prose does not move either metric, so paying for a
    # second LLM call per (case, profile) is pure waste.
    triage = EmergencyTriage(rules=rules, config=cfg, composer=None)

    per_case_records: list[dict] = []
    preds_by_profile: dict[str, list[Prediction]] = {p: [] for p in PROFILES}
    start = time.monotonic()

    for case in cases:
        query = case.turns[-1].text
        history = _history_from_turns(case.turns[:-1])
        try:
            symptoms = await extractor.extract(query, history)
        except Exception as exc:  # noqa: BLE001
            # An extractor failure on one case must not abort the run
            # — local models hiccup. Record the failure and continue;
            # the aggregator will count the case as routine/skipped.
            logger.exception("[composite e2e] extractor failed on case %s", case.id)
            per_case_records.append(
                {
                    "case_id": case.id,
                    "language": case.language,
                    "ground_truth_level": case.ground_truth_level,
                    "ground_truth_rule_id": case.ground_truth_rule_id,
                    "is_adversarial": case.is_adversarial,
                    "extracted": None,
                    "extractor_error": f"{type(exc).__name__}: {exc}",
                    "predictions": {},
                }
            )
            for profile in PROFILES:
                preds_by_profile[profile].append(
                    Prediction(
                        case_id=case.id,
                        profile=profile,  # type: ignore[arg-type]
                        predicted_level="routine",
                        skipped=True,
                        skip_reason=f"extractor exception: {type(exc).__name__}",
                    )
                )
            continue

        record: dict = {
            "case_id": case.id,
            "language": case.language,
            "ground_truth_level": case.ground_truth_level,
            "ground_truth_rule_id": case.ground_truth_rule_id,
            "is_adversarial": case.is_adversarial,
            "extracted": symptoms.model_dump(),
            "predictions": {},
        }
        for profile in PROFILES:
            assessment = await triage.assess_from_symptoms(
                symptoms,
                sensitivity=profile,  # type: ignore[arg-type]
                language=case.language,
            )
            pred = Prediction(
                case_id=case.id,
                profile=profile,  # type: ignore[arg-type]
                predicted_level=assessment.level,
                matched_rule_ids=[m.rule_id for m in assessment.matched_rules],
            )
            preds_by_profile[profile].append(pred)
            record["predictions"][profile] = {
                "predicted_level": pred.predicted_level,
                "matched_rule_ids": pred.matched_rule_ids,
            }
        per_case_records.append(record)

    wall = time.monotonic() - start
    metrics_by_profile = {
        profile: aggregate(cases, preds, profile=profile).model_dump()
        for profile, preds in preds_by_profile.items()
    }

    artifact = {
        "provider_id": e2e_provider_id,
        "model_name": provider.model,
        "case_count": len(cases),
        "wall_clock_seconds": round(wall, 2),
        "balanced_recall_floor": BALANCED_RECALL_FLOOR,
        "balanced_adv_fpr_ceiling": BALANCED_ADV_FPR_CEILING,
        "metrics_by_profile": metrics_by_profile,
        "per_case": per_case_records,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / f"emergency_composite_{e2e_provider_id}.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    balanced = metrics_by_profile["balanced"]
    logger.info(
        "[composite e2e] provider=%s wall=%.1fs balanced: "
        "recall=%.2f precision=%.2f f2=%.2f adv_fpr=%.2f -> %s",
        e2e_provider_id,
        wall,
        balanced["critical_recall"],
        balanced["critical_precision"],
        balanced["f_beta_2"],
        balanced["adversarial_fpr"],
        artifact_path,
    )

    # Headline soft assertions — bare-minimum floors. Fine-grained
    # tuning happens off-line against the dumped artifact, not by
    # ratcheting these numbers in CI.
    assert balanced["critical_recall"] >= BALANCED_RECALL_FLOOR, (
        f"composite critical_recall under `balanced` = "
        f"{balanced['critical_recall']:.2f} < floor "
        f"{BALANCED_RECALL_FLOOR:.2f}. Likely root cause: extractor failed "
        f"to surface canonical qualifiers (e.g. radiation_left_arm).\n"
        f"  artifact: {artifact_path}"
    )
    assert balanced["adversarial_fpr"] <= BALANCED_ADV_FPR_CEILING, (
        f"composite adversarial FPR under `balanced` = "
        f"{balanced['adversarial_fpr']:.2f} > ceiling "
        f"{BALANCED_ADV_FPR_CEILING:.2f}. Likely root cause: extractor is "
        f"over-canonicalizing benign prose into critical qualifiers.\n"
        f"  artifact: {artifact_path}"
    )
