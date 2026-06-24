"""Load cases from YAML, run them through ``EmergencyTriage`` per profile.

Wire diagram::

    load_cases(paths)                       # YAML → list[Case]
        │
        ├── structured cases  (case.symptoms set)
        │       → triage.assess_from_symptoms(symptoms, sensitivity=…)
        │
        └── text cases        (case.turns set)
                ├── extractor available → triage.assess(query, history, …)
                └── no extractor        → Prediction(skipped=True)

The runner is deliberately profile-agnostic: ``run_profile`` takes a
profile name + a constructed :class:`EmergencyTriage` and returns the
predictions. ``run_all_profiles`` is the convenience entry point that
loops over the four canonical profiles using one rule pack / config
loaded once.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterable

import yaml

from claritymed.core.emergency import EmergencyAssessment, EmergencyTriage
from claritymed.core.emergency.rules import load_validated_emergency_config

from evals.emergency.schemas import Case, Prediction

logger = logging.getLogger(__name__)

PROFILES: tuple[str, ...] = ("strict", "balanced", "lenient", "off")


def load_cases(paths: Iterable[Path]) -> list[Case]:
    """Load + validate every YAML file's ``cases:`` list.

    YAML shape::

        cases:
          - id: …
            source: …
            language: en
            symptoms: { primary_complaint: chest_pain, qualifiers: [...] }
            ground_truth_level: critical
            …

    Empty files and missing top-level ``cases:`` keys are tolerated so
    operators can drop a partially-curated YAML in place without
    blocking the rest of the run.
    """
    cases: list[Case] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        for entry in raw.get("cases", []):
            cases.append(Case.model_validate(entry))
    return cases


async def _predict_one(
    case: Case,
    triage: EmergencyTriage,
    *,
    sensitivity: str,
    has_extractor: bool,
) -> Prediction:
    """Run one case through the gate; build a :class:`Prediction`."""
    if case.symptoms is not None:
        result = await triage.assess_from_symptoms(
            case.symptoms,
            sensitivity=sensitivity,  # type: ignore[arg-type]
            language=case.language,
        )
        return _result_to_prediction(case.id, sensitivity, result)
    if not has_extractor:
        return Prediction(
            case_id=case.id,
            profile=sensitivity,  # type: ignore[arg-type]
            predicted_level="routine",
            skipped=True,
            skip_reason="text-mode case requires extractor; none wired",
        )
    query = case.turns[-1].text
    history = _history_from_turns(case.turns[:-1])
    result = await triage.assess(
        query,
        history,
        sensitivity=sensitivity,  # type: ignore[arg-type]
        language=case.language,
    )
    return _result_to_prediction(case.id, sensitivity, result)


def _history_from_turns(turns: list) -> list:
    """Convert authored turns into pydantic-ai history shapes.

    Returns an empty list when no prior turns exist. Cases with
    multi-turn history exercise the extractor's history-awareness
    (plan §"Sparse-input handling" worked example).
    """
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    out: list = []
    for turn in turns:
        if turn.role == "user":
            out.append(ModelRequest(parts=[UserPromptPart(content=turn.text)]))
        else:
            out.append(ModelResponse(parts=[TextPart(content=turn.text)]))
    return out


def _result_to_prediction(
    case_id: str, profile: str, result: EmergencyAssessment
) -> Prediction:
    """Project an :class:`EmergencyAssessment` into a :class:`Prediction`."""
    return Prediction(
        case_id=case_id,
        profile=profile,  # type: ignore[arg-type]
        predicted_level=result.level,
        matched_rule_ids=[m.rule_id for m in result.matched_rules],
    )


async def run_profile(
    cases: list[Case],
    triage: EmergencyTriage,
    *,
    sensitivity: str,
    has_extractor: bool,
) -> list[Prediction]:
    """Run every case against the gate at one sensitivity profile."""
    out: list[Prediction] = []
    for case in cases:
        try:
            pred = await _predict_one(
                case,
                triage,
                sensitivity=sensitivity,
                has_extractor=has_extractor,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "case %s failed under profile %s; counting as routine",
                case.id,
                sensitivity,
            )
            pred = Prediction(
                case_id=case.id,
                profile=sensitivity,  # type: ignore[arg-type]
                predicted_level="routine",
            )
        out.append(pred)
    return out


async def run_all_profiles(
    cases: list[Case],
    *,
    extractor=None,
    composer=None,
) -> dict[str, list[Prediction]]:
    """Run cases under ``strict``, ``balanced``, ``lenient``, ``off``.

    Loads the production rule pack + config once. Returns a dict keyed
    by profile name so the caller can hand each list to ``aggregate``.
    """
    cfg, rules = load_validated_emergency_config()
    triage = EmergencyTriage(
        rules=rules, config=cfg, composer=composer, extractor=extractor
    )
    has_extractor = extractor is not None
    out: dict[str, list[Prediction]] = {}
    for profile in PROFILES:
        out[profile] = await run_profile(
            cases, triage, sensitivity=profile, has_extractor=has_extractor
        )
    return out


def _sources_dir() -> Path:
    """Default directory containing committed YAML case files."""
    return Path(__file__).resolve().parent / "sources"


def discover_case_files() -> list[Path]:
    """Return every ``*.yaml`` under ``sources/`` (alphabetical)."""
    return sorted(_sources_dir().glob("*.yaml"))


def main() -> int:
    """CLI entry: ``python -m evals.emergency.runner``.

    Loads every YAML under ``sources/``, runs all four profiles
    structured-mode only (no extractor), and prints per-profile
    metrics to stdout. Suitable for ``uv run`` + CI smoke checks.

    The eval is offline tooling but still touches code paths that
    expect a request context (the ``off`` profile emits a
    ``redflag.gate_disabled`` audit event). We bracket the run with
    :func:`apply_context` so audit calls succeed cleanly under a tag
    that operators can filter out (``request_id`` prefix
    ``eval-emergency``) rather than failing in a try/except loop
    every off-mode case.
    """
    from claritymed.context import apply_context, reset_context

    from evals.emergency.metrics import aggregate

    paths = discover_case_files()
    if not paths:
        print("No case files found under evals/emergency/sources/")
        return 1
    cases = load_cases(paths)
    if not cases:
        print(f"Loaded {len(paths)} files but no cases.")
        return 1
    tokens = apply_context(request_id="eval-emergency", user_id="e2e", language="en")
    try:
        preds = asyncio.run(run_all_profiles(cases))
    finally:
        reset_context(tokens)
    print(f"Loaded {len(cases)} cases from {len(paths)} files.")
    for profile in PROFILES:
        m = aggregate(cases, preds[profile], profile=profile)
        print()
        print(f"=== profile: {profile} ===")
        print(f"  total={m.total} scored={m.scored} skipped={m.skipped}")
        print(
            f"  critical_recall={m.critical_recall:.3f} "
            f"critical_precision={m.critical_precision:.3f} "
            f"f_beta_2={m.f_beta_2:.3f}"
        )
        print(
            f"  adversarial_fpr={m.adversarial_fpr:.3f} "
            f"alert_rate_per_100={m.alert_rate_per_100:.1f}"
        )
        if m.per_rule_recall:
            print("  per_rule_recall:")
            for rid, r in sorted(m.per_rule_recall.items()):
                print(f"    {rid}: {r:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
