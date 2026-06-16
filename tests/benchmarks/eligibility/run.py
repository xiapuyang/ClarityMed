"""Eligibility A/B benchmark — direct / term_service / translation.

Runs the curated complaint set from :mod:`tests.benchmarks.eligibility.cases`
through each available strategy and reports precision / recall / F1 /
p95 latency per ``(strategy, lang)`` cell.

Usage
-----

Run every strategy on every case::

    uv run python -m tests.benchmarks.eligibility.run \\
      --strategies direct,term_service,translation \\
      --out data/bench/eligibility/<ts>/

Skip strategies that need backing services::

    uv run python -m tests.benchmarks.eligibility.run \\
      --strategies direct,translation

Restrict to one language::

    uv run python -m tests.benchmarks.eligibility.run \\
      --strategies direct --langs en

Outputs
-------

``trials.jsonl``
    One row per ``(strategy, case, trial_index)`` — strategy, case
    name, lang, expected_eligible, predicted_eligible, reason,
    evidence_hits, confidence, latency_ms, error.

``summary.csv``
    Aggregated per ``(strategy, lang)`` — TP/FP/FN/TN, precision,
    recall, F1, p95 latency. Cost in $ is *not* computed; latency is
    the only model-spend proxy this benchmark emits. Multiply by your
    provider's $/sec separately when comparing strategies that hit
    an LLM (translation) against ones that don't (direct).

Skip semantics
--------------

A strategy that can't be constructed (missing vocab file, no term
service, no reachable local provider) is logged once at startup and
skipped — the rest of the benchmark still runs and the summary CSV
omits the skipped row. This is by design: the benchmark must produce
*some* output even in a degraded environment so trends across runs
stay comparable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claritymed.config import DATA_DIR, load_env_file, load_symptoms_config
from claritymed.core.schemas.patient import Profile
from claritymed.core.symptoms.eligibility.base import (
    EligibilityResult,
    EligibilityStrategy,
)
from claritymed.core.symptoms.eligibility.direct import (
    DirectEligibility,
    EvidenceVocabMap,
)
from claritymed.core.symptoms.eligibility.translation import TranslationEligibility
from claritymed.core.symptoms.schemas import (
    DatasetSpec,
    TermServiceEligibilityEntry,
    TranslationEligibilityEntry,
)
from claritymed.errors import (
    EligibilityStrategyConfigError,
    EligibilityStrategyUnavailableError,
)
from claritymed.orchestrator.features.symptoms_plugin import _load_ddxplus_vocab
from claritymed.stores.models import is_provider_available, resolve_provider

from tests.benchmarks.eligibility.cases import (
    EligibilityCase,
    select_cases,
)

logger = logging.getLogger("eligibility_bench")

# Strategy ids the runner knows how to wire.
SUPPORTED_STRATEGIES = ("direct", "term_service", "translation")

# Default output root — mirrors tool_invoke's ``data/bench/<family>/<ts>/``
# pattern so existing operator habits transfer.
DEFAULT_OUT_ROOT = Path("data") / "bench" / "eligibility"


# --- vocab + strategy construction -----------------------------------------


def _resolve_dataset_for_bench() -> DatasetSpec:
    """Pick the dataset to benchmark against.

    DDXPlus is the only dataset shipped today, so we resolve it from
    the loaded symptoms config rather than hardcoding a synthetic
    DatasetSpec — this way the benchmark exercises the same
    ``native_language`` / ``severity_high_specificity_evidence_ids``
    the production wiring sees.
    """
    config = load_symptoms_config()
    for ds in config.datasets:
        if ds.id == "ddxplus":
            return ds
    raise RuntimeError(
        "expected a 'ddxplus' dataset in configs/symptoms.yaml; the "
        "benchmark requires it to anchor expected_eligible labels."
    )


def _load_vocab_map() -> EvidenceVocabMap:
    """Build ``{ddxplus: {evidence_id: token_set}}`` from disk."""
    data_dir = str(Path(str(DATA_DIR)) / "symptoms" / "ddxplus")
    per_evidence = _load_ddxplus_vocab(data_dir)
    if not per_evidence:
        logger.warning(
            "ddxplus vocab not found at %s — direct + translation "
            "strategies will skip. Run the ddxplus prepare step first.",
            data_dir,
        )
        return {}
    return {"ddxplus": per_evidence}


def _build_direct(vocabs: EvidenceVocabMap) -> EligibilityStrategy | None:
    """Construct ``DirectEligibility`` from disk-loaded vocabs.

    Returns ``None`` when vocabs are missing — the caller logs a
    single skip line.
    """
    if not vocabs:
        return None
    return DirectEligibility(vocabs=vocabs)


def _build_term_service() -> EligibilityStrategy | None:
    """Try to construct ``TermServiceEligibility``.

    Pulls the catalog entry from ``configs/symptoms.yaml`` so the
    benchmark uses the same wiring production would. Returns ``None``
    when:

    * The catalog has no ``kind: term_service`` entry.
    * The term service is unreachable (no concepts.jsonl, no UMLS
      backing).
    * The sidecars file isn't present for ddxplus.
    """
    from claritymed.core.rag.terms.factory import build_term_service
    from claritymed.core.symptoms.eligibility.term_service import (
        SidecarMap,
        TermServiceEligibility,
    )

    config = load_symptoms_config()
    entry = next(
        (
            e
            for e in config.eligibility.catalog
            if isinstance(e, TermServiceEligibilityEntry)
        ),
        None,
    )
    if entry is None:
        logger.warning(
            "no kind=term_service entry in eligibility catalog — "
            "term_service strategy skipped."
        )
        return None

    try:
        term_service = build_term_service()
    except Exception:
        logger.warning(
            "term_service unavailable (build_term_service failed); strategy skipped",
            exc_info=True,
        )
        return None

    sidecars: SidecarMap = _load_ddxplus_sidecars()
    if not sidecars:
        logger.warning(
            "ddxplus concept sidecars not found — term_service "
            "strategy will report strategy_unavailable for every case."
        )
    return TermServiceEligibility(term_service=term_service, sidecars=sidecars)


def _load_ddxplus_sidecars() -> "dict[str, dict[str, str]]":
    """Best-effort load of evidence→concept sidecar JSON.

    Reads the same file the production plugin reads
    (``DATA_DIR/symptoms/ddxplus/evidence_concepts.json``, produced by
    ``prepare.py --build-sidecar``). Missing sidecars degrade
    term_service to "strategy_unavailable for all", which the summary
    still captures honestly.
    """
    candidate = Path(str(DATA_DIR)) / "symptoms" / "ddxplus" / "evidence_concepts.json"
    if not candidate.exists():
        return {}
    try:
        with candidate.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        logger.warning(
            "failed to parse %s; treating as empty", candidate, exc_info=True
        )
        return {}
    if not isinstance(raw, dict):
        return {}
    return {"ddxplus": raw}


def _build_translation(
    direct: EligibilityStrategy | None,
) -> EligibilityStrategy | None:
    """Construct ``TranslationEligibility`` wrapping the direct strategy.

    Returns ``None`` when the underlying ``direct`` is unavailable
    (no shared matching layer) or the configured local provider isn't
    reachable.
    """
    if not isinstance(direct, DirectEligibility):
        logger.warning(
            "translation strategy requires a working direct strategy; "
            "underlying direct is None — translation skipped."
        )
        return None

    config = load_symptoms_config()
    entry = next(
        (
            e
            for e in config.eligibility.catalog
            if isinstance(e, TranslationEligibilityEntry)
        ),
        None,
    )
    if entry is None:
        logger.warning(
            "no kind=translation entry in eligibility catalog — "
            "translation strategy skipped."
        )
        return None

    provider = resolve_provider(override=entry.provider_id)
    if provider.kind != "local":
        logger.warning(
            "translation provider %r has kind=%r; benchmark refuses to "
            "send PHI to a cloud provider — strategy skipped.",
            entry.provider_id,
            provider.kind,
        )
        return None
    if not is_provider_available(provider):
        logger.warning(
            "translation provider %r unreachable (env var / endpoint "
            "not configured); strategy skipped.",
            entry.provider_id,
        )
        return None

    try:
        return TranslationEligibility(
            provider_id=entry.provider_id,
            prompt_name=entry.prompt_name,
            max_tokens=entry.max_tokens,
            direct_strategy=direct,
        )
    except (
        EligibilityStrategyConfigError,
        EligibilityStrategyUnavailableError,
    ) as exc:
        logger.warning("translation strategy unavailable (%s); skipped", exc)
        return None


def build_strategies(
    requested: list[str],
) -> "dict[str, EligibilityStrategy]":
    """Build every requested strategy that can be constructed.

    Logs a single skip line for each strategy that can't be wired
    (no exception bubbles up — the benchmark always returns *some*
    result).
    """
    unknown = [s for s in requested if s not in SUPPORTED_STRATEGIES]
    if unknown:
        raise ValueError(
            f"unknown strategy ids: {unknown!r}; "
            f"supported: {list(SUPPORTED_STRATEGIES)!r}"
        )

    out: dict[str, EligibilityStrategy] = {}
    vocabs = _load_vocab_map()
    direct = _build_direct(vocabs)
    if "direct" in requested and direct is not None:
        out["direct"] = direct
    if "term_service" in requested:
        ts = _build_term_service()
        if ts is not None:
            out["term_service"] = ts
    if "translation" in requested:
        tr = _build_translation(direct)
        if tr is not None:
            out["translation"] = tr
    return out


# --- trial loop ------------------------------------------------------------


async def _run_one_trial(
    strategy: EligibilityStrategy,
    case: EligibilityCase,
    dataset: DatasetSpec,
    profile: Profile,
) -> "tuple[EligibilityResult | None, float, str | None]":
    """Run one check and return ``(result, latency_ms, error_str)``.

    Errors are caught and surfaced as a string in the third slot so
    one strategy blowing up on one case doesn't kill the rest of the
    matrix. ``result`` is ``None`` when an error occurred.
    """
    started = time.perf_counter()
    try:
        result = await strategy.check(
            case.complaint,
            case.lang,
            profile,
            dataset,
        )
    except Exception as exc:  # noqa: BLE001 — surface to row, don't kill the run
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return None, elapsed_ms, f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result, elapsed_ms, None


def _trial_row(
    *,
    strategy_id: str,
    trial_index: int,
    case: EligibilityCase,
    result: EligibilityResult | None,
    latency_ms: float,
    error: str | None,
) -> dict[str, Any]:
    """Flatten one trial to a JSON-serializable row."""
    predicted = result.eligible if result is not None else None
    return {
        "strategy": strategy_id,
        "case": case.name,
        "lang": case.lang,
        "category": case.category,
        "expected_eligible": case.expected_eligible,
        "predicted_eligible": predicted,
        "reason": result.reason if result is not None else None,
        "evidence_hits": list(result.evidence_hits) if result is not None else [],
        "confidence": result.confidence if result is not None else None,
        "latency_ms": round(latency_ms, 2),
        "trial_index": trial_index,
        "error": error,
        "complaint": case.complaint,
    }


# --- aggregation -----------------------------------------------------------


def _p95(values: list[float]) -> float:
    """Return the 95th percentile of ``values`` (0.0 if empty)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(round(0.95 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Roll ``trials.jsonl`` rows into one row per ``(strategy, lang)``.

    Skipped cells (no predictions at all) produce a row with zeroed
    metrics and ``n=0`` rather than being silently dropped — this
    keeps the summary table shape stable across environments.
    """
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["strategy"], row["lang"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (strategy, lang), cell_rows in sorted(cells.items()):
        tp = fp = fn = tn = errored = 0
        latencies: list[float] = []
        for row in cell_rows:
            if row["error"] is not None:
                errored += 1
                continue
            latencies.append(row["latency_ms"])
            expected = row["expected_eligible"]
            predicted = row["predicted_eligible"]
            if expected and predicted:
                tp += 1
            elif not expected and predicted:
                fp += 1
            elif expected and not predicted:
                fn += 1
            else:
                tn += 1

        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        summary.append(
            {
                "strategy": strategy,
                "lang": lang,
                "n": total,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "errored": errored,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "latency_p50_ms": round(statistics.median(latencies), 2)
                if latencies
                else 0.0,
                "latency_p95_ms": round(_p95(latencies), 2),
                "latency_mean_ms": round(statistics.mean(latencies), 2)
                if latencies
                else 0.0,
            }
        )
    return summary


# --- output ----------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary_table(summary: list[dict[str, Any]]) -> None:
    """Pretty-print the summary CSV rows to stdout.

    Plain ASCII — no rich/tabulate dependency. Column order matches
    the CSV so a screenshot of the terminal matches the file on disk.
    """
    if not summary:
        print("(no rows — every strategy was skipped)")
        return

    cols = (
        "strategy",
        "lang",
        "n",
        "tp",
        "fp",
        "fn",
        "tn",
        "errored",
        "precision",
        "recall",
        "f1",
        "latency_p50_ms",
        "latency_p95_ms",
    )
    widths = {c: max(len(c), max(len(str(r[c])) for r in summary)) for c in cols}

    def _fmt(row: dict[str, Any]) -> str:
        return "  ".join(str(row[c]).rjust(widths[c]) for c in cols)

    header_row = {c: c for c in cols}
    print(_fmt(header_row))
    print("  ".join("-" * widths[c] for c in cols))
    for row in summary:
        print(_fmt(row))


# --- entrypoint ------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--strategies",
        default=",".join(SUPPORTED_STRATEGIES),
        help=(
            "comma-separated strategy ids to run (subset of "
            f"{list(SUPPORTED_STRATEGIES)!r}; default: all)"
        ),
    )
    p.add_argument(
        "--langs",
        default="en,zh",
        help="comma-separated languages to include (default: en,zh)",
    )
    p.add_argument(
        "--cases",
        default=None,
        help="optional comma-separated case names to restrict to",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=1,
        help=(
            "trials per (strategy, case) — >1 gives a more honest p95 "
            "for translation but burns more LLM calls (default: 1)"
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        help="output directory (default: data/bench/eligibility/<ts>/)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    requested = [s.strip() for s in args.strategies.split(",") if s.strip()]
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    case_names = (
        [n.strip() for n in args.cases.split(",") if n.strip()] if args.cases else None
    )

    strategies = build_strategies(requested)
    if not strategies:
        logger.error("no strategies could be constructed; nothing to run.")
        return 2

    dataset = _resolve_dataset_for_bench()
    profile = Profile()
    cases = select_cases(langs=langs, names=case_names)  # type: ignore[arg-type]
    if not cases:
        logger.error("case filter excluded everything; nothing to run.")
        return 2

    rows: list[dict[str, Any]] = []
    for strategy_id, strategy in strategies.items():
        logger.info(
            "running strategy=%s over %d cases × %d trial(s)",
            strategy_id,
            len(cases),
            args.trials,
        )
        for case in cases:
            for trial_index in range(args.trials):
                result, latency_ms, error = await _run_one_trial(
                    strategy, case, dataset, profile
                )
                rows.append(
                    _trial_row(
                        strategy_id=strategy_id,
                        trial_index=trial_index,
                        case=case,
                        result=result,
                        latency_ms=latency_ms,
                        error=error,
                    )
                )

    summary = aggregate(rows)

    out_dir = (
        Path(args.out)
        if args.out
        else DEFAULT_OUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    _write_jsonl(out_dir / "trials.jsonl", rows)
    _write_csv(out_dir / "summary.csv", summary)
    logger.info(
        "wrote %d trial rows and %d summary rows to %s",
        len(rows),
        len(summary),
        out_dir,
    )

    print()  # blank line before the table
    _print_summary_table(summary)
    return 0


def main() -> int:
    load_env_file()
    args = _parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        logger.warning("interrupted; partial outputs (if any) are on disk.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
