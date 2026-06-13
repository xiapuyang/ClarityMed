"""LLM-as-judge over a benchmark run's ``trials.jsonl``.

Stage 2 of the ingest-tool benchmark — see ``run.py`` for stage 1.

The judge is a one-shot LLM call per trial:

* No tools, no AskService, no PHI guard wiring. Just ``Agent(model,
  output_type=Judgment)`` with the bilingual prompt template in
  ``judge_prompt.yaml``.
* Default judge is ``omlx`` (local, free, PHI-safe). Any catalog
  provider id can be passed via ``--judge-provider``.
* By default scores only ``hard`` tier — base predicates are
  deterministic and judge calls there are wasted spend. Override with
  ``--tiers``.

Example::

    uv run python -m tests.benchmarks.ingest_tools.judge \\
        --trials data/bench/20260612_104500/trials.jsonl \\
        --judge-provider omlx
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from claritymed.config import load_env_file
from claritymed.core.llm.model import build_model
from claritymed.stores.models import resolve_provider

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent / "judge_prompt.yaml"


# --- output schema ---------------------------------------------------


class Judgment(BaseModel):
    """Structured output the judge model returns."""

    score: int = Field(ge=0, le=5)
    reason: str = Field(min_length=1, max_length=400)


@dataclass
class JudgedRecord:
    request_id: str
    model: str
    lang: str
    case_name: str
    tier: str
    judge_provider: str
    score: int
    reason: str
    judge_error: str | None


# --- prompt rendering ------------------------------------------------


def _load_templates() -> dict[str, str]:
    return yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))


def _format_tool_calls(tool_calls: list[dict]) -> str:
    if not tool_calls:
        return "(none)"
    lines = []
    for c in tool_calls:
        # Keep args compact; full args present in jsonl already.
        args_json = json.dumps(c["args"], ensure_ascii=False)
        lines.append(f"- {c['tool_name']}({args_json})")
    return "\n".join(lines)


def _format_ask_block(ask_questions: list[dict]) -> str:
    if not ask_questions:
        return "(none)"
    lines = []
    for i, payload in enumerate(ask_questions, 1):
        qs = payload.get("questions", [])
        for q in qs:
            lines.append(f"- [{i}] {q.get('question', '?')}")
    return "\n".join(lines) if lines else "(none)"


def _expected_target_hint(trial: dict) -> str:
    eb = trial["expected_behavior"]
    if eb == "call_tool":
        return f"expected tool: {trial['expected_tool']}"
    if eb == "call_tools":
        return f"expected tools (any order): {trial['expected_tools']}"
    if eb == "decline":
        return "no destructive tool should fire"
    if eb == "ask_tool":
        return (
            "model MUST call ask_user_question (enumerable answer space) "
            "instead of guessing or asking in plain text"
        )
    if eb == "ask_tool_or_text":
        return (
            "model should clarify before saving — either via "
            "ask_user_question or a plain-text question is acceptable"
        )
    if eb == "ask_then_call_tool":
        return (
            "model should ask for the missing field, then call the "
            f"target ingest tool: {trial.get('expected_tool')}"
        )
    return eb


def _render_prompt(template: str, trial: dict) -> str:
    return template.format(
        user_prompt=trial["user_prompt"],
        expected_behavior=trial["expected_behavior"],
        expected_target_hint=_expected_target_hint(trial),
        tool_calls_block=_format_tool_calls(trial["tool_calls"]),
        ask_count=trial["ask_user_q_count"],
        ask_block=_format_ask_block(trial["ask_questions"]),
        final_response_text=(trial["final_response_text"] or "(empty)").strip()[:2000],
        outcome=trial["outcome"],
        predicate_pass=trial["predicate_pass"],
        predicate_reason=trial["predicate_reason"],
    )


# --- judging ---------------------------------------------------------


async def _judge_one(
    agent: Agent[None, Judgment],
    templates: dict[str, str],
    trial: dict,
    judge_provider: str,
) -> JudgedRecord:
    template = templates.get(trial["lang"], templates["en"])
    prompt = _render_prompt(template, trial)
    try:
        result = await agent.run(prompt)
        judgment = result.output
        return JudgedRecord(
            request_id=trial["request_id"],
            model=trial["model"],
            lang=trial["lang"],
            case_name=trial["case_name"],
            tier=trial["tier"],
            judge_provider=judge_provider,
            score=judgment.score,
            reason=judgment.reason,
            judge_error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return JudgedRecord(
            request_id=trial["request_id"],
            model=trial["model"],
            lang=trial["lang"],
            case_name=trial["case_name"],
            tier=trial["tier"],
            judge_provider=judge_provider,
            score=-1,
            reason="",
            judge_error=f"{type(exc).__name__}: {exc}",
        )


# --- aggregate -------------------------------------------------------


def _summary_rows(judged: list[JudgedRecord]) -> list[dict]:
    by_cell: dict[tuple[str, str, str], list[JudgedRecord]] = defaultdict(list)
    for j in judged:
        by_cell[(j.model, j.lang, j.case_name)].append(j)

    rows: list[dict] = []
    for (model, lang, case_name), cell in sorted(by_cell.items()):
        scores = [j.score for j in cell if j.judge_error is None and j.score >= 0]
        rows.append(
            {
                "model": model,
                "lang": lang,
                "case": case_name,
                "tier": cell[0].tier,
                "judge_provider": cell[0].judge_provider,
                "n_judged": len(scores),
                "n_judge_errors": sum(1 for j in cell if j.judge_error),
                "mean_score": round(statistics.mean(scores), 2) if scores else "",
                "median_score": round(statistics.median(scores), 2) if scores else "",
                "stdev_score": (
                    round(statistics.stdev(scores), 2) if len(scores) > 1 else ""
                ),
                "min_score": min(scores) if scores else "",
                "max_score": max(scores) if scores else "",
            }
        )
    return rows


# --- CLI -------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trials",
        required=True,
        help="path to trials.jsonl emitted by run.py",
    )
    p.add_argument(
        "--judge-provider",
        default="omlx",
        help="provider id of the judge model (default: omlx)",
    )
    p.add_argument(
        "--tiers",
        default="base,hard,fp",
        help="trial tiers to judge (default: hard; use 'base,hard,fp' for all)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="parallel judge calls (default: 4)",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="output dir for judge_<provider>.{jsonl,csv} (default: same dir as --trials)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _load_trials(path: Path, tiers: set[str]) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("tier") in tiers:
                out.append(row)
    return out


async def _main_async(args: argparse.Namespace) -> int:
    load_env_file()

    trials_path = Path(args.trials)
    if not trials_path.exists():
        print(f"trials file not found: {trials_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else trials_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}
    trials = _load_trials(trials_path, tiers)
    if not trials:
        print(f"no trials in tiers={tiers} found in {trials_path}", file=sys.stderr)
        return 2

    provider = resolve_provider(override=args.judge_provider)
    model = build_model(provider)
    agent = Agent(model=model, output_type=Judgment, retries=2)

    templates = _load_templates()

    jsonl_path = out_dir / f"judge_{provider.id}.jsonl"
    csv_path = out_dir / f"judge_{provider.id}.csv"

    print(
        f"judging {len(trials)} trials via {provider.id} "
        f"(concurrency={args.concurrency}); tiers={sorted(tiers)} → {jsonl_path}"
    )

    sem = asyncio.Semaphore(args.concurrency)
    results: list[JudgedRecord] = []

    async def _bounded(trial: dict, idx: int, total: int) -> JudgedRecord:
        async with sem:
            rec = await _judge_one(agent, templates, trial, provider.id)
            tag = f"err:{rec.judge_error}" if rec.judge_error else f"score={rec.score}"
            print(
                f"[{idx + 1}/{total}] {rec.model:<22} {rec.lang} "
                f"{rec.case_name:<30} -> {tag}"
            )
            return rec

    tasks = [_bounded(t, i, len(trials)) for i, t in enumerate(trials)]

    with jsonl_path.open("w", encoding="utf-8") as fh:
        # gather collects in order; writing all at end keeps JSONL ordered
        # the same as trials, which is the easiest to eyeball.
        for coro in asyncio.as_completed(tasks):
            rec = await coro
            results.append(rec)
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            fh.flush()

    rows = _summary_rows(results)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nwrote {len(results)} judgments → {jsonl_path}")
    print(f"wrote {len(rows)} cell summaries → {csv_path}")
    return 0


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
