"""Inspect failing trials in a benchmark run.

Filters ``trials.jsonl`` by outcome and pretty-prints user prompt, tool
calls, ask_user_question payloads, and the final text reply. When a
``judge_<provider>.jsonl`` sits alongside in the same run dir, judge
scores + reasons are joined in.

Default: shows every trial whose ``outcome != "correct"``. Use
``--outcomes`` to filter to a specific failure category, or
``--cases`` / ``--models`` / ``--langs`` to narrow further.

Examples::

    # all fails in a run
    uv run python -m tests.benchmarks.ingest_tools.inspect_fails \\
        --run data/bench/20260612_104500/

    # only the ask_* failures on omlx
    uv run python -m tests.benchmarks.ingest_tools.inspect_fails \\
        --run data/bench/20260612_104500/ \\
        --outcomes no_ask,asked_instead --models omlx

For wire-level debugging (system prompt, raw model messages, tool
schemas the LLM saw), enable Phoenix tracing in ``configs/app.yaml``
and re-run; each trial's ``request_id`` in trials.jsonl is the same
value that lands in audit.log lines and the Phoenix span baggage
``claritymed.request_id`` — grep one to find the other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CORRECT_OUTCOMES: frozenset[str] = frozenset(
    {
        "correct",
        "correct_with_extra",
        "asked_with_call",
        "correct_text_ask",
        "called_without_asking",
    }
)


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_judge_index(run_dir: Path) -> dict[str, dict]:
    """Map ``request_id`` → judge record (from any ``judge_*.jsonl`` found)."""
    index: dict[str, dict] = {}
    for path in sorted(run_dir.glob("judge_*.jsonl")):
        for rec in _load_jsonl(path):
            # If multiple judges exist, last write wins; user can pass
            # one specific path via --judge if they need precision.
            index[rec["request_id"]] = rec
    return index


def _format_tool_calls(tool_calls: list[dict]) -> str:
    if not tool_calls:
        return "  (none)"
    lines = []
    for c in tool_calls:
        args_json = json.dumps(c["args"], ensure_ascii=False, indent=2)
        indented = "\n    ".join(args_json.splitlines())
        lines.append(f"  - {c['tool_name']}\n    {indented}")
    return "\n".join(lines)


def _format_ask(ask_questions: list[dict]) -> str:
    if not ask_questions:
        return "  (none)"
    lines = []
    for i, payload in enumerate(ask_questions, 1):
        for q in payload.get("questions", []):
            opts = ", ".join(o.get("label", "?") for o in q.get("options", []))
            lines.append(f"  [{i}] {q.get('question', '?')}")
            if opts:
                lines.append(f"      options: {opts}")
    return "\n".join(lines)


def _print_trial(trial: dict, judge: dict | None) -> None:
    print("=" * 78)
    print(
        f"{trial['case_name']:<28} {trial['model']:<22} "
        f"u={trial['lang']} t={trial.get('tool_prompt_lang', trial['lang'])}  "
        f"tier={trial['tier']}"
    )
    print(f"request_id={trial['request_id']}  (grep audit.log / Phoenix)")
    print(
        f"outcome={trial['outcome']}  "
        f"predicate_pass={trial['predicate_pass']}  "
        f"latency={trial['latency_ms']:.0f}ms"
    )
    print(f"expected: {trial['expected_behavior']}", end="")
    if trial.get("expected_tool"):
        print(f" → {trial['expected_tool']}")
    elif trial.get("expected_tools"):
        print(f" → {trial['expected_tools']}")
    else:
        print()
    print(f"predicate_reason: {trial['predicate_reason']}")
    if trial.get("had_error"):
        print(f"error: {trial.get('error_msg', '<missing>')}")
    print("\n--- USER PROMPT ---")
    print(f"  {trial['user_prompt']}")
    print("\n--- TOOL CALLS ---")
    print(_format_tool_calls(trial["tool_calls"]))
    print("\n--- ASK_USER_QUESTION ---")
    print(_format_ask(trial["ask_questions"]))
    print("\n--- FINAL TEXT ---")
    text = trial["final_response_text"] or "(empty)"
    for line in text.splitlines() or [""]:
        print(f"  {line}")
    if judge is not None:
        print("\n--- JUDGE ---")
        print(f"  provider: {judge['judge_provider']}  score: {judge['score']}")
        print(f"  reason: {judge['reason']}")
        if judge.get("judge_error"):
            print(f"  judge_error: {judge['judge_error']}")
    print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        required=True,
        help="run directory (containing trials.jsonl and optional judge_*.jsonl)",
    )
    p.add_argument(
        "--outcomes",
        default=None,
        help=(
            "comma-separated outcome filter; default = everything except "
            "correct/correct_with_extra/asked_with_call"
        ),
    )
    p.add_argument("--cases", default=None, help="comma-separated case names")
    p.add_argument("--models", default=None, help="comma-separated provider ids")
    p.add_argument("--langs", default=None, help="comma-separated languages")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after N trials (useful when many fail)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run)
    trials_path = run_dir / "trials.jsonl"
    if not trials_path.exists():
        print(f"trials.jsonl not found: {trials_path}", file=sys.stderr)
        return 2

    trials = _load_jsonl(trials_path)
    judge_index = _load_judge_index(run_dir)

    if args.outcomes:
        wanted = {o.strip() for o in args.outcomes.split(",") if o.strip()}
        trials = [t for t in trials if t["outcome"] in wanted]
    else:
        trials = [t for t in trials if t["outcome"] not in CORRECT_OUTCOMES]

    if args.cases:
        names = {n.strip() for n in args.cases.split(",") if n.strip()}
        trials = [t for t in trials if t["case_name"] in names]
    if args.models:
        names = {n.strip() for n in args.models.split(",") if n.strip()}
        trials = [t for t in trials if t["model"] in names]
    if args.langs:
        names = {n.strip() for n in args.langs.split(",") if n.strip()}
        trials = [t for t in trials if t["lang"] in names]

    if args.limit:
        trials = trials[: args.limit]

    if not trials:
        print("no matching trials.")
        return 0

    for t in trials:
        _print_trial(t, judge_index.get(t["request_id"]))

    print(f"[{len(trials)} trial(s) shown]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
