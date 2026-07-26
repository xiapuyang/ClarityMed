"""Latency bench for the symptoms tool + surrounding LLM path.

Measures three wall-clock latencies a user actually experiences:

1. **LLM baseline** — a bare non-tool prompt through the local
   provider. Establishes "how fast does the base LLM respond when it
   doesn't touch any tool". Skippable when the provider isn't reachable.

2. **Symptom session start** — POST to the symptoms server on port
   8084 with a symptom complaint. Includes SapBERT init matching +
   first-question rendering. This is the "user typed a symptom → the
   tool opens with question 1" wall-clock.

3. **Per-turn latency** — POST /turn after an answer. Server does
   IG next_action + reveal + diagnose + question render. Loop N turns
   (or until session terminates). This is the "user tapped an answer
   → next question appears" wall-clock.

Requires: symptoms server running on ``http://127.0.0.1:8084`` (start with
``uv run --extra symptoms-server claritymed-symptoms-server``). LLM
provider is optional — passing ``--skip-llm`` skips metric 1.

All metrics reported as p50 / p90 / p99 across N repeats. Writes JSON
to ``data/bench/symptoms/latency_<timestamp>.json``.

Usage::

    uv run --extra symptoms-server python scripts/bench_symptoms_latency.py \\
        --dataset ddxplus_pneumonia_flu --repeats 20 --provider omlx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import httpx

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "bench" / "symptoms"

DEFAULT_SYMPTOMS_URL = "http://127.0.0.1:8084"
DEFAULT_DATASET_ID = "ddxplus_pneumonia_flu"
DEFAULT_COMPLAINT = "I've had a high fever, chills and productive cough for three days."
DEFAULT_SUMMARY = (
    "3-day history of high fever, chills, and productive cough — respiratory infection"
)
DEFAULT_LLM_PROMPT = "In two sentences, explain what type 2 diabetes is."


@dataclass
class Percentiles:
    n: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def compute(cls, samples_ms: list[float]) -> "Percentiles":
        s = sorted(samples_ms)
        n = len(s)
        return cls(
            n=n,
            mean_ms=statistics.mean(s),
            p50_ms=_percentile(s, 50),
            p90_ms=_percentile(s, 90),
            p99_ms=_percentile(s, 99),
            min_ms=s[0],
            max_ms=s[-1],
        )


def _percentile(sorted_samples: list[float], p: int) -> float:
    if not sorted_samples:
        return float("nan")
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    # Linear interpolation between closest ranks — matches numpy default.
    k = (len(sorted_samples) - 1) * (p / 100)
    lo = int(k)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = k - lo
    return sorted_samples[lo] + frac * (sorted_samples[hi] - sorted_samples[lo])


# ---------------------------------------------------------------------------
# Metric 1: LLM baseline (bare prompt through pydantic-ai Agent)
# ---------------------------------------------------------------------------


async def bench_llm_baseline(
    provider_id: str, prompt: str, repeats: int
) -> list[float]:
    """One Agent.run per repeat, no tools, no RAG, no system prompt.

    Uses the same ``build_model`` production uses so the numbers reflect
    what a stripped-down turn would take against the same provider the
    orchestrator uses. Warmup call excluded from the reported latencies.
    """
    from pydantic_ai import Agent

    from claritymed.core.llm.model import build_model
    from claritymed.stores.models import resolve_provider

    provider = resolve_provider(override=provider_id)
    model = build_model(provider)
    agent: Agent = Agent(model)

    print(f"[llm] warmup call to {provider.id} ({provider.model})...")
    await agent.run(prompt)  # discard warmup

    latencies_ms: list[float] = []
    for i in range(repeats):
        t0 = time.perf_counter()
        result = await agent.run(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Log first response length so silent-empty-completion regressions
        # are visible when comparing runs later.
        text_len = len(getattr(result, "output", "") or "")
        latencies_ms.append(elapsed_ms)
        print(f"[llm] {i + 1}/{repeats}  {elapsed_ms:.0f}ms  (output={text_len} chars)")
    return latencies_ms


# ---------------------------------------------------------------------------
# Metric 2: symptom session start (POST /sessions)
# ---------------------------------------------------------------------------


def bench_session_start(
    base_url: str,
    dataset_id: str,
    complaint: str,
    summary: str,
    repeats: int,
) -> list[float]:
    """POST /sessions — the "prompt → first question" server wall-clock.

    Includes SapBERT init matching (~440MB model, CPU-warm after first
    call), first IG pick or first init-matcher question render, and the
    canonical question payload serialization. Each repeat creates a
    brand-new session so state doesn't cross contaminate.
    """
    endpoint = f"{base_url}/v1/datasets/{dataset_id}/sessions"
    body = {
        "complaint": complaint,
        "profile": {"age_years": 45, "sex": "M"},
        "language": "en",
        "symptom_summary": summary,
    }

    latencies_ms: list[float] = []
    with httpx.Client(timeout=60.0) as client:
        # Warmup — SapBERT init matcher lazy-loads on first call.
        print(f"[start] warmup POST {endpoint}...")
        r = client.post(endpoint, json=body)
        r.raise_for_status()

        for i in range(repeats):
            t0 = time.perf_counter()
            r = client.post(endpoint, json=body)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            r.raise_for_status()
            _ = r.json()["session_id"]  # sanity check
            latencies_ms.append(elapsed_ms)
            print(f"[start] {i + 1}/{repeats}  {elapsed_ms:.0f}ms")
    return latencies_ms


# ---------------------------------------------------------------------------
# Metric 3: per-turn latency (POST /sessions/{sid}/turn)
# ---------------------------------------------------------------------------


def _next_answer_body(question: dict) -> dict:
    """Build a schema-valid answer for whatever kind of question we got.

    The server rejects "No" on non-binary questions (500), so we can't
    hardcode a single answer. Strategy: pick the first option's
    ``value`` and send it via ``answer_value`` — that bypasses locale
    label matching and works for binary / categorical / multi. For
    numeric questions, send the low bound.
    """
    numeric = question.get("numeric")
    if numeric:
        # Send low bound as a float — always valid.
        lo = float(numeric.get("min", 0))
        return {"answer": lo, "language": "en"}
    options = question.get("options") or []
    if not options:
        # Fallback for unusual shapes — server will 500 and we'll log.
        return {"answer": "No", "language": "en"}
    first = options[0]
    value = first.get("value")
    label = first.get("label", "")
    if question.get("multi_select"):
        return {
            "answer": [label],
            "answer_value": [value] if value else None,
            "language": "en",
        }
    return {
        "answer": label,
        "answer_value": value,
        "language": "en",
    }


def bench_turn(
    base_url: str,
    dataset_id: str,
    complaint: str,
    summary: str,
    n_sessions: int,
    max_turns_per_session: int,
) -> list[float]:
    """Per-turn wall-clock. Opens N sessions, runs up to M turns each.

    Session termination shortens turn count for that session (server
    returns ``done=true``). Answer is picked from the current question's
    first option (or the low bound for numeric) — schema-valid across
    binary/categorical/multi. The point is round-trip latency, not
    diagnostic accuracy.
    """
    start_endpoint = f"{base_url}/v1/datasets/{dataset_id}/sessions"
    body = {
        "complaint": complaint,
        "profile": {"age_years": 45, "sex": "M"},
        "language": "en",
        "symptom_summary": summary,
    }

    all_turn_ms: list[float] = []
    with httpx.Client(timeout=60.0) as client:
        for s in range(n_sessions):
            r = client.post(start_endpoint, json=body)
            r.raise_for_status()
            resp = r.json()
            sid = resp["session_id"]
            current_question = resp["first_question"]
            turn_endpoint = f"{base_url}/v1/datasets/{dataset_id}/sessions/{sid}/turn"

            session_turns: list[float] = []
            done = False
            error_at: int | None = None
            for t in range(max_turns_per_session):
                answer_body = _next_answer_body(current_question)
                t0 = time.perf_counter()
                r = client.post(turn_endpoint, json=answer_body)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if r.status_code == 404:
                    print(f"[turn] session {sid[:8]} vanished at turn {t + 1}")
                    break
                if r.status_code >= 400:
                    error_at = t + 1
                    print(
                        f"[turn] {r.status_code} at turn {t + 1} "
                        f"(answer_body keys={list(answer_body.keys())}) — "
                        f"session {sid[:8]} aborted"
                    )
                    break
                session_turns.append(elapsed_ms)
                payload = r.json()
                done = bool(payload.get("done"))
                if done:
                    break
                current_question = payload.get("next_question") or payload.get(
                    "question"
                )
                if current_question is None:
                    print(
                        f"[turn] no next_question in response at turn {t + 1}, stopping"
                    )
                    break
            all_turn_ms.extend(session_turns)
            reason = (
                "done"
                if done
                else ("error@" + str(error_at) if error_at else "maxstep")
            )
            mean_str = (
                f"{statistics.mean(session_turns):.0f}ms"
                if session_turns
                else "(no turns)"
            )
            print(
                f"[turn] session {s + 1}/{n_sessions} "
                f"({sid[:8]}): {len(session_turns)} turns, "
                f"mean {mean_str}, terminated={reason}"
            )
    return all_turn_ms


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_summary(label: str, p: Percentiles) -> None:
    print(
        f"\n  [{label}]  n={p.n}  "
        f"mean={p.mean_ms:.0f}ms  p50={p.p50_ms:.0f}ms  "
        f"p90={p.p90_ms:.0f}ms  p99={p.p99_ms:.0f}ms  "
        f"(min={p.min_ms:.0f}, max={p.max_ms:.0f})"
    )


def _print_histogram(label: str, samples_ms: list[float]) -> None:
    if not samples_ms:
        return
    hi = max(samples_ms)
    lo = min(samples_ms)
    n_bins = 10
    if hi == lo:
        print(f"  {label} histogram: all samples at {lo:.0f}ms")
        return
    width = (hi - lo) / n_bins
    bins = [0] * n_bins
    for x in samples_ms:
        idx = min(int((x - lo) / width), n_bins - 1)
        bins[idx] += 1
    peak = max(bins) or 1
    for i, count in enumerate(bins):
        bar_lo = lo + i * width
        bar_hi = bar_lo + width
        bar = "#" * int(round(count * 40 / peak))
        print(f"    [{bar_lo:>5.0f}, {bar_hi:>5.0f})ms  n={count:>3d}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--symptoms-url",
        default=DEFAULT_SYMPTOMS_URL,
        help="Base URL of the running symptoms server.",
    )
    ap.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument(
        "--turn-sessions",
        type=int,
        default=5,
        help="Number of sessions to open for per-turn latency measurement.",
    )
    ap.add_argument(
        "--turn-max-per-session",
        type=int,
        default=18,
        help="Max turns per session (matches production maxstep).",
    )
    ap.add_argument("--complaint", default=DEFAULT_COMPLAINT)
    ap.add_argument("--symptom-summary", default=DEFAULT_SUMMARY)
    ap.add_argument("--llm-prompt", default=DEFAULT_LLM_PROMPT)
    ap.add_argument(
        "--provider",
        default="omlx",
        help="LLM provider id for metric 1. Ignored when --skip-llm is set.",
    )
    ap.add_argument("--skip-llm", action="store_true", help="Skip metric 1.")
    ap.add_argument("--output", type=pathlib.Path, default=None)
    args = ap.parse_args()

    print(f"[bench] symptoms server: {args.symptoms_url}")
    print(f"[bench] dataset: {args.dataset}")
    print(f"[bench] repeats: {args.repeats}   turn sessions: {args.turn_sessions}")

    llm_ms: list[float] = []
    if not args.skip_llm:
        print(
            f"\n=== Metric 1: LLM baseline ({args.provider}, {args.repeats} calls) ==="
        )
        try:
            llm_ms = asyncio.run(
                bench_llm_baseline(args.provider, args.llm_prompt, args.repeats)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[llm] SKIPPED — {type(exc).__name__}: {exc}")
            llm_ms = []
    else:
        print("\n=== Metric 1: LLM baseline — SKIPPED (--skip-llm) ===")

    print(f"\n=== Metric 2: Symptom session start ({args.repeats} calls) ===")
    start_ms = bench_session_start(
        args.symptoms_url,
        args.dataset,
        args.complaint,
        args.symptom_summary,
        args.repeats,
    )

    print(
        f"\n=== Metric 3: Per-turn latency "
        f"({args.turn_sessions} sessions × ≤{args.turn_max_per_session} turns) ==="
    )
    turn_ms = bench_turn(
        args.symptoms_url,
        args.dataset,
        args.complaint,
        args.symptom_summary,
        args.turn_sessions,
        args.turn_max_per_session,
    )

    print("\n\n=== Summary ===")
    llm_p = Percentiles.compute(llm_ms) if llm_ms else None
    start_p = Percentiles.compute(start_ms)
    turn_p = Percentiles.compute(turn_ms)
    if llm_p is not None:
        _print_summary("LLM baseline", llm_p)
    else:
        print("  [LLM baseline]  skipped")
    _print_summary("Symptom session start", start_p)
    _print_summary("Per-turn latency", turn_p)

    print("\n  Symptom session start histogram:")
    _print_histogram("start", start_ms)
    print("\n  Per-turn latency histogram:")
    _print_histogram("turn", turn_ms)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or (
        OUT_DIR / f"latency_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symptoms_url": args.symptoms_url,
        "dataset_id": args.dataset,
        "provider": args.provider if not args.skip_llm else None,
        "repeats": args.repeats,
        "turn_sessions": args.turn_sessions,
        "turn_max_per_session": args.turn_max_per_session,
        "complaint": args.complaint,
        "llm_prompt": args.llm_prompt,
        "metric_llm_baseline": {
            "samples_ms": llm_ms,
            "percentiles": asdict(llm_p) if llm_p is not None else None,
        },
        "metric_session_start": {
            "samples_ms": start_ms,
            "percentiles": asdict(start_p),
        },
        "metric_per_turn": {
            "samples_ms": turn_ms,
            "percentiles": asdict(turn_p),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
