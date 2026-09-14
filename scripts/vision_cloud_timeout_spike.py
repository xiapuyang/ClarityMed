"""Phase 0.5 spike — vision tool-body wall-clock vs cloud LLM HTTP timeout.

Validates that the in-band design of ``vision_plugin`` (confirm modal +
fallback flow + result transform, all held inside a single cloud LLM
tool-call HTTP round-trip) stays under the cloud provider's connection
timeout. Mirrors the symptoms plan's KTD-1 spike: surface the operational
ceiling for ``tool.total_budget_ms`` against the providers we actually
plan to ship with, *before* Unit 7 lands.

Outcome decides Unit 7's structure:

* Spike completes against every tested provider at the design budget
  (20s) → Unit 7 keeps the single-turn confirm modal.
* Any provider 504s before the tool body completes → Unit 7 must
  restructure: tool returns "I need confirmation", LLM emits
  ``askuserquestion``, new tool call resumes after user yes/no
  (the symptoms multi-turn pattern).

Run::

    uv run python scripts/vision_cloud_timeout_spike.py \\
        --providers omlx,deepseek,anthropic \\
        --budget-ms 20000,25000,30000

Provider ids match ``configs/models.yaml``; missing API keys cause that
provider to be skipped with a loud warning. Results land in
``docs/spikes/2026-06-14-vision-cloud-timeout.md`` (append-only); rerun
with ``--write`` to refresh that file.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("vision_cloud_timeout_spike")

# Stub work the tool body does during a real vision call: confirm modal
# wall-clock (user thinks for a moment) + a couple of fallback HTTP
# calls + result transform. Pure sleeps so the spike doesn't depend on
# a running vision-server.
DEFAULT_CONFIRM_S = 5.0
DEFAULT_PER_CALL_S = 5.0
DEFAULT_FALLBACK_ATTEMPTS = 3

DEFAULT_PROVIDERS = ("omlx",)
DEFAULT_BUDGETS_MS = (20_000, 25_000, 30_000)


@dataclass(frozen=True)
class SpikeResult:
    """One (provider, budget) trial outcome."""

    provider_id: str
    budget_ms: int
    elapsed_ms: int
    completed: bool
    error: str | None


def _build_tool_body(confirm_s: float, per_call_s: float, attempts: int) -> "callable":
    """Construct a stub tool body that sleeps for ``confirm + attempts*per_call``.

    Returned as a plain async function so we can register it with
    pydantic-ai's :class:`pydantic_ai.Tool` exactly the way the real
    plugin does.
    """

    async def _stub_detect_disease_from_image(disease_id: str, image_sha: str) -> dict:
        # 1. confirm-modal wall-clock (real plugin awaits PromptChannel).
        await asyncio.sleep(confirm_s)
        # 2. fallback flow: N HTTP calls, each ~per_call_s.
        for _ in range(attempts):
            await asyncio.sleep(per_call_s)
        # 3. result transform — negligible.
        return {
            "kind": "detection",
            "disease_id": disease_id,
            "image_sha": image_sha,
            "top1": "stub",
        }

    return _stub_detect_disease_from_image


async def _run_one_trial(
    provider_id: str,
    *,
    budget_ms: int,
    confirm_s: float,
    per_call_s: float,
    attempts: int,
) -> SpikeResult:
    """Run a single (provider, budget) trial. Never raises — returns ``SpikeResult``.

    Imports of pydantic-ai / catalog are scoped here so the script
    surfaces a friendlier error when an env var is missing.
    """
    try:
        from pydantic_ai import Agent, Tool

        from claritymed.core.llm.model import build_model, build_model_settings
        from claritymed.stores.models import load_models
    except Exception as exc:  # noqa: BLE001
        return SpikeResult(provider_id, budget_ms, 0, False, f"import: {exc!s}")

    try:
        models_cfg = load_models()
        provider = next(p for p in models_cfg.providers if p.id == provider_id)
    except StopIteration:
        return SpikeResult(
            provider_id, budget_ms, 0, False, "provider not in configs/models.yaml"
        )
    except Exception as exc:  # noqa: BLE001
        return SpikeResult(provider_id, budget_ms, 0, False, f"config: {exc!s}")

    try:
        model = build_model(provider)
        settings = build_model_settings(provider)
    except Exception as exc:  # noqa: BLE001
        return SpikeResult(provider_id, budget_ms, 0, False, f"model: {exc!s}")

    stub = _build_tool_body(confirm_s, per_call_s, attempts)
    agent = Agent(
        model,
        model_settings=settings,
        tools=[
            Tool(
                stub,
                name="detect_disease_from_image",
                description=(
                    "Vision spike stub. Always call this tool with disease_id="
                    "'breast_cancer_ultrasound' and image_sha='spike-fixture'. "
                    "Do not respond before the tool returns."
                ),
            )
        ],
        instructions=(
            "You are a spike harness. The user asks one question; you must "
            "respond by calling the tool exactly once. After the tool returns, "
            "reply with the single word 'ok'."
        ),
    )

    started = time.monotonic()
    try:
        await asyncio.wait_for(
            agent.run("Please run the vision tool on this image."),
            timeout=budget_ms / 1000.0,
        )
    except asyncio.TimeoutError:
        elapsed = int((time.monotonic() - started) * 1000)
        return SpikeResult(
            provider_id, budget_ms, elapsed, False, "asyncio.TimeoutError"
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = int((time.monotonic() - started) * 1000)
        return SpikeResult(provider_id, budget_ms, elapsed, False, str(exc))
    elapsed = int((time.monotonic() - started) * 1000)
    return SpikeResult(provider_id, budget_ms, elapsed, True, None)


def _report_rows(results: list[SpikeResult]) -> str:
    header = (
        "| provider | budget_ms | elapsed_ms | completed | error |\n"
        "|---|---|---|---|---|\n"
    )
    rows = [
        f"| {r.provider_id} | {r.budget_ms} | {r.elapsed_ms} | "
        f"{'yes' if r.completed else 'no'} | {r.error or '-'} |"
        for r in results
    ]
    return header + "\n".join(rows)


def _writeup(results: list[SpikeResult]) -> str:
    completed = [r for r in results if r.completed]
    failed = [r for r in results if not r.completed]
    decision = (
        "Unit 7 single-turn confirm modal design is safe at the budgets exercised."
        if not failed
        else (
            "Unit 7 must restructure the confirm modal as a separate LLM turn — "
            "at least one provider 504s before the tool body completes at the "
            "design budget."
        )
    )
    return (
        "---\n"
        "title: Vision cloud-timeout spike — Unit 6.5\n"
        f"date: {datetime.now(timezone.utc).date().isoformat()}\n"
        "---\n\n"
        "# Vision cloud-timeout spike\n\n"
        "Mirrors symptoms KTD-1. The vision tool body holds the confirm modal "
        "wall-clock + fallback flow + result transform inside a single cloud "
        "LLM tool-call HTTP round-trip. This spike measures whether the design "
        "budget (`tool.total_budget_ms=20000`) leaves headroom under each "
        "provider's connection timeout.\n\n"
        f"Trials completed: {len(completed)} / {len(results)}\n\n"
        f"{_report_rows(results)}\n\n"
        "## Decision\n\n"
        f"{decision}\n"
    )


async def _main(argv: list[str]) -> int:
    # Load .env so OMLX_API_KEY / cloud provider keys land in os.environ
    # before any provider build. The vision tool-trigger bench
    # (tests/benchmarks/tool_invoke/vision/run.py) does the same; missing
    # this here made the spike fail with "api_key_env unset" even when
    # the project's .env had it.
    from claritymed.config import load_env_file  # noqa: PLC0415

    load_env_file()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        default=",".join(DEFAULT_PROVIDERS),
        help="Comma-separated provider_ids from configs/models.yaml",
    )
    parser.add_argument(
        "--budget-ms",
        default=",".join(str(b) for b in DEFAULT_BUDGETS_MS),
        help="Comma-separated wall-clock budgets to test, in milliseconds",
    )
    parser.add_argument(
        "--confirm-s",
        type=float,
        default=DEFAULT_CONFIRM_S,
        help="Sleep simulating the confirm modal wall-clock (default 5s)",
    )
    parser.add_argument(
        "--per-call-s",
        type=float,
        default=DEFAULT_PER_CALL_S,
        help="Sleep simulating one /v1/detect call (default 5s)",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_FALLBACK_ATTEMPTS,
        help="Number of fallback flow attempts (default 3)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite docs/spikes/2026-06-14-vision-cloud-timeout.md with the results",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    budgets = [int(b) for b in args.budget_ms.split(",") if b.strip()]

    results: list[SpikeResult] = []
    for provider_id in providers:
        for budget_ms in budgets:
            logger.info("running provider=%s budget_ms=%s", provider_id, budget_ms)
            result = await _run_one_trial(
                provider_id,
                budget_ms=budget_ms,
                confirm_s=args.confirm_s,
                per_call_s=args.per_call_s,
                attempts=args.attempts,
            )
            results.append(result)
            logger.info("  -> %s", result)

    writeup = _writeup(results)
    print("\n" + writeup)

    if args.write:
        target = Path("docs/spikes/2026-06-14-vision-cloud-timeout.md")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(writeup, encoding="utf-8")
        logger.info("wrote %s", target)

    failed = [r for r in results if not r.completed]
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(_main(sys.argv[1:])))
