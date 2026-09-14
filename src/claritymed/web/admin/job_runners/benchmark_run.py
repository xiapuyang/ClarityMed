"""Benchmark runner — kicks ``claritymed.evals.runners.<runner>`` as a
subprocess and streams the tail of its stdout into the JobRegistry.

Subprocess (not asyncio task) for two reasons:
1. Eval runs hold a large model in memory; isolating them in a child
   process lets the web worker reclaim memory cleanly on completion.
2. Cancel via ``proc.terminate()`` is far more reliable than
   ``asyncio.Task.cancel()`` when the runner spends most of its time
   inside C extensions that don't yield.

Run params:

* ``runner`` — module path under ``claritymed.evals.runners`` (default:
  ``lm_eval_runner``).
* ``dataset`` — passed as ``--dataset``.
* ``provider_id`` — passed as ``--provider``.
* ``sample_size`` — passed as ``--sample-size`` (optional).
* ``extra_args`` — list of additional CLI args, appended verbatim.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claritymed.web.admin.jobs import JobRegistry, JobSpec

logger = logging.getLogger(__name__)

CANCEL_GRACE_S = 5.0


async def run(spec: "JobSpec", registry: "JobRegistry") -> None:
    params = spec.params
    runner = params.get("runner", "lm_eval_runner")
    args = _build_args(params)

    registry.update(spec.id, progress=f"launching {runner}")

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        f"claritymed.evals.runners.{runner}",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    try:
        await _stream_stdout(proc, spec, registry)
        rc = await proc.wait()
    except asyncio.CancelledError:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=CANCEL_GRACE_S)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        raise
    if rc != 0:
        registry.update(spec.id, exit_code=rc)
        raise RuntimeError(f"benchmark runner exited with {rc}")
    registry.update(spec.id, exit_code=rc, progress="done")


def _build_args(params: dict) -> list[str]:
    args: list[str] = []
    dataset = params.get("dataset")
    if dataset:
        args.extend(["--dataset", str(dataset)])
    provider_id = params.get("provider_id")
    if provider_id:
        args.extend(["--provider", str(provider_id)])
    sample_size = params.get("sample_size")
    if sample_size is not None:
        args.extend(["--sample-size", str(sample_size)])
    extra = params.get("extra_args") or []
    args.extend(str(a) for a in extra)
    return args


async def _stream_stdout(proc, spec, registry) -> None:
    assert proc.stdout is not None
    while True:
        chunk = await proc.stdout.readline()
        if not chunk:
            return
        try:
            line = chunk.decode("utf-8", errors="replace").rstrip()
        except Exception:  # noqa: BLE001
            continue
        if line:
            registry.append_stdout(spec.id, line)
