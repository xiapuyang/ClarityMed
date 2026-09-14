"""``claritymed bench …`` — backfill helpers for benchmark output dirs.

The main use case today is uploading historical bench runs to Phoenix
Experiments after the upload pipeline lands. The bench runners
themselves upload automatically on completion (when Phoenix is wired);
this command is for past runs sitting under ``data/bench/`` that
predate that wiring, or for re-running an upload after an endpoint
change.

The CLI deliberately lives in ``cli/commands/`` next to ``prompts.py``
even though the implementation reaches into ``tests/benchmarks/`` — the
uploader logic is bench-runner-adjacent because the trial schema is
defined there. Pulling it into ``src/`` would force a circular import
of the runner's case definitions, which the uploader already avoids by
re-deriving identity from ``trials.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

bench_app = typer.Typer(
    name="bench",
    help="Benchmark output utilities (Phoenix upload backfill, etc.).",
    no_args_is_help=True,
)

# Mirrors ``cases_baseline.RUNNERS`` — kept hard-coded here so the
# Typer choices help string doesn't depend on a heavy import path.
_RUNNERS = ("ingest", "symptoms", "vision")


@bench_app.command("upload")
def upload(
    run_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Path to a bench run dir, e.g. data/bench/ingest/20260617_142233_001",
    ),
) -> None:
    """Upload a completed bench run's results to Phoenix Experiments.

    Reads ``manifest.json`` + ``trials.jsonl`` under ``run_dir``. Local
    files are never modified. Skips silently only when no Phoenix
    endpoint is configured (``tracing.endpoint`` in ``configs/app.yaml``
    or the ``PHOENIX_COLLECTOR_ENDPOINT`` env var). When an endpoint
    IS configured, a 2 s reachability probe runs before any heavy work
    so a downed Phoenix fails fast instead of burning the full httpx
    timeout chain.

    Exit codes:
      0  upload completed (possibly with non-fatal per-run errors
         printed under "! [stage] detail" lines)
      1  upload was skipped because no Phoenix endpoint is configured
      2  unrecoverable: Phoenix unreachable, manifest missing or
         unparseable, or client init failed. Local data/bench/ files
         are unaffected; fix the issue and re-run this command.

    Legacy run dirs from before the manifest layer (no ``manifest.json``)
    are NOT supported — re-run the benchmark to generate one.
    """
    # Imported lazily so ``claritymed --help`` doesn't pay for the
    # bench module's heavy imports on every CLI invocation.
    from tests.benchmarks.tool_invoke import phoenix_upload

    try:
        result = phoenix_upload.upload_run(run_dir)
    except phoenix_upload.PhoenixUnreachable as exc:
        # Configuration explicitly asks for Phoenix but the server
        # isn't answering. Exit non-zero so scripts / CI pipelines
        # surface the problem instead of pretending the upload worked.
        typer.echo(f"❌ phoenix: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except FileNotFoundError as exc:
        # Missing manifest.json — almost always means the user pointed
        # at a pre-this-PR run dir. Legacy backfill isn't supported;
        # re-running the bench is the only path.
        typer.echo(f"❌ {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if result.skipped:
        typer.echo(f"phoenix: skipped ({result.skip_reason})")
        raise typer.Exit(code=1)

    n_runs = sum(e.n_runs for e in result.experiments)
    n_evals = sum(e.n_evals for e in result.experiments)
    typer.echo(
        f"phoenix: dataset={result.dataset_name} "
        f"experiments={len(result.experiments)} runs={n_runs} evals={n_evals} "
        f"({result.endpoint})"
    )
    for exp in result.experiments:
        url = exp.experiment_url or "(no url)"
        typer.echo(f"  {exp.cell_key:<40} runs={exp.n_runs:<3} {url}")

    has_init_error = any(e.stage in ("client_init", "manifest") for e in result.errors)
    for err in result.errors:
        # Error lines go to stderr so the success summary on stdout
        # stays parseable by agents capturing experiment URLs.
        typer.echo(f"  ! [{err.stage}] {err.detail}", err=True)
    if has_init_error:
        raise typer.Exit(code=2)


@bench_app.command("manifest")
def show_manifest(
    run_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Path to a bench run dir.",
    ),
) -> None:
    """Print the manifest.json fields for a run dir (human-readable summary)."""
    from tests.benchmarks.tool_invoke.manifest import read_manifest

    manifest = read_manifest(run_dir)
    typer.echo(f"runner:        {manifest.runner}")
    typer.echo(f"bench_ts:      {manifest.bench_ts}")
    typer.echo(f"commit_sha:    {manifest.commit_sha or '(none)'}")
    typer.echo(f"started:       {manifest.started_at}")
    typer.echo(f"finished:      {manifest.finished_at}")
    typer.echo(
        f"cases:         {manifest.cases.count} cases "
        f"(tiers={','.join(manifest.cases.tiers)}, "
        f"sha256={manifest.cases.content_sha256[:12]})"
    )
    if manifest.prompt_versions:
        typer.echo("prompts:")
        for name, version in sorted(manifest.prompt_versions.items()):
            typer.echo(f"  {name:<40} {version}")
    else:
        typer.echo("prompts:       (none captured)")
    cfg = manifest.config
    tpl: list[str] | None = cfg.tool_prompt_langs
    tpl_str = ",".join(tpl) if tpl else "-"
    typer.echo(
        f"config:        models={','.join(cfg.models)} "
        f"user_langs={','.join(cfg.user_langs)} "
        f"tool_prompt_langs={tpl_str} "
        f"trials={cfg.trials}"
    )


@bench_app.command("case")
def show_case(
    runner: str = typer.Argument(
        ...,
        help=f"Runner name; one of: {', '.join(_RUNNERS)}",
    ),
    case_id: str = typer.Argument(
        ...,
        help="Case identifier in the form '<name>@v<revision>', e.g. save_allergy@v2",
    ),
    output_format: str = typer.Option(
        "pretty",
        "--format",
        "-f",
        help="Output format: 'pretty' (human-readable) or 'json' (raw dataset example).",
    ),
) -> None:
    """Look up the historical content of a case from Phoenix.

    Walks ``tool_invoke.<runner>`` dataset versions newest-first until
    an example with matching ``metadata.case_id`` is found. Prints the
    prompts dict, predicate source, expected behaviour, etc. as
    captured at the time that revision was uploaded.

    Limitations:
      - Phoenix must be configured (tracing.endpoint in app.yaml or
        PHOENIX_COLLECTOR_ENDPOINT env var). Exit 1 otherwise.
      - A revision that was never uploaded (e.g. bench ran with
        --no-phoenix-upload) is unrecoverable; exit 2.
    """
    if runner not in _RUNNERS:
        typer.echo(
            f"unknown runner {runner!r}; expected one of: {', '.join(_RUNNERS)}",
            err=True,
        )
        raise typer.Exit(code=2)

    from tests.benchmarks.tool_invoke import case_history

    try:
        result = case_history.lookup_case(runner=runner, case_id=case_id)
    except case_history.CaseLookupError as exc:
        msg = str(exc)
        # No-endpoint case → exit 1 (not configured); not-found → exit 2.
        code = 1 if "no Phoenix endpoint" in msg else 2
        typer.echo(f"case-history: {msg}", err=True)
        raise typer.Exit(code=code) from exc

    if output_format == "json":
        typer.echo(
            json.dumps(
                {
                    "case_id": result.case_id,
                    "dataset_name": result.dataset_name,
                    "dataset_version_id": result.dataset_version_id,
                    "example_id": result.example_id,
                    "input": result.input,
                    "output": result.output,
                    "metadata": result.metadata,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    inp = result.input
    out = result.output
    typer.echo(f"case_id:        {result.case_id}")
    typer.echo(f"dataset:        {result.dataset_name}")
    typer.echo(f"version_id:     {result.dataset_version_id}")
    typer.echo(f"tier:           {inp.get('tier', '(unknown)')}")
    typer.echo(f"expected:       {inp.get('expected_behavior', '(unknown)')}")
    expected_tool = out.get("expected_tool")
    expected_tools = out.get("expected_tools") or []
    if expected_tool:
        typer.echo(f"expected_tool:  {expected_tool}")
    if expected_tools:
        typer.echo(f"expected_tools: {', '.join(expected_tools)}")
    prompts = inp.get("prompts") or {}
    if prompts:
        typer.echo("prompts:")
        for lang in sorted(prompts):
            typer.echo(f"  [{lang}] {prompts[lang]}")
    elif inp.get("user_prompt"):
        # Legacy trial-fallback record (no prompts template captured).
        typer.echo(f"user_prompt:    {inp['user_prompt']}")
        typer.echo(
            "  (note: only the formatted user_prompt is archived — this "
            "run pre-dates cases_snapshot.json)"
        )
    src = inp.get("args_predicate_src")
    if src:
        typer.echo("args_predicate_src:")
        for line in src.splitlines():
            typer.echo(f"  {line}")
    seed_src = inp.get("seed_src")
    if seed_src:
        typer.echo("seed_src:")
        for line in seed_src.splitlines():
            typer.echo(f"  {line}")


__all__ = ["bench_app"]
