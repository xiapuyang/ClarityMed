"""Upload one completed tool_invoke bench run to Phoenix.

After a runner writes ``data/bench/<runner>/<ts>/``, this module:

1. Builds a Phoenix Dataset (one per runner, e.g. ``tool_invoke.ingest``)
   with one example per ``(case_name, revision)`` pair from the
   manifest. Phoenix versions the dataset by content hash — re-uploading
   with unchanged cases is a no-op, adding/changing a case produces a
   new dataset version automatically.

2. For each ``(model, user_lang, tool_prompt_lang)`` cell present in
   ``trials.jsonl`` creates one Experiment bound to that dataset
   version. The experiment's metadata carries the full
   ``prompt_versions`` dict + commit sha so a comparison can be filtered
   to "all experiments where prompt X was version N".

3. Per experiment, creates one Run per ``(case, trial_idx)`` and one
   evaluation per run (``predicate_pass`` score 0/1). Per-trial
   granularity is preserved so the Phoenix UI surfaces flakiness
   (case passes 2/3 trials) instead of just the mean.

The bench runner's local outputs (``trials.jsonl`` / ``summary.csv`` /
``report.html``) are read but never modified — Phoenix is a parallel
sink, the JSONL stays canonical for offline tooling.

Phoenix endpoint resolution
---------------------------

The endpoint comes from the same source the rest of the codebase uses:
``configs/app.yaml`` ``tracing.endpoint`` (when ``tracing.enabled``) or
the ``PHOENIX_COLLECTOR_ENDPOINT`` env var. When neither is set
``upload_run`` returns an ``UploadResult`` marked ``skipped=True`` so
the runner can print "phoenix: not configured" without raising.

Error handling
--------------

Every Phoenix HTTP call is wrapped so a remote outage / 4xx /
unreachable host doesn't break the bench. Failures populate
``UploadResult.errors`` and the runner logs but does not exit non-zero
— the local JSONL is the source of truth, Phoenix is enhancement.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from tests.benchmarks.tool_invoke.cases_snapshot import (
    CasesSnapshot,
    read_snapshot,
)
from tests.benchmarks.tool_invoke.manifest import (
    MANIFEST_FILENAME,
    RunManifest,
    read_manifest,
)

logger = logging.getLogger(__name__)

# Phoenix REST endpoints. We talk directly here for experiment runs +
# evaluations because the high-level ``client.experiments.run_experiment``
# drives the task itself — which doesn't fit a "task already ran offline"
# model. Datasets *do* go through the high-level client because the
# wrapper handles the content-versioning math for us.
_EXPERIMENT_RUNS_PATH = "v1/experiments/{experiment_id}/runs"
_EXPERIMENT_EVALS_PATH = "v1/experiment_evaluations"
_DATASET_EXPERIMENT_PATH = "v1/datasets/{dataset_id}/experiments"

# Evaluator names that appear on the Phoenix Experiment runs page. Pick
# stable spellings — renaming later means losing aggregation continuity.
_EVAL_PREDICATE = "predicate_pass"

# How long the pre-flight reachability probe will wait for Phoenix to
# answer before declaring it down. Kept short so a user who forgot to
# start Phoenix gets feedback in seconds, not the 30s default httpx
# would otherwise burn through on every blocked call in sequence.
_REACHABILITY_TIMEOUT_S = 2.0


class PhoenixUnreachable(RuntimeError):
    """Raised when the configured Phoenix endpoint doesn't respond to
    the pre-flight probe.

    The caller is expected to surface this loudly — the configuration
    says Phoenix is expected, so silently skipping the upload would
    bury the misconfiguration. Local bench files are unaffected; the
    user fixes Phoenix and re-runs ``claritymed bench upload <dir>``.
    """


# --- public datatypes ------------------------------------------------


@dataclass
class UploadError:
    """One failure step from an upload attempt."""

    stage: str  # "manifest" | "dataset" | "experiment" | "run" | "evaluation"
    detail: str


@dataclass
class UploadedExperiment:
    """One Phoenix experiment that was created (or attempted) for a cell."""

    cell_key: str  # "model__u_<ulang>__t_<tlang>" stable identifier
    experiment_id: Optional[str]
    experiment_url: Optional[str]
    n_runs: int
    n_evals: int


@dataclass
class UploadResult:
    """Per-call summary the bench runner prints + the CLI returns."""

    skipped: bool = False
    skip_reason: Optional[str] = None
    endpoint: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_version_id: Optional[str] = None
    experiments: list[UploadedExperiment] = field(default_factory=list)
    errors: list[UploadError] = field(default_factory=list)


# --- endpoint resolution --------------------------------------------


def resolve_endpoint() -> Optional[str]:
    """Return the Phoenix base URL, or ``None`` if Phoenix isn't wired.

    Honours the same precedence as ``core.observability.tracing``:
    ``configs/app.yaml`` ``tracing.endpoint`` (when enabled) > the
    ``PHOENIX_COLLECTOR_ENDPOINT`` env var. Returns ``None`` when
    neither side is configured so callers can short-circuit.
    """
    env = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "").strip()
    try:
        from claritymed.core.observability.tracing import _load_config

        cfg = _load_config()
    except Exception:  # noqa: BLE001
        # App config not loadable in this context (e.g. minimal CLI
        # backfill from a different cwd). Fall back to env only.
        return env or None
    if cfg.enabled and cfg.endpoint:
        return cfg.endpoint.rstrip("/")
    return env.rstrip("/") if env else None


def is_enabled() -> bool:
    """True when a Phoenix endpoint is configured. Cheap pre-check the
    runner uses to skip imports of the phoenix client when not needed."""
    return resolve_endpoint() is not None


# --- public entry ---------------------------------------------------


def upload_run(
    run_dir: Path,
    *,
    client: Any = None,
    http: Any = None,
) -> UploadResult:
    """Upload one bench run's results to Phoenix.

    ``client`` / ``http`` are injection seams for tests. In production
    both come from ``_default_phoenix_client()``: ``client`` is the
    high-level ``phoenix.client.Client`` (datasets), ``http`` is the
    underlying httpx session (raw REST for experiments + runs + evals).

    Reads ``run_dir/manifest.json`` (required) and ``run_dir/trials.jsonl``
    (required). Per-trial granularity is preserved; per-cell aggregation
    happens inside Phoenix from the per-run scores.

    Raises:
      :class:`PhoenixUnreachable` — when an endpoint is configured but
      the server doesn't respond within :data:`_REACHABILITY_TIMEOUT_S`.
      Callers must catch this and surface to the user; silent skip is
      not the right behaviour when the configuration explicitly asks
      for Phoenix.
    """
    endpoint = resolve_endpoint()
    if endpoint is None:
        return UploadResult(skipped=True, skip_reason="no PHOENIX endpoint configured")

    # Pre-flight probe so a downed Phoenix fails in ~2 s instead of
    # burning the full httpx-default timeout on every blocked call
    # (dataset create + each experiment + N runs + N evals — adds up
    # to minutes of dead waiting). Skipped when callers inject a
    # client (tests use stubs that don't speak HTTP).
    if client is None and http is None:
        _assert_reachable(endpoint)

    try:
        manifest = read_manifest(run_dir)
    except FileNotFoundError as exc:
        # Missing manifest.json is unrecoverable for upload — every
        # downstream object depends on prompt_versions / commit_sha /
        # cases.content_sha256 that only the runner can record. Re-run
        # the benchmark instead of trying to upload an orphan dir.
        raise FileNotFoundError(
            f"{run_dir / MANIFEST_FILENAME} not found — re-run the "
            "benchmark to generate one; legacy run dirs cannot be "
            "uploaded."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        return _result_with_error("manifest", str(exc), endpoint=endpoint)

    trials = list(_load_trials(run_dir))
    if not trials:
        return UploadResult(
            skipped=True,
            skip_reason=f"{run_dir / 'trials.jsonl'} empty — nothing to upload",
            endpoint=endpoint,
        )

    if client is None or http is None:
        try:
            client, http = _default_phoenix_client(endpoint)
        except Exception as exc:  # noqa: BLE001
            return _result_with_error("client_init", str(exc), endpoint=endpoint)

    result = UploadResult(endpoint=endpoint)

    # 1. Dataset (one per runner). Phoenix versions by content hash, so
    #    re-uploading unchanged cases is a no-op revision-wise. We prefer
    #    the per-run ``cases_snapshot.json`` over deriving examples from
    #    trial rows: the snapshot carries the full case body (prompts
    #    templates, predicate source) so a Phoenix dataset version is a
    #    real case-history archive, not a "what the LLM saw this run"
    #    excerpt. Snapshot may be absent for legacy run dirs predating
    #    this layer — in that case we fall back to trial-derived data.
    snapshot: CasesSnapshot | None = None
    try:
        snapshot = read_snapshot(run_dir)
    except Exception as exc:  # noqa: BLE001
        # Corrupt snapshot is worth surfacing but shouldn't kill the
        # upload — fall back to the trial-derived path with a note.
        result.errors.append(
            UploadError(stage="snapshot", detail=f"{exc}; falling back to trials")
        )
    dataset_name = _dataset_name(manifest.runner)
    try:
        dataset_id, dataset_version_id = _upload_dataset(
            client, dataset_name, manifest, trials, snapshot
        )
    except Exception as exc:  # noqa: BLE001
        result.errors.append(UploadError(stage="dataset", detail=str(exc)))
        return result
    result.dataset_name = dataset_name
    result.dataset_id = dataset_id
    result.dataset_version_id = dataset_version_id

    # 2 + 3. Experiments + runs + evals per cell.
    for cell_key, cell_trials in _trials_by_cell(trials).items():
        exp = _upload_one_experiment(
            http=http,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            dataset_name=dataset_name,
            manifest=manifest,
            cell_key=cell_key,
            cell_trials=cell_trials,
            errors=result.errors,
            endpoint=endpoint,
        )
        result.experiments.append(exp)

    return result


# --- internals: client construction ---------------------------------


def _assert_reachable(endpoint: str) -> None:
    """Quick HTTP probe — raise :class:`PhoenixUnreachable` if Phoenix
    doesn't answer within :data:`_REACHABILITY_TIMEOUT_S`.

    Probes ``GET <endpoint>/`` which is the cheapest endpoint Phoenix
    exposes; any 2xx / 3xx / even 4xx means the server is up. Only
    ``httpx.ConnectError`` / ``httpx.ConnectTimeout`` / ``httpx.ReadTimeout``
    mean "down" — other exceptions propagate as ``PhoenixUnreachable``
    too because we can't safely distinguish them from "down" in the
    pre-flight position.
    """
    import httpx

    try:
        resp = httpx.get(endpoint.rstrip("/") + "/", timeout=_REACHABILITY_TIMEOUT_S)
        # Any HTTP response (including 4xx) means a server answered.
        # Only a transport-level failure counts as "unreachable".
        _ = resp.status_code
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
        raise PhoenixUnreachable(
            f"Phoenix endpoint {endpoint!r} did not respond within "
            f"{_REACHABILITY_TIMEOUT_S:.0f}s ({type(exc).__name__}). "
            "Start Phoenix (`uvx arize-phoenix serve` / "
            "`docker run -p 6006:6006 arizephoenix/phoenix:latest`), or "
            "set `tracing.enabled: false` in configs/app.yaml / unset "
            "PHOENIX_COLLECTOR_ENDPOINT if Phoenix is intentionally off."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise PhoenixUnreachable(
            f"Phoenix endpoint {endpoint!r} probe failed: {exc!r}. "
            "Treating as unreachable to avoid hanging on real uploads."
        ) from exc


def _default_phoenix_client(endpoint: str) -> tuple[Any, Any]:
    """Build the high-level client + raw http session against ``endpoint``.

    The two share the same auth header (``PHOENIX_API_KEY`` env var when
    set) and base URL. We construct the httpx client explicitly rather
    than reaching for ``client._client`` because that attribute is
    private and has changed between minor versions.
    """
    import httpx
    from phoenix.client import Client

    headers: dict[str, str] = {}
    api_key = os.environ.get("PHOENIX_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    client = Client(base_url=endpoint, api_key=api_key or None)
    http = httpx.Client(base_url=endpoint, headers=headers, timeout=30.0)
    return client, http


# --- internals: dataset ---------------------------------------------


def _dataset_name(runner: str) -> str:
    """Stable Phoenix dataset name per runner."""
    return f"tool_invoke.{runner}"


def _upload_dataset(
    client: Any,
    dataset_name: str,
    manifest: RunManifest,
    trials: list[dict],
    snapshot: CasesSnapshot | None,
) -> tuple[str, Optional[str]]:
    """Create or version the dataset for this runner.

    Example source priority:
      1. ``cases_snapshot.json`` (rich: prompts dict, predicate source)
      2. Trial rows (lossy fallback: only formatted user_prompt)

    Either way the dataset is keyed by ``case_id = "<name>@v<rev>"``
    so Phoenix's content-versioning still does its job — re-uploading
    with unchanged cases produces no new version, bumping a revision
    does.

    Returns ``(dataset_id, dataset_version_id)``. ``version_id`` may be
    ``None`` when the client wrapper doesn't surface it; the experiment
    create call doesn't require it as a separate field.
    """
    examples = _dataset_examples(snapshot, trials)
    inputs = [e["input"] for e in examples]
    outputs = [e["output"] for e in examples]
    metadata = [e["metadata"] for e in examples]

    dataset = client.datasets.create_dataset(
        name=dataset_name,
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
    )
    # Phoenix client may return either a dict-like or an object. Probe
    # both shapes so we keep working across minor version bumps.
    dataset_id = _attr(dataset, "id") or _attr(dataset, "dataset_id")
    version_id = _attr(dataset, "version_id") or _attr(dataset, "dataset_version_id")
    if dataset_id is None:
        msg = (
            f"phoenix returned a dataset without an id; raw={dataset!r}. "
            "Cannot proceed without one — check phoenix-client version."
        )
        raise RuntimeError(msg)
    return str(dataset_id), str(version_id) if version_id else None


def _dataset_examples(snapshot: CasesSnapshot | None, trials: list[dict]) -> list[dict]:
    """Build one Phoenix dataset example per ``(case_name, revision)``.

    Snapshot path (preferred): one example per snapshot entry, carrying
    the full case body — prompts dict, predicate source, seed source —
    so the dataset version is the case-history archive.

    Trial-fallback path: derive from trial rows when no snapshot exists.
    Same composite key (case_id) so the experiment-side wiring is
    unchanged; just less content in the example body.
    """
    if snapshot is not None and snapshot.cases:
        return _examples_from_snapshot(snapshot)
    return _examples_from_trials(trials)


def _examples_from_snapshot(snapshot: CasesSnapshot) -> list[dict]:
    """Build dataset examples from the rich per-run snapshot.

    Fields are split across input / output / metadata to match Phoenix
    UI conventions: ``input`` is what the case asks (case identity +
    prompt templates), ``output`` is the success criterion (expected
    tool), ``metadata`` is the searchable header (the only place
    ``case_id`` is searched against by the lookup CLI).
    """
    examples: list[dict] = []
    for entry in sorted(snapshot.cases, key=lambda e: e.case_id):
        examples.append(
            {
                "input": {
                    "case_name": entry.name,
                    "revision": entry.revision,
                    "tier": entry.tier,
                    "expected_behavior": entry.expected_behavior,
                    "prompts": entry.prompts,
                    "args_predicate_src": entry.args_predicate_src,
                    "seed_src": entry.seed_src,
                },
                "output": {
                    "expected_tool": entry.expected_tool,
                    "expected_tools": entry.expected_tools,
                },
                "metadata": {
                    "case_id": entry.case_id,
                    "case_name": entry.name,
                    "revision": entry.revision,
                    "tier": entry.tier,
                    "content_sha256": entry.content_sha256,
                    "source": "snapshot",
                },
            }
        )
    return examples


def _examples_from_trials(trials: list[dict]) -> list[dict]:
    """Legacy fallback: derive examples from trial rows.

    Only used for run dirs that predate ``cases_snapshot.json``. The
    example body is sparser — no prompts template, no predicate source
    — but the case_id key is still correct so per-experiment links
    still work; the lookup CLI just returns less.
    """
    seen: dict[tuple[str, int], dict] = {}
    for t in trials:
        key = (t["case_name"], int(t.get("case_revision", 1)))
        if key in seen:
            continue
        seen[key] = {
            "input": {
                "case_name": t["case_name"],
                "revision": int(t.get("case_revision", 1)),
                "tier": t.get("tier"),
                "expected_behavior": t.get("expected_behavior"),
                "user_prompt": t.get("user_prompt"),
            },
            "output": {
                "expected_tool": t.get("expected_tool"),
                "expected_tools": t.get("expected_tools") or [],
            },
            "metadata": {
                "case_name": t["case_name"],
                "revision": int(t.get("case_revision", 1)),
                "tier": t.get("tier"),
                "case_id": f"{t['case_name']}@v{int(t.get('case_revision', 1))}",
                "source": "trials-fallback",
            },
        }
    return [seen[k] for k in sorted(seen)]


# --- internals: experiments + runs + evals --------------------------


def _trials_by_cell(trials: list[dict]) -> dict[str, list[dict]]:
    """Group trial rows by their experiment cell.

    Cell key encodes the comparison axis: same model, same user lang,
    same tool-prompt lang → same experiment.  ``tool_prompt_lang`` may
    be absent on symptoms/vision rows; missing values bucket together
    under the ``-`` placeholder.
    """
    by: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        model = t.get("model", "?")
        ulang = t.get("lang", "?")
        tpl = t.get("tool_prompt_lang") or "-"
        key = f"{model}__u_{ulang}__t_{tpl}"
        by[key].append(t)
    return dict(by)


def _upload_one_experiment(
    *,
    http: Any,
    dataset_id: str,
    dataset_version_id: Optional[str],
    dataset_name: str,
    manifest: RunManifest,
    cell_key: str,
    cell_trials: list[dict],
    errors: list[UploadError],
    endpoint: str,
) -> UploadedExperiment:
    """POST one experiment + its runs + their evaluations.

    Idempotent: if an experiment with the same deterministic name
    already exists on this dataset (re-upload of the same run dir),
    skip the POST and return the existing experiment id with
    ``n_runs=0`` to signal "nothing new uploaded".
    """
    experiment_name = _experiment_name(manifest, cell_key)
    experiment_metadata = _experiment_metadata(manifest, cell_key, cell_trials)

    existing_id = _find_existing_experiment(
        http=http,
        dataset_id=dataset_id,
        metadata=experiment_metadata,
    )
    if existing_id:
        return UploadedExperiment(
            cell_key=f"{cell_key} (existing)",
            experiment_id=existing_id,
            experiment_url=f"{endpoint}/experiments/{existing_id}",
            n_runs=0,
            n_evals=0,
        )

    create_payload: dict[str, Any] = {
        "name": experiment_name,
        "description": _experiment_description(manifest, cell_key, cell_trials),
        "metadata": experiment_metadata,
    }
    if dataset_version_id:
        # Pin the experiment to the exact dataset version we just
        # uploaded so a later case-add doesn't retroactively make this
        # experiment look like it skipped questions.
        create_payload["version_id"] = dataset_version_id

    try:
        resp = http.post(
            _DATASET_EXPERIMENT_PATH.format(dataset_id=dataset_id),
            json=create_payload,
        )
        resp.raise_for_status()
        body = resp.json()
        # Phoenix wraps responses in {"data": {...}} sometimes; unwrap.
        body = body.get("data", body) if isinstance(body, dict) else body
        experiment_id = body.get("id") if isinstance(body, dict) else None
    except Exception as exc:  # noqa: BLE001
        errors.append(UploadError(stage="experiment", detail=f"{cell_key}: {exc}"))
        return UploadedExperiment(
            cell_key=cell_key,
            experiment_id=None,
            experiment_url=None,
            n_runs=0,
            n_evals=0,
        )

    if not experiment_id:
        errors.append(
            UploadError(
                stage="experiment",
                detail=f"{cell_key}: phoenix returned no experiment id",
            )
        )
        return UploadedExperiment(
            cell_key=cell_key,
            experiment_id=None,
            experiment_url=None,
            n_runs=0,
            n_evals=0,
        )

    n_runs = 0
    n_evals = 0
    # Phoenix runs / evals are per-(example, trial). The dataset example
    # id is derivable from (case_name, revision) — Phoenix returns those
    # in the dataset upload response too, but the metadata.case_id we
    # uploaded is a stable string so we re-derive locally and resolve
    # to dataset_example_id via a fetch on first use.
    example_id_by_case = _fetch_example_ids(
        http=http, dataset_id=dataset_id, dataset_version_id=dataset_version_id
    )

    # Phoenix requires ``repetition_number`` per run — 1-indexed per
    # (experiment, example) pair. trials.jsonl doesn't carry an index,
    # so we count within this cell per case_id in stream order.
    repetition_by_case: dict[str, int] = defaultdict(int)
    for trial in cell_trials:
        case_id = f"{trial['case_name']}@v{int(trial.get('case_revision', 1))}"
        example_id = example_id_by_case.get(case_id)
        if not example_id:
            errors.append(
                UploadError(
                    stage="run",
                    detail=(
                        f"{cell_key}: no dataset example for {case_id}; "
                        "skipping run+eval"
                    ),
                )
            )
            continue
        repetition_by_case[case_id] += 1
        run_id = _post_run(
            http=http,
            experiment_id=experiment_id,
            example_id=example_id,
            trial=trial,
            repetition_number=repetition_by_case[case_id],
            errors=errors,
            cell_key=cell_key,
        )
        if run_id is None:
            continue
        n_runs += 1
        if _post_predicate_eval(
            http=http,
            experiment_id=experiment_id,
            run_id=run_id,
            trial=trial,
            errors=errors,
            cell_key=cell_key,
        ):
            n_evals += 1

    return UploadedExperiment(
        cell_key=cell_key,
        experiment_id=str(experiment_id),
        experiment_url=f"{endpoint}/experiments/{experiment_id}",
        n_runs=n_runs,
        n_evals=n_evals,
    )


def _experiment_name(manifest: RunManifest, cell_key: str) -> str:
    """Human + machine-readable experiment name carrying the comparison axes."""
    sha = (manifest.commit_sha or "nosha")[:7]
    # ``prompts_<sha7>`` of the prompt_versions map gives a stable handle
    # to "what set of prompt versions drove this run". Two runs with
    # identical prompt_versions dicts get the same suffix.
    pv_blob = json.dumps(manifest.prompt_versions, sort_keys=True)
    import hashlib

    pv_sha = hashlib.sha256(pv_blob.encode("utf-8")).hexdigest()[:7]
    return (
        f"{manifest.runner}__{cell_key}__prompts_{pv_sha}__"
        f"code_{sha}__{manifest.bench_ts}"
    )


def _experiment_metadata(
    manifest: RunManifest, cell_key: str, cell_trials: list[dict]
) -> dict[str, Any]:
    """All the per-experiment fields Phoenix's UI / API can filter on.

    Filter targets:
      - ``model`` / ``user_lang`` / ``tool_prompt_lang`` for the cell
      - ``prompt_versions`` (nested dict) for "all experiments where
        tool_proposal=v4"
      - ``commit_sha`` for "all experiments since deploy X"
      - ``bench_ts`` for absolute time ordering when comparing back-to-back

    Aggregate fields (``n_trials``, ``pass_count``, ``accuracy``) are a
    hint for cards / hovers. Phoenix derives its own aggregates from the
    per-run scores too, so these are redundant-but-cheap.
    """
    parts = cell_key.split("__")
    model = parts[0] if parts else "?"
    ulang = parts[1].removeprefix("u_") if len(parts) > 1 else "?"
    tpl = parts[2].removeprefix("t_") if len(parts) > 2 else "-"

    n_trials = len(cell_trials)
    n_pass = sum(1 for t in cell_trials if t.get("predicate_pass"))
    n_err = sum(1 for t in cell_trials if t.get("had_error"))
    latencies = [
        float(t["latency_ms"])
        for t in cell_trials
        if t.get("latency_ms") is not None and not t.get("had_error")
    ]
    mean_latency = round(sum(latencies) / len(latencies), 1) if latencies else None

    return {
        "runner": manifest.runner,
        "model": model,
        "user_lang": ulang,
        "tool_prompt_lang": tpl,
        "bench_ts": manifest.bench_ts,
        "commit_sha": manifest.commit_sha,
        "prompt_versions": dict(manifest.prompt_versions),
        "cases_content_sha256": manifest.cases.content_sha256,
        "n_trials": n_trials,
        "n_pass": n_pass,
        "n_error": n_err,
        "accuracy": round(n_pass / n_trials, 4) if n_trials else None,
        "mean_latency_ms": mean_latency,
    }


def _experiment_description(
    manifest: RunManifest, cell_key: str, cell_trials: list[dict]
) -> str:
    """Compact one-line description that fits in the experiment card."""
    n = len(cell_trials)
    n_pass = sum(1 for t in cell_trials if t.get("predicate_pass"))
    pv = ", ".join(f"{k}={v}" for k, v in sorted(manifest.prompt_versions.items()))
    return (
        f"{manifest.runner} {cell_key} "
        f"({n_pass}/{n} passed) — {pv or 'no prompt versions captured'}"
    )


# --- per-trial run + eval -------------------------------------------


def _post_run(
    *,
    http: Any,
    experiment_id: str,
    example_id: str,
    trial: dict,
    repetition_number: int,
    errors: list[UploadError],
    cell_key: str,
) -> Optional[str]:
    """POST one experiment run. Returns its id, or ``None`` on failure."""
    output = {
        # Predicate-side decisions the runner already computed. Phoenix
        # treats output as opaque JSON; structure it as the UI is most
        # likely to render: a primary outcome + the underlying detail.
        "outcome": trial.get("outcome"),
        "predicate_pass": bool(trial.get("predicate_pass")),
        "predicate_reason": trial.get("predicate_reason"),
        "tool_calls": trial.get("tool_calls") or [],
        "ask_questions": trial.get("ask_questions") or [],
        "final_response_text": trial.get("final_response_text"),
        "had_error": bool(trial.get("had_error")),
        "error_msg": trial.get("error_msg"),
        # Case identity + cell axes — embedded on the run row for the
        # run-details panel and REST/jq access. The Phoenix compare-view
        # filter cannot reach these (it queries the dataset example, not
        # the run), so don't chase them with ``output[...]`` filters —
        # use ``metadata.case_name`` against the example side instead.
        "case_name": trial.get("case_name"),
        "case_revision": int(trial.get("case_revision", 1)),
        "case_id": f"{trial['case_name']}@v{int(trial.get('case_revision', 1))}",
        "model": trial.get("model"),
        "user_lang": trial.get("lang"),
        "tool_prompt_lang": trial.get("tool_prompt_lang"),
        "tier": trial.get("tier"),
        "expected_tool": trial.get("expected_tool"),
    }
    start_iso, end_iso = _trial_time_window(trial)
    payload = {
        "dataset_example_id": example_id,
        "output": output,
        "repetition_number": repetition_number,
        "start_time": start_iso,
        "end_time": end_iso,
        "trace_id": trial.get("request_id"),
    }
    try:
        resp = http.post(
            _EXPERIMENT_RUNS_PATH.format(experiment_id=experiment_id),
            json=payload,
        )
        resp.raise_for_status()
        body = resp.json()
        body = body.get("data", body) if isinstance(body, dict) else body
        run_id = body.get("id") if isinstance(body, dict) else None
        if not run_id:
            errors.append(
                UploadError(
                    stage="run",
                    detail=f"{cell_key}/{trial.get('case_name')}: phoenix returned no run id",
                )
            )
            return None
        return str(run_id)
    except Exception as exc:  # noqa: BLE001
        body_text = ""
        try:
            body_text = resp.text[:500] if "resp" in locals() else ""
        except Exception:  # noqa: BLE001
            body_text = ""
        errors.append(
            UploadError(
                stage="run",
                detail=f"{cell_key}/{trial.get('case_name')}: {exc} | body={body_text}",
            )
        )
        return None


_IDENTITY_KEYS = ("bench_ts", "model", "user_lang", "tool_prompt_lang")


def _find_existing_experiment(
    *, http: Any, dataset_id: str, metadata: dict
) -> Optional[str]:
    """Return the id of an existing experiment matching the cell identity.

    Phoenix's list endpoint omits the ``name`` field even though we set
    it on create, so we match on the metadata keys that uniquely
    identify one (run_dir, cell) tuple: ``bench_ts`` pins the run dir,
    ``model``/``user_lang``/``tool_prompt_lang`` pin the cell within it.

    Returns ``None`` on lookup failure (treat as "no existing" and let
    the POST proceed).
    """
    want = {k: metadata.get(k) for k in _IDENTITY_KEYS}
    try:
        resp = http.get(_DATASET_EXPERIMENT_PATH.format(dataset_id=dataset_id))
        resp.raise_for_status()
        body = resp.json()
        body = body.get("data", body) if isinstance(body, dict) else body
        experiments = body if isinstance(body, list) else []
    except Exception:  # noqa: BLE001
        logger.debug("phoenix-upload: experiment lookup failed", exc_info=True)
        return None
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        meta = exp.get("metadata") or {}
        if not all(meta.get(k) == want[k] for k in _IDENTITY_KEYS):
            continue
        # If the user emptied the experiment via Phoenix UI (deleted
        # all its runs), treat it as not-yet-uploaded so a re-upload
        # actually lands. The empty shell stays — Phoenix forbids
        # duplicate names, but our names embed bench_ts so collision
        # only happens on identical runs the user is clearly trying
        # to re-populate.
        n_runs = int(exp.get("successful_run_count") or 0) + int(
            exp.get("failed_run_count") or 0
        )
        if n_runs == 0:
            return None
        ex_id = exp.get("id")
        return str(ex_id) if ex_id else None
    return None


def _trial_time_window(trial: dict) -> tuple[str, str]:
    """Derive ``(start_iso, end_iso)`` from a trial row.

    ``timestamp_utc`` is recorded right after ``latency_ms`` is computed,
    so it represents the trial's end time. Start = end - latency. Both
    fields are required by Phoenix's experiment-run schema.
    """
    end_raw = trial.get("timestamp_utc")
    try:
        end_dt = (
            datetime.fromisoformat(end_raw) if end_raw else datetime.now(timezone.utc)
        )
    except ValueError:
        end_dt = datetime.now(timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    latency_ms = trial.get("latency_ms") or 0
    start_dt = end_dt - timedelta(milliseconds=float(latency_ms))
    return start_dt.isoformat(), end_dt.isoformat()


def _post_predicate_eval(
    *,
    http: Any,
    experiment_id: str,
    run_id: str,
    trial: dict,
    errors: list[UploadError],
    cell_key: str,
) -> bool:
    """POST a deterministic ``predicate_pass`` evaluation for one run."""
    passed = bool(trial.get("predicate_pass"))
    # Deterministic CODE evaluator — ran instantly alongside the trial,
    # so eval start == eval end == trial end time. Phoenix's schema still
    # requires both fields.
    _, end_iso = _trial_time_window(trial)
    payload = {
        "experiment_run_id": run_id,
        "name": _EVAL_PREDICATE,
        "annotator_kind": "CODE",
        "start_time": end_iso,
        "end_time": end_iso,
        "result": {
            "label": "pass" if passed else "fail",
            "score": 1.0 if passed else 0.0,
            "explanation": trial.get("predicate_reason") or "",
        },
    }
    try:
        resp = http.post(_EXPERIMENT_EVALS_PATH, json=payload)
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        body_text = ""
        try:
            body_text = resp.text[:500] if "resp" in locals() else ""
        except Exception:  # noqa: BLE001
            body_text = ""
        errors.append(
            UploadError(
                stage="evaluation",
                detail=f"{cell_key}/{trial.get('case_name')}: {exc} | body={body_text}",
            )
        )
        return False


# --- dataset example id lookup --------------------------------------


def _fetch_example_ids(
    *, http: Any, dataset_id: str, dataset_version_id: Optional[str]
) -> dict[str, str]:
    """Map our stable ``case_id`` strings to Phoenix's dataset example ids.

    Phoenix's dataset upload response doesn't reliably surface per-example
    ids across all client versions. The list endpoint is the safe path:
    ``GET /v1/datasets/{id}/examples?version_id=...`` returns examples
    with their ids and metadata. We index by ``metadata.case_id``.
    """
    params: dict[str, str] = {}
    if dataset_version_id:
        params["version_id"] = dataset_version_id
    try:
        resp = http.get(f"v1/datasets/{dataset_id}/examples", params=params)
        resp.raise_for_status()
        body = resp.json()
        body = body.get("data", body) if isinstance(body, dict) else body
        examples = body.get("examples", body) if isinstance(body, dict) else body
    except Exception:  # noqa: BLE001
        logger.exception("phoenix-upload: failed to list dataset examples")
        return {}

    out: dict[str, str] = {}
    if not isinstance(examples, list):
        return out
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        meta = ex.get("metadata") or {}
        case_id = meta.get("case_id")
        ex_id = ex.get("id")
        if case_id and ex_id:
            out[case_id] = str(ex_id)
    return out


# --- io helpers -----------------------------------------------------


def _load_trials(run_dir: Path) -> Iterable[dict]:
    """Yield trial dicts from ``trials.jsonl``. Skips blank lines."""
    path = run_dir / "trials.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("phoenix-upload: skip malformed trial line in %s", path)
    return out


def _attr(obj: Any, name: str) -> Any:
    """Lookup ``name`` on ``obj``, handling dict / object / pydantic shapes."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _result_with_error(stage: str, detail: str, *, endpoint: str) -> UploadResult:
    return UploadResult(
        endpoint=endpoint,
        errors=[UploadError(stage=stage, detail=detail)],
    )


__all__ = [
    "PhoenixUnreachable",
    "UploadError",
    "UploadResult",
    "UploadedExperiment",
    "is_enabled",
    "resolve_endpoint",
    "upload_run",
]
