"""Phoenix upload pipeline.

Pure unit tests — no Phoenix server contact. The two collaborators
(``phoenix.client.Client.datasets`` for dataset upload and the raw
``httpx`` session for experiment/run/eval REST POSTs) are stubbed so we
assert on the exact payloads the uploader would have sent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.benchmarks.tool_invoke import phoenix_upload
from tests.benchmarks.tool_invoke.manifest import (
    CasesSection,
    ConfigSection,
    RunManifest,
    write_manifest,
)


# --- fixtures -------------------------------------------------------


def _trial(
    case_name: str,
    *,
    model: str = "omlx",
    user_lang: str = "en",
    tool_prompt_lang: str = "en",
    revision: int = 1,
    predicate_pass: bool = True,
    request_id: str | None = None,
    latency_ms: float | None = 1234.0,
) -> dict:
    """Build one trials.jsonl row with the fields the uploader reads."""
    return {
        "request_id": request_id or f"req-{case_name}-{user_lang}",
        "timestamp_utc": "2026-06-17T12:00:00+00:00",
        "model": model,
        "lang": user_lang,
        "case_name": case_name,
        "case_revision": revision,
        "tier": "base",
        "expected_behavior": "call_tool",
        "expected_tool": case_name,
        "expected_tools": [],
        "tool_prompt_lang": tool_prompt_lang,
        "user_prompt": f"prompt for {case_name}",
        "tool_calls": [{"tool_name": case_name, "args": {}}],
        "ask_questions": [],
        "final_response_text": "ok",
        "outcome": "correct" if predicate_pass else "wrong_tool",
        "predicate_pass": predicate_pass,
        "predicate_reason": "stub",
        "had_error": False,
        "error_msg": None,
        "latency_ms": latency_ms,
    }


@pytest.fixture
def bench_run_dir(tmp_path: Path) -> Path:
    """Materialize a minimal bench run dir (manifest + trials.jsonl)."""
    out = tmp_path / "ingest" / "20260617_120000_001"
    out.mkdir(parents=True)
    write_manifest(
        out,
        RunManifest(
            runner="ingest",
            bench_ts="20260617_120000_001",
            started_at="2026-06-17T12:00:00+00:00",
            finished_at="2026-06-17T12:05:00+00:00",
            commit_sha="abc1234",
            cases=CasesSection(
                count=2,
                tiers=["base"],
                content_sha256="d" * 64,
            ),
            prompt_versions={"tool_proposal": "v4", "save_record_tool": "v1"},
            config=ConfigSection(
                models=["omlx"],
                user_langs=["en", "zh"],
                tool_prompt_langs=["en", "zh"],
                trials=2,
            ),
        ),
    )
    trials = [
        _trial("save_allergy", user_lang="en", tool_prompt_lang="en"),
        _trial(
            "save_allergy",
            user_lang="en",
            tool_prompt_lang="en",
            predicate_pass=False,
        ),
        _trial("save_medication", user_lang="en", tool_prompt_lang="en"),
        _trial("save_allergy", user_lang="zh", tool_prompt_lang="zh"),
    ]
    (out / "trials.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trials) + "\n", encoding="utf-8"
    )
    return out


# --- stub Phoenix client + http session -----------------------------


class _StubDatasetsApi:
    """Mimic phoenix.client.Client.datasets just enough for the uploader."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create_dataset(self, *, name, inputs, outputs, metadata):
        self.created.append(
            {"name": name, "inputs": inputs, "outputs": outputs, "metadata": metadata}
        )
        # Return a dict-shaped dataset object; the uploader handles both
        # dict and attribute access via `_attr`.
        return {"id": "dataset-1", "version_id": "version-1"}


class _StubClient:
    def __init__(self) -> None:
        self.datasets = _StubDatasetsApi()


class _StubResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _StubHttp:
    """Record every POST/GET the uploader makes and reply with canned data."""

    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.gets: list[tuple[str, dict[str, Any]]] = []
        # 2 examples uploaded → 2 ids returned.
        self.example_ids = {
            "save_allergy@v1": "ex-allergy-1",
            "save_medication@v1": "ex-med-1",
        }
        self._next_run = 1
        self._next_experiment = 1

    def post(self, path: str, *, json: dict[str, Any]) -> _StubResponse:
        self.posts.append((path, json))
        if path.startswith("v1/datasets/") and path.endswith("/experiments"):
            payload = {"data": {"id": f"exp-{self._next_experiment}"}}
            self._next_experiment += 1
            return _StubResponse(payload)
        if path.startswith("v1/experiments/") and path.endswith("/runs"):
            payload = {"data": {"id": f"run-{self._next_run}"}}
            self._next_run += 1
            return _StubResponse(payload)
        if path == "v1/experiment_evaluations":
            return _StubResponse({"data": {"id": "eval-1"}})
        return _StubResponse({})

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> _StubResponse:
        self.gets.append((path, params or {}))
        if path.endswith("/examples"):
            examples = [
                {"id": ex_id, "metadata": {"case_id": case_id}}
                for case_id, ex_id in self.example_ids.items()
            ]
            return _StubResponse({"examples": examples})
        return _StubResponse({})


# --- tests ----------------------------------------------------------


def test_resolve_endpoint_skips_when_unset(monkeypatch):
    """No env, no tracing.enabled config → upload disabled."""
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    monkeypatch.setattr(
        phoenix_upload,
        "_load_config_if_available",
        lambda: None,
        raising=False,
    )
    # tracing config may load successfully but with enabled=False —
    # simulate that by patching the imported _load_config to a stub
    # that returns a falsy-enabled config.
    from claritymed.core.observability import tracing as _t

    monkeypatch.setattr(_t, "_load_config", lambda: _t.TracingConfig(enabled=False))
    assert phoenix_upload.resolve_endpoint() is None
    assert phoenix_upload.is_enabled() is False


def test_resolve_endpoint_env_wins_when_tracing_disabled(monkeypatch):
    """``PHOENIX_COLLECTOR_ENDPOINT`` resolves the endpoint even when
    tracing.enabled is False."""
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phx.example/")
    from claritymed.core.observability import tracing as _t

    monkeypatch.setattr(_t, "_load_config", lambda: _t.TracingConfig(enabled=False))
    assert phoenix_upload.resolve_endpoint() == "http://phx.example"
    assert phoenix_upload.is_enabled() is True


def test_upload_skipped_when_no_endpoint(bench_run_dir, monkeypatch):
    """Skip path: no errors raised, result.skipped is True."""
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    from claritymed.core.observability import tracing as _t

    monkeypatch.setattr(_t, "_load_config", lambda: _t.TracingConfig(enabled=False))
    result = phoenix_upload.upload_run(bench_run_dir)
    assert result.skipped is True
    assert "no PHOENIX endpoint" in (result.skip_reason or "")
    assert result.errors == []


def test_upload_raises_when_manifest_missing(tmp_path, monkeypatch):
    """No manifest.json (legacy run dir) → FileNotFoundError with a
    clear remediation. Legacy backfill is explicitly NOT supported —
    re-running the bench is the only path."""
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phx.example")
    # Bypass reachability probe so the test doesn't depend on real network.
    monkeypatch.setattr(phoenix_upload, "_assert_reachable", lambda endpoint: None)
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "trials.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="re-run the benchmark"):
        phoenix_upload.upload_run(empty)


def test_upload_raises_when_phoenix_unreachable(bench_run_dir, monkeypatch):
    """Configured endpoint + Phoenix down → PhoenixUnreachable with
    actionable remediation. NEVER silently skip — config says Phoenix
    is expected; silently dropping the upload would hide the
    misconfiguration."""
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://nope.invalid")

    def _boom(endpoint: str) -> None:
        raise phoenix_upload.PhoenixUnreachable(
            f"Phoenix endpoint {endpoint!r} did not respond within 2s "
            "(simulated). Start Phoenix or unset the endpoint."
        )

    monkeypatch.setattr(phoenix_upload, "_assert_reachable", _boom)
    with pytest.raises(phoenix_upload.PhoenixUnreachable, match="did not respond"):
        phoenix_upload.upload_run(bench_run_dir)


def test_assert_reachable_raises_on_connect_failure(monkeypatch):
    """Probe wraps httpx ConnectError into PhoenixUnreachable with the
    remediation hint baked in."""
    import httpx

    def _fail(*args, **kwargs):
        raise httpx.ConnectError("Cannot assign requested address")

    monkeypatch.setattr(httpx, "get", _fail)
    with pytest.raises(phoenix_upload.PhoenixUnreachable, match="did not respond"):
        phoenix_upload._assert_reachable("http://localhost:6006")


def test_assert_reachable_passes_when_server_answers(monkeypatch):
    """Any HTTP response (even 404) means the server is up; no raise."""
    import httpx

    class _StubResp:
        status_code = 404

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _StubResp())
    phoenix_upload._assert_reachable("http://localhost:6006")  # no raise


def test_upload_happy_path(bench_run_dir, monkeypatch):
    """End-to-end with stubbed Phoenix.

    Asserts:
    * Dataset created once with deduped examples (2 unique cases from 4
      trials).
    * One experiment per cell (en/en + zh/zh = 2).
    * Per-trial runs (4 total) and per-run evals (4 total).
    * Experiment metadata carries prompt_versions + commit_sha.
    """
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phx.example")
    client = _StubClient()
    http = _StubHttp()
    result = phoenix_upload.upload_run(bench_run_dir, client=client, http=http)

    assert result.skipped is False
    assert result.errors == [], result.errors
    assert result.dataset_id == "dataset-1"

    # Dataset uploaded once with deduped (case_name, revision) pairs.
    assert len(client.datasets.created) == 1
    created = client.datasets.created[0]
    assert created["name"] == "tool_invoke.ingest"
    assert len(created["inputs"]) == 2
    names = sorted(i["case_name"] for i in created["inputs"])
    assert names == ["save_allergy", "save_medication"]
    case_ids = sorted(m["case_id"] for m in created["metadata"])
    assert case_ids == ["save_allergy@v1", "save_medication@v1"]

    # 2 experiments (en/en cell with 3 trials, zh/zh cell with 1).
    assert len(result.experiments) == 2
    cell_keys = sorted(exp.cell_key for exp in result.experiments)
    assert cell_keys == [
        "omlx__u_en__t_en",
        "omlx__u_zh__t_zh",
    ]

    # Per-trial runs land at v1/experiments/{id}/runs.
    run_posts = [p for p, _ in http.posts if "/runs" in p]
    assert len(run_posts) == 4  # one per trial

    # Per-run evals land at v1/experiment_evaluations.
    eval_posts = [b for p, b in http.posts if p == "v1/experiment_evaluations"]
    assert len(eval_posts) == 4
    # predicate_pass scores: 2 pass (en) + 1 fail (en) + 1 pass (zh)
    scores = sorted(e["result"]["score"] for e in eval_posts)
    assert scores == [0.0, 1.0, 1.0, 1.0]
    assert all(e["name"] == "predicate_pass" for e in eval_posts)

    # Experiment metadata carries prompt_versions + commit_sha + accuracy.
    exp_posts = [
        body
        for path, body in http.posts
        if path.startswith("v1/datasets/") and path.endswith("/experiments")
    ]
    assert len(exp_posts) == 2
    for body in exp_posts:
        meta = body["metadata"]
        assert meta["commit_sha"] == "abc1234"
        assert meta["prompt_versions"] == {
            "tool_proposal": "v4",
            "save_record_tool": "v1",
        }
        assert meta["bench_ts"] == "20260617_120000_001"
        assert meta["accuracy"] in (1.0, round(2 / 3, 4))


def test_dataset_examples_dedupe_and_sort():
    """Two trials of the same case + same revision → one example
    (trial-fallback path when no snapshot is provided)."""
    trials = [
        _trial("save_allergy"),
        _trial("save_allergy"),
        _trial("save_medication"),
    ]
    examples = phoenix_upload._dataset_examples(None, trials)
    assert len(examples) == 2
    # Sorted by name for hash stability.
    assert examples[0]["input"]["case_name"] == "save_allergy"
    assert examples[1]["input"]["case_name"] == "save_medication"
    # Fallback path marks itself as such so a Phoenix consumer can see
    # the body is lossy (no prompts dict, no predicate source).
    assert all(e["metadata"]["source"] == "trials-fallback" for e in examples)


def test_dataset_examples_distinguish_revisions():
    """Bumping ``revision`` produces a *separate* example, not a merge."""
    trials = [_trial("save_allergy", revision=1), _trial("save_allergy", revision=2)]
    examples = phoenix_upload._dataset_examples(None, trials)
    assert len(examples) == 2
    revisions = sorted(e["input"]["revision"] for e in examples)
    assert revisions == [1, 2]
    case_ids = sorted(e["metadata"]["case_id"] for e in examples)
    assert case_ids == ["save_allergy@v1", "save_allergy@v2"]


def test_dataset_examples_prefer_snapshot_over_trials():
    """When a snapshot is present, examples carry the FULL case body
    (prompts dict + predicate source), not the formatted user_prompt."""
    from tests.benchmarks.tool_invoke.cases_snapshot import (
        CaseSnapshotEntry,
        CasesSnapshot,
    )

    snapshot = CasesSnapshot(
        runner="ingest",
        bench_ts="20260617_120000_001",
        cases=[
            CaseSnapshotEntry(
                case_id="save_allergy@v2",
                name="save_allergy",
                revision=2,
                content_sha256="abc" * 21 + "d",  # 64 chars
                tier="base",
                expected_behavior="call_tool",
                expected_tool="save_allergy",
                prompts={"en": "TEMPLATE EN", "zh": "TEMPLATE ZH"},
                args_predicate_src="def _p_substance_penicillin(args):\n    return True, ''",
                seed_src=None,
            )
        ],
    )
    # Trials are present but should be ignored in favour of the snapshot.
    trials = [_trial("save_allergy", revision=2)]
    examples = phoenix_upload._dataset_examples(snapshot, trials)
    assert len(examples) == 1
    inp = examples[0]["input"]
    assert inp["prompts"] == {"en": "TEMPLATE EN", "zh": "TEMPLATE ZH"}
    assert "args_predicate_src" in inp and "penicillin" in inp["args_predicate_src"]
    # Make sure the lossy fallback didn't sneak in.
    assert "user_prompt" not in inp
    md = examples[0]["metadata"]
    assert md["case_id"] == "save_allergy@v2"
    assert md["source"] == "snapshot"
    assert md["content_sha256"].startswith("abc")


def test_trials_by_cell_groups_by_axis():
    """Cell key includes model, user_lang, and tool_prompt_lang."""
    trials = [
        _trial("a", model="omlx", user_lang="en", tool_prompt_lang="en"),
        _trial("a", model="omlx", user_lang="en", tool_prompt_lang="zh"),
        _trial("a", model="deepseek", user_lang="en", tool_prompt_lang="en"),
    ]
    cells = phoenix_upload._trials_by_cell(trials)
    assert set(cells) == {
        "omlx__u_en__t_en",
        "omlx__u_en__t_zh",
        "deepseek__u_en__t_en",
    }


def test_experiment_name_stable_across_runs_with_same_axes():
    """Same (runner, cell, prompt_versions, commit) → same name."""
    m = RunManifest(
        runner="ingest",
        bench_ts="20260617_120000_001",
        started_at="x",
        finished_at="y",
        commit_sha="abc1234",
        cases=CasesSection(count=1, tiers=["base"], content_sha256="x"),
        prompt_versions={"tool_proposal": "v4"},
        config=ConfigSection(models=["omlx"], user_langs=["en"], trials=1),
    )
    n1 = phoenix_upload._experiment_name(m, "omlx__u_en__t_en")
    n2 = phoenix_upload._experiment_name(m, "omlx__u_en__t_en")
    assert n1 == n2
    assert "ingest__omlx__u_en__t_en__prompts_" in n1
    assert "__code_abc1234" in n1


def test_experiment_metadata_computes_aggregates():
    """``n_pass`` / ``accuracy`` come from the cell trials themselves."""
    m = RunManifest(
        runner="ingest",
        bench_ts="t",
        started_at="x",
        finished_at="y",
        commit_sha="c",
        cases=CasesSection(count=1, tiers=["base"], content_sha256="h"),
        prompt_versions={"tool_proposal": "v4"},
        config=ConfigSection(models=["omlx"], user_langs=["en"], trials=1),
    )
    cell_trials = [
        _trial("a", predicate_pass=True, latency_ms=1000),
        _trial("a", predicate_pass=True, latency_ms=2000),
        _trial("a", predicate_pass=False, latency_ms=3000),
    ]
    md = phoenix_upload._experiment_metadata(m, "omlx__u_en__t_en", cell_trials)
    assert md["n_trials"] == 3
    assert md["n_pass"] == 2
    assert md["accuracy"] == round(2 / 3, 4)
    assert md["mean_latency_ms"] == 2000.0
    assert md["prompt_versions"] == {"tool_proposal": "v4"}
