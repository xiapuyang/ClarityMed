"""Tests for ``claritymed bench …`` Typer subcommands.

The CLI is a thin orchestration shell around three uploader/manifest
helpers in ``tests/benchmarks/tool_invoke/``:

* ``phoenix_upload.upload_run`` — the upload pipeline.
* ``manifest.read_manifest`` — manifest reader.
* ``case_history.lookup_case`` — historical case lookup.

Tests pin the CLI's dispatch + exit-code + format behaviour by patching
those helpers; the helpers' own logic is exercised by their own suites.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from claritymed.cli.commands.bench import bench_app

runner = CliRunner()


# --- shared fakes -------------------------------------------------------


def _fake_manifest():
    """Build a real :class:`RunManifest` so the printer sees field types it
    expects (.runner, .bench_ts, etc.). Cheaper than mocking the whole
    dotted-attribute tree."""
    from tests.benchmarks.tool_invoke.manifest import (
        CasesSection,
        ConfigSection,
        RunManifest,
    )

    return RunManifest(
        runner="ingest",
        bench_ts="20260101_010101_001",
        started_at="2026-01-01T01:01:01Z",
        finished_at="2026-01-01T01:11:11Z",
        commit_sha="deadbeefcafe",
        cases=CasesSection(count=12, tiers=["happy", "edge"], content_sha256="ab" * 32),
        prompt_versions={"tool_proposal": "v4", "ask": "v3"},
        config=ConfigSection(
            models=["omlx", "deepseek"],
            user_langs=["en", "zh"],
            tool_prompt_langs=["en"],
            trials=3,
        ),
    )


# --- bench upload --------------------------------------------------------


def test_upload_skipped_no_endpoint_exits_with_1(monkeypatch, tmp_path):
    """No Phoenix endpoint configured → exit 1, message printed."""
    from tests.benchmarks.tool_invoke import phoenix_upload

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def _fake_run(_dir):
        return phoenix_upload.UploadResult(
            skipped=True, skip_reason="no endpoint configured"
        )

    monkeypatch.setattr(phoenix_upload, "upload_run", _fake_run)

    result = runner.invoke(bench_app, ["upload", str(run_dir)])
    assert result.exit_code == 1
    assert "skipped" in result.stdout
    assert "no endpoint" in result.stdout


def test_upload_phoenix_unreachable_exits_with_2(monkeypatch, tmp_path):
    """Phoenix unreachable → exit 2 with the error on stderr."""
    from tests.benchmarks.tool_invoke import phoenix_upload

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def _fake_run(_dir):
        raise phoenix_upload.PhoenixUnreachable("connection refused")

    monkeypatch.setattr(phoenix_upload, "upload_run", _fake_run)

    result = runner.invoke(bench_app, ["upload", str(run_dir)])
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "phoenix" in combined and "connection refused" in combined


def test_upload_missing_manifest_exits_with_2(monkeypatch, tmp_path):
    """Pre-manifest run dir → FileNotFoundError → exit 2."""
    from tests.benchmarks.tool_invoke import phoenix_upload

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def _fake_run(_dir):
        raise FileNotFoundError("manifest.json not found under …/run")

    monkeypatch.setattr(phoenix_upload, "upload_run", _fake_run)

    result = runner.invoke(bench_app, ["upload", str(run_dir)])
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "manifest.json" in combined


def test_upload_happy_path_prints_summary_and_experiments(monkeypatch, tmp_path):
    from tests.benchmarks.tool_invoke import phoenix_upload

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    exp = phoenix_upload.UploadedExperiment(
        cell_key="omlx__u_en__t_en",
        experiment_id="exp-1",
        experiment_url="https://phoenix.local/exp/1",
        n_runs=10,
        n_evals=20,
    )

    def _fake_run(_dir):
        return phoenix_upload.UploadResult(
            endpoint="https://phoenix.local",
            dataset_name="tool_invoke.ingest",
            experiments=[exp],
        )

    monkeypatch.setattr(phoenix_upload, "upload_run", _fake_run)

    result = runner.invoke(bench_app, ["upload", str(run_dir)])
    assert result.exit_code == 0, result.stdout
    assert "dataset=tool_invoke.ingest" in result.stdout
    assert "experiments=1" in result.stdout
    assert "runs=10" in result.stdout
    assert "evals=20" in result.stdout
    assert "https://phoenix.local/exp/1" in result.stdout


def test_upload_non_fatal_errors_print_but_exit_zero(monkeypatch, tmp_path):
    """Per-run / per-evaluation failures land on stderr but do not flip exit code."""
    from tests.benchmarks.tool_invoke import phoenix_upload

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def _fake_run(_dir):
        return phoenix_upload.UploadResult(
            endpoint="https://phoenix.local",
            dataset_name="tool_invoke.ingest",
            experiments=[],
            errors=[
                phoenix_upload.UploadError(stage="run", detail="row 3 failed"),
                phoenix_upload.UploadError(stage="evaluation", detail="eval timed out"),
            ],
        )

    monkeypatch.setattr(phoenix_upload, "upload_run", _fake_run)

    result = runner.invoke(bench_app, ["upload", str(run_dir)])
    assert result.exit_code == 0, result.stdout
    combined = result.stdout + (result.stderr or "")
    assert "row 3 failed" in combined
    assert "eval timed out" in combined


def test_upload_client_init_error_exits_with_2(monkeypatch, tmp_path):
    """``client_init`` / ``manifest`` errors are unrecoverable → exit 2."""
    from tests.benchmarks.tool_invoke import phoenix_upload

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def _fake_run(_dir):
        return phoenix_upload.UploadResult(
            endpoint="https://phoenix.local",
            dataset_name="tool_invoke.ingest",
            experiments=[],
            errors=[
                phoenix_upload.UploadError(
                    stage="client_init", detail="auth handshake failed"
                ),
            ],
        )

    monkeypatch.setattr(phoenix_upload, "upload_run", _fake_run)

    result = runner.invoke(bench_app, ["upload", str(run_dir)])
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "auth handshake failed" in combined


# --- bench manifest -----------------------------------------------------


def test_manifest_prints_all_fields(monkeypatch, tmp_path):
    from tests.benchmarks.tool_invoke import manifest as mfst

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(mfst, "read_manifest", lambda _d: _fake_manifest())

    result = runner.invoke(bench_app, ["manifest", str(run_dir)])
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "runner:" in out and "ingest" in out
    assert "deadbeefcafe" in out
    assert "12 cases" in out
    # Prompt versions section.
    assert "tool_proposal" in out and "v4" in out
    # Config section.
    assert "models=omlx,deepseek" in out
    assert "user_langs=en,zh" in out
    assert "tool_prompt_langs=en" in out
    assert "trials=3" in out


def test_manifest_with_no_prompts_says_none_captured(monkeypatch, tmp_path):
    from tests.benchmarks.tool_invoke import manifest as mfst
    from tests.benchmarks.tool_invoke.manifest import (
        CasesSection,
        ConfigSection,
        RunManifest,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    no_prompts = RunManifest(
        runner="symptoms",
        bench_ts="20260601_000000_002",
        started_at="x",
        finished_at="y",
        commit_sha=None,  # also exercise the (none) branch
        cases=CasesSection(count=0, tiers=[], content_sha256="cd" * 32),
        prompt_versions={},
        config=ConfigSection(models=["m"], user_langs=["en"], trials=1),
    )
    monkeypatch.setattr(mfst, "read_manifest", lambda _d: no_prompts)

    result = runner.invoke(bench_app, ["manifest", str(run_dir)])
    assert result.exit_code == 0
    assert "(none captured)" in result.stdout
    assert "(none)" in result.stdout  # commit_sha None
    # tool_prompt_langs absent → dash.
    assert "tool_prompt_langs=-" in result.stdout


# --- bench case ---------------------------------------------------------


def test_case_unknown_runner_exits_with_2(monkeypatch):
    result = runner.invoke(bench_app, ["case", "totally-wrong-runner", "x@v1"])
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "unknown runner" in combined


def test_case_lookup_error_no_endpoint_exits_with_1(monkeypatch):
    from tests.benchmarks.tool_invoke import case_history

    def _no_endpoint(*, runner, case_id):
        raise case_history.CaseLookupError(
            "no Phoenix endpoint configured (set tracing.endpoint in app.yaml)"
        )

    monkeypatch.setattr(case_history, "lookup_case", _no_endpoint)

    result = runner.invoke(bench_app, ["case", "ingest", "x@v1"])
    assert result.exit_code == 1
    combined = result.stdout + (result.stderr or "")
    assert "no Phoenix endpoint" in combined


def test_case_lookup_error_not_found_exits_with_2(monkeypatch):
    from tests.benchmarks.tool_invoke import case_history

    def _missing(*, runner, case_id):
        raise case_history.CaseLookupError(
            "case 'x@v1' not found in any version of tool_invoke.ingest"
        )

    monkeypatch.setattr(case_history, "lookup_case", _missing)

    result = runner.invoke(bench_app, ["case", "ingest", "x@v1"])
    assert result.exit_code == 2


def _fake_case_result(**overrides):
    from tests.benchmarks.tool_invoke.case_history import CaseLookupResult

    defaults = dict(
        case_id="save_allergy@v2",
        dataset_name="tool_invoke.ingest",
        dataset_version_id="ver-1",
        example_id="ex-1",
        input={
            "tier": "happy",
            "expected_behavior": "calls save_allergy",
            "prompts": {"en": "I'm allergic to peanuts", "zh": "我对花生过敏"},
            "args_predicate_src": "def predicate(args):\n    return args['substance'] == 'peanut'\n",
            "seed_src": "seed = {'allergies': []}\n",
        },
        output={
            "expected_tool": "save_allergy",
            "expected_tools": ["save_allergy", "update_profile"],
        },
        metadata={"revision": 2},
    )
    defaults.update(overrides)
    return CaseLookupResult(**defaults)


def test_case_pretty_format_prints_human_readable_fields(monkeypatch):
    from tests.benchmarks.tool_invoke import case_history

    monkeypatch.setattr(case_history, "lookup_case", lambda **kw: _fake_case_result())

    result = runner.invoke(bench_app, ["case", "ingest", "save_allergy@v2"])
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    assert "case_id:" in out and "save_allergy@v2" in out
    assert "dataset:" in out and "tool_invoke.ingest" in out
    assert "expected_tool:  save_allergy" in out
    assert "expected_tools: save_allergy, update_profile" in out
    # Prompts dict is rendered language-sorted ([en] then [zh]).
    assert "[en] I'm allergic to peanuts" in out
    assert "[zh] 我对花生过敏" in out
    assert "args_predicate_src:" in out
    assert "seed_src:" in out


def test_case_json_format_dumps_raw_record(monkeypatch):
    from tests.benchmarks.tool_invoke import case_history

    monkeypatch.setattr(case_history, "lookup_case", lambda **kw: _fake_case_result())

    result = runner.invoke(
        bench_app, ["case", "ingest", "save_allergy@v2", "--format", "json"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["case_id"] == "save_allergy@v2"
    assert payload["dataset_name"] == "tool_invoke.ingest"
    assert payload["output"]["expected_tool"] == "save_allergy"


def test_case_legacy_record_falls_back_to_user_prompt(monkeypatch):
    """Records pre-cases_snapshot.json have only ``user_prompt``, no ``prompts``."""
    from tests.benchmarks.tool_invoke import case_history

    legacy = _fake_case_result(
        input={
            "tier": "happy",
            "expected_behavior": "calls save_allergy",
            "user_prompt": "I'm allergic to peanuts (formatted)",
        }
    )
    monkeypatch.setattr(case_history, "lookup_case", lambda **kw: legacy)

    result = runner.invoke(bench_app, ["case", "ingest", "save_allergy@v2"])
    assert result.exit_code == 0
    out = result.stdout
    assert "user_prompt:" in out and "(formatted)" in out
    assert "pre-dates cases_snapshot.json" in out


def test_case_minimal_record_without_optional_blocks(monkeypatch):
    """Bare record (no prompts, no args_predicate_src, no seed_src) — just prints headers."""
    from tests.benchmarks.tool_invoke import case_history

    minimal = _fake_case_result(input={}, output={})
    monkeypatch.setattr(case_history, "lookup_case", lambda **kw: minimal)

    result = runner.invoke(bench_app, ["case", "ingest", "save_allergy@v2"])
    assert result.exit_code == 0
    out = result.stdout
    # Sections that depend on optional fields should NOT appear.
    assert "args_predicate_src:" not in out
    assert "seed_src:" not in out
    assert "prompts:" not in out
    assert "user_prompt:" not in out
    # Unknown fields fall back to the (unknown) marker.
    assert "(unknown)" in out
