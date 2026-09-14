"""End-to-end-shaped tests for ``claritymed record …``.

Exercises the typer surface in-process via CliRunner so the WAL
orchestration + JSON event streaming + cleanup gate run for real
against tmp ``DATA_DIR``s.

The OCR-extract and health tests stub the OCR provider at the factory
level so the test suite doesn't depend on installed model weights;
the wiring being tested is "we read the cache then call the provider,"
not provider correctness (which has its own tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

FIXTURES = (
    Path(__file__).parent.parent / "ingest" / "records" / "fixtures" / "templates"
)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def redirected_home(monkeypatch, tmp_path):
    """Repoint DATA_DIR + initialize a 'test' user."""
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path / "data"))
    import importlib

    from claritymed import config as _cfg

    importlib.reload(_cfg)

    from claritymed.stores.account import init_user

    init_user("test", display_name="Test")
    return tmp_path


@pytest.fixture
def record_app():
    """Fresh ``record_app`` typer for invocation. Imports happen here so
    the module reads the redirected DATA_DIR at command time."""
    from claritymed.cli.commands.record import record_app

    return record_app


# --- import-from-template ---------------------------------------------


def test_import_from_template_dry_run(redirected_home, runner, record_app):
    res = runner.invoke(
        record_app,
        [
            "import-from-template",
            str(FIXTURES / "minimal_single_user"),
            "--user",
            "test",
            "--dry-run",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["event"] == "dry_run"
    assert "test" in payload["plan"]
    assert payload["plan"]["test"]["cases"] == 1
    # No state dir should have been created on dry-run.
    state_dir = redirected_home / "data" / "_imports"
    assert not state_dir.exists() or not any(state_dir.iterdir())


def test_import_from_template_happy_path(redirected_home, runner, record_app):
    res = runner.invoke(
        record_app,
        [
            "import-from-template",
            str(FIXTURES / "minimal_single_user"),
            "--user",
            "test",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    lines = [
        json.loads(line) for line in res.stdout.strip().splitlines() if line.strip()
    ]
    summary = lines[-1]
    assert summary["event"] == "summary"
    assert summary["any_error"] is False
    assert summary["row_count"] >= 1
    counts = summary["user_counts"]["test"]
    assert counts["done"] >= 1
    # Template deleted on clean success.
    state_dirs = list((redirected_home / "data" / "_imports").iterdir())
    assert len(state_dirs) == 1
    template_dir = state_dirs[0] / "template"
    assert not template_dir.exists()


def test_import_from_template_idempotent_rerun(redirected_home, runner, record_app):
    """Second run with same template bytes → all rows skipped."""
    first = runner.invoke(
        record_app,
        [
            "import-from-template",
            str(FIXTURES / "minimal_single_user"),
            "--user",
            "test",
            "--json",
        ],
    )
    assert first.exit_code == 0

    second = runner.invoke(
        record_app,
        [
            "import-from-template",
            str(FIXTURES / "minimal_single_user"),
            "--user",
            "test",
            "--json",
        ],
    )
    assert second.exit_code == 0, second.stdout
    lines = [
        json.loads(line) for line in second.stdout.strip().splitlines() if line.strip()
    ]
    summary = lines[-1]
    counts = summary["user_counts"]["test"]
    # Every row skipped (or recovered — same intent: no fresh writes).
    assert counts.get("error", 0) == 0


def test_import_from_template_keep_template_flag(redirected_home, runner, record_app):
    res = runner.invoke(
        record_app,
        [
            "import-from-template",
            str(FIXTURES / "minimal_single_user"),
            "--user",
            "test",
            "--keep-template",
            "--json",
        ],
    )
    assert res.exit_code == 0
    state_dirs = list((redirected_home / "data" / "_imports").iterdir())
    template_dir = state_dirs[0] / "template"
    assert template_dir.exists()


def test_import_from_template_unknown_user_fails(redirected_home, runner, record_app):
    res = runner.invoke(
        record_app,
        [
            "import-from-template",
            str(FIXTURES / "minimal_single_user"),
            "--user",
            "nonexistent",
            "--json",
        ],
    )
    assert res.exit_code == 1
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["event"] == "fatal"


def test_import_from_template_invalid_template_fails_fatally(
    redirected_home, runner, record_app
):
    res = runner.invoke(
        record_app,
        [
            "import-from-template",
            str(FIXTURES / "invalid_extra_file"),
            "--user",
            "test",
            "--json",
        ],
    )
    assert res.exit_code == 1
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["event"] == "fatal"


# --- import-status ----------------------------------------------------


def test_import_status_after_clean_import(redirected_home, runner, record_app):
    res = runner.invoke(
        record_app,
        [
            "import-from-template",
            str(FIXTURES / "minimal_single_user"),
            "--user",
            "test",
            "--json",
        ],
    )
    assert res.exit_code == 0
    lines = [
        json.loads(line) for line in res.stdout.strip().splitlines() if line.strip()
    ]
    summary = lines[-1]
    import_id = summary["import_id"]

    status_res = runner.invoke(
        record_app,
        ["import-status", import_id, "--user", "test", "--json"],
    )
    assert status_res.exit_code == 0, status_res.stdout
    payload = json.loads(status_res.stdout.strip().splitlines()[-1])
    assert payload["import_id"] == import_id
    assert payload["row_count"] >= 1


# --- import-resume ----------------------------------------------------


def test_import_resume_after_template_deleted_fails(
    redirected_home, runner, record_app
):
    """If template/ was cleaned up after a clean import, resume can't
    re-derive the bundle — fail loudly so the user knows."""
    first = runner.invoke(
        record_app,
        [
            "import-from-template",
            str(FIXTURES / "minimal_single_user"),
            "--user",
            "test",
            "--json",
        ],
    )
    assert first.exit_code == 0
    summary = [
        json.loads(line) for line in first.stdout.strip().splitlines() if line.strip()
    ][-1]
    import_id = summary["import_id"]
    # Template was cleaned up (no errors, no --keep-template).
    res = runner.invoke(
        record_app,
        ["import-resume", import_id, "--user", "test", "--json"],
    )
    assert res.exit_code == 1
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["event"] == "fatal"


# --- health -----------------------------------------------------------


def test_health_passes_under_default_config(
    redirected_home, runner, record_app, monkeypatch
):
    """Default ``configs/ocr.yaml`` ships ``phi_policy: local-only`` and
    local providers — health should pass cleanly.

    Stubs ``make_ocr_provider`` because the project's default chain
    contains an LLM-backed entry that needs an env-var API key the test
    env doesn't have. The behavior under test is "health declares OK
    when the build succeeds and every chain entry is local," not the
    provider build itself."""

    class _StubLocalProvider:
        is_local = True

    class _StubRouter:
        document_chain = [_StubLocalProvider()]
        image_chain = [_StubLocalProvider()]

    monkeypatch.setattr(
        "claritymed.core.ocr.factory.make_ocr_provider",
        lambda: _StubRouter(),
    )
    monkeypatch.setattr(
        "claritymed.cli.commands.record.make_ocr_provider",
        lambda: _StubRouter(),
        raising=False,
    )

    res = runner.invoke(record_app, ["health"])
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True, payload
    assert res.exit_code == 0


def test_health_fails_when_phi_policy_not_local_only(
    redirected_home, runner, record_app, monkeypatch
):
    """Mutate the loaded config so phi_policy != local-only → health
    refuses to declare ready."""
    from claritymed.core.schemas.ocr import load_ocr_config

    real_cfg = load_ocr_config()
    fake = real_cfg.model_copy(update={"phi_policy": "any"})
    monkeypatch.setattr("claritymed.core.schemas.ocr.load_ocr_config", lambda: fake)

    res = runner.invoke(record_app, ["health"])
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert "phi_policy_not_local_only" in payload["failed"]
    assert res.exit_code == 1


def test_health_fails_when_no_user_account(monkeypatch, tmp_path, runner, record_app):
    monkeypatch.setenv("CLARITYMED_DATA_DIR", str(tmp_path / "data"))
    import importlib

    from claritymed import config as _cfg

    importlib.reload(_cfg)

    res = runner.invoke(record_app, ["health"])
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert "no_user_account" in payload["failed"]
    assert res.exit_code == 1


# --- ocr-extract ------------------------------------------------------


def test_ocr_extract_rejects_when_phi_policy_not_local_only(
    redirected_home, runner, record_app, monkeypatch, tmp_path
):
    """Identical gate to ``health`` — refuses to OCR if the policy
    would let cloud providers see PHI."""
    from claritymed.core.schemas.ocr import load_ocr_config

    real_cfg = load_ocr_config()
    fake = real_cfg.model_copy(update={"phi_policy": "any"})
    monkeypatch.setattr("claritymed.core.schemas.ocr.load_ocr_config", lambda: fake)

    file = tmp_path / "doc.txt"
    file.write_text("hello")

    res = runner.invoke(
        record_app,
        ["ocr-extract", str(file), "--user", "test", "--json"],
    )
    assert res.exit_code == 1
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["failed_reason"] == "phi_policy_not_local_only"


def test_ocr_extract_cache_hit_short_circuits(
    redirected_home, runner, record_app, tmp_path
):
    """Pre-populate the sentinel for a sha; ocr-extract returns
    cache_hit=true without invoking the OCR provider."""
    import hashlib

    from claritymed.stores.blob_store import BlobStore

    file = tmp_path / "doc.txt"
    content = b"hello cached"
    file.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()

    blob = BlobStore("test")
    blob.store(content, "txt")
    blob.write_ocr_result(
        sha,
        status="ok",
        kind="ocr",
        ext="txt",
        provider="stub",
        chain_tried=["stub"],
        reason=None,
        text="hello cached body",
        original_filename="doc.txt",
    )
    assert blob.ocr_done(sha)

    res = runner.invoke(
        record_app,
        ["ocr-extract", str(file), "--user", "test", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["cache_hit"] is True
    assert payload["sha256"] == sha
    assert payload["text"] == "hello cached body"
