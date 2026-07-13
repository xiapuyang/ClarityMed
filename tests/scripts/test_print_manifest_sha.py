"""Tests for the print_manifest_sha operator helper.

Verifies the D2 fix — both manifest field names (``sha256`` used by
typed_basd + xgb, ``sha256_weights`` used by vision) resolve to the
correct declared digest so the ``manifest declares:`` line prints the
right value and the mismatch WARNING triggers only on real mismatches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import print_manifest_sha as pms


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(pms._cfg, "CLARITYMED_HOME", tmp_path)
    return tmp_path


def _write_manifest(
    dir_path: Path,
    *,
    weights_filename: str,
    weights_bytes: bytes,
    algorithm_module: str,
    sha_field: str,
    sha_value: str | None = None,
) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / weights_filename).write_bytes(weights_bytes)
    real_sha = hashlib.sha256(weights_bytes).hexdigest()
    manifest = {
        "algorithm_module": algorithm_module,
        sha_field: sha_value if sha_value is not None else real_sha,
    }
    (dir_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_typed_basd_manifest_declares_sha256_field(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    subpath = Path("symptoms") / "typed_basd_v2"
    _write_manifest(
        fake_home / "models" / subpath,
        weights_filename="weights.pt",
        weights_bytes=b"typed-basd-weights",
        algorithm_module="typed_basd",
        sha_field="sha256",
    )
    rc = pms.main([str(subpath)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "manifest declares" in out
    assert "<missing>" not in out  # sha resolved via "sha256" key


def test_xgb_manifest_uses_weights_pkl_filename(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    subpath = Path("symptoms") / "xgb_v1"
    _write_manifest(
        fake_home / "models" / subpath,
        weights_filename="weights.pkl",
        weights_bytes=b"xgb-joblib-blob",
        algorithm_module="xgb",
        sha_field="sha256",
    )
    rc = pms.main([str(subpath)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "weights.pkl sha256" in out


def test_vision_manifest_declares_sha256_weights_field(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Vision manifests use ``sha256_weights`` — the fix must accept both."""
    subpath = Path("vision") / "busi_v1"
    _write_manifest(
        fake_home / "models" / subpath,
        weights_filename="weights.pt",
        weights_bytes=b"vision-weights",
        algorithm_module="vision_busi",
        sha_field="sha256_weights",
    )
    rc = pms.main([str(subpath)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "<missing>" not in out


def test_mismatched_sha_returns_warning_exit_code(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    subpath = Path("symptoms") / "typed_basd_v2"
    _write_manifest(
        fake_home / "models" / subpath,
        weights_filename="weights.pt",
        weights_bytes=b"real-bytes",
        algorithm_module="typed_basd",
        sha_field="sha256",
        sha_value="0" * 64,  # deliberately wrong
    )
    rc = pms.main([str(subpath)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "WARNING" in err


def test_missing_manifest_returns_1(
    fake_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    subpath = Path("symptoms") / "nonexistent"
    (fake_home / "models" / subpath).mkdir(parents=True)
    rc = pms.main([str(subpath)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "manifest not found" in err
