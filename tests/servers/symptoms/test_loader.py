"""Loader unit tests — sha256 chain + path helpers + dispatch branches.

The loader module is explicitly designed for isolation testing: every
public callable takes paths + specs as arguments and either returns a
:class:`DatasetLoaded` or raises a clear exception. These tests exercise
the testable surface without loading torch checkpoints — the torch path
(``load_torch_agent``, ``load_dataset`` end-to-end) is covered indirectly
by ``test_app.py`` via the stub agent fixture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from claritymed import config as _cfg
from claritymed.core.symptoms.schemas import DatasetSpec, ModelSpec
from claritymed.servers.symptoms.loader import (
    apply_model_overrides,
    data_root,
    flush_mps_cache,
    load_dataset,
    load_enabled_datasets,
    models_root,
    sha256_file,
    verify_manifest_chain,
)


# --- artifact fixture ----------------------------------------------------


def _write_artifact(
    weights_dir: Path,
    *,
    weights_bytes: bytes = b"weights-placeholder",
    manifest_overrides: dict[str, Any] | None = None,
    drop_sha_field: bool = False,
) -> tuple[Path, Path, str]:
    """Write a manifest.json + weights.pt pair; return paths + manifest sha.

    The manifest's own ``sha256`` field is set to the digest of the
    written ``weights.pt`` so the second chain link passes by default.
    Callers tampering with that link should pass ``manifest_overrides`` to
    point the field at a wrong hash.
    """
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "weights.pt"
    weights_path.write_bytes(weights_bytes)
    weights_sha = hashlib.sha256(weights_bytes).hexdigest()

    manifest: dict[str, Any] = {
        "sha256": weights_sha,
        "model_id": "test_model",
        "trained_on": "ddxplus",
    }
    if drop_sha_field:
        manifest.pop("sha256")
    if manifest_overrides:
        manifest.update(manifest_overrides)

    manifest_path = weights_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = sha256_file(manifest_path)
    return manifest_path, weights_path, manifest_sha


def _model_spec(manifest_sha: str, *, subpath: str = "test_model") -> ModelSpec:
    return ModelSpec(
        id="test_model",
        algorithm_module="claritymed.ingest.symptoms.typed_basd",
        weights_subpath=subpath,
        manifest_sha256=manifest_sha,
        maxstep=10,
    )


# --- sha256_file + path helpers -----------------------------------------


def test_sha256_file_matches_hashlib_on_large_payload(tmp_path: Path) -> None:
    """``sha256_file`` reads in 1MB chunks — verify equivalence vs in-memory hash."""
    blob = (b"\x42" * 1024) * 1300  # ~1.3 MB so the chunk loop iterates twice
    p = tmp_path / "big.bin"
    p.write_bytes(blob)
    assert sha256_file(p) == hashlib.sha256(blob).hexdigest()


def test_sha256_file_empty(tmp_path: Path) -> None:
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")
    assert sha256_file(p) == hashlib.sha256(b"").hexdigest()


def test_models_root_is_under_claritymed_home() -> None:
    root = models_root()
    assert root == _cfg.CLARITYMED_HOME / "models" / "symptoms"


def test_data_root_is_under_claritymed_home() -> None:
    root = data_root()
    assert root == _cfg.CLARITYMED_HOME / "data" / "symptoms"


# --- verify_manifest_chain ----------------------------------------------


def test_verify_manifest_chain_happy_returns_parsed_manifest(tmp_path: Path) -> None:
    weights_dir = tmp_path / "model"
    _, _, manifest_sha = _write_artifact(weights_dir)
    spec = _model_spec(manifest_sha)
    parsed = verify_manifest_chain(spec, weights_dir)
    assert parsed["model_id"] == "test_model"
    assert parsed["trained_on"] == "ddxplus"


def test_verify_manifest_chain_missing_manifest_raises_with_hint(
    tmp_path: Path,
) -> None:
    weights_dir = tmp_path / "model"
    weights_dir.mkdir()
    (weights_dir / "weights.pt").write_bytes(b"w")
    spec = _model_spec("0" * 64)
    with pytest.raises(FileNotFoundError) as ei:
        verify_manifest_chain(spec, weights_dir)
    # Hint points the operator at the training pipeline so they know what
    # to run to produce a manifest.
    assert "manifest.json" in str(ei.value)
    assert "train" in str(ei.value)


def test_verify_manifest_chain_missing_weights_raises(tmp_path: Path) -> None:
    weights_dir = tmp_path / "model"
    weights_dir.mkdir()
    # Write only the manifest — weights file absent.
    (weights_dir / "manifest.json").write_text(
        json.dumps({"sha256": "0" * 64}), encoding="utf-8"
    )
    spec = _model_spec(sha256_file(weights_dir / "manifest.json"))
    with pytest.raises(FileNotFoundError) as ei:
        verify_manifest_chain(spec, weights_dir)
    assert "weights.pt" in str(ei.value)


def test_verify_manifest_chain_manifest_sha_mismatch_raises(tmp_path: Path) -> None:
    weights_dir = tmp_path / "model"
    _write_artifact(weights_dir)
    # Pin the wrong manifest digest in the spec.
    spec = _model_spec("a" * 64)
    with pytest.raises(RuntimeError) as ei:
        verify_manifest_chain(spec, weights_dir)
    msg = str(ei.value)
    assert "manifest sha256 mismatch" in msg
    assert "a" * 64 in msg
    assert spec.id in msg


def test_verify_manifest_chain_manifest_missing_sha_field_raises(
    tmp_path: Path,
) -> None:
    weights_dir = tmp_path / "model"
    _, _, manifest_sha = _write_artifact(weights_dir, drop_sha_field=True)
    spec = _model_spec(manifest_sha)
    with pytest.raises(RuntimeError) as ei:
        verify_manifest_chain(spec, weights_dir)
    assert "missing 'sha256'" in str(ei.value)


def test_verify_manifest_chain_weights_sha_mismatch_raises(tmp_path: Path) -> None:
    weights_dir = tmp_path / "model"
    # Manifest claims weights sha "deadbeef..." but the actual file
    # hashes to something else. Spec is rebuilt to pin the wrong-manifest
    # so the first link passes.
    _, weights_path, manifest_sha = _write_artifact(
        weights_dir, manifest_overrides={"sha256": "d" * 64}
    )
    spec = _model_spec(manifest_sha)
    with pytest.raises(RuntimeError) as ei:
        verify_manifest_chain(spec, weights_dir)
    msg = str(ei.value)
    assert "weights sha256 mismatch" in msg
    assert "d" * 64 in msg
    assert str(weights_path) in msg


# --- apply_model_overrides ----------------------------------------------


def test_apply_model_overrides_with_both_values_set() -> None:
    agent = SimpleNamespace(temp=1.0, thres=0.5)
    spec = ModelSpec(
        id="m",
        algorithm_module="claritymed.ingest.symptoms.typed_basd",
        weights_subpath="m",
        manifest_sha256="0" * 64,
        maxstep=10,
        patho_temp=2.5,
        stop_thres=0.7,
    )
    apply_model_overrides(agent, spec)
    assert agent.temp == 2.5
    assert agent.thres == 0.7


def test_apply_model_overrides_with_none_keeps_checkpoint_values() -> None:
    agent = SimpleNamespace(temp=1.0, thres=0.5)
    spec = ModelSpec(
        id="m",
        algorithm_module="claritymed.ingest.symptoms.typed_basd",
        weights_subpath="m",
        manifest_sha256="0" * 64,
        maxstep=10,
        # Both override fields default to None — checkpoint values win.
    )
    apply_model_overrides(agent, spec)
    assert agent.temp == 1.0
    assert agent.thres == 0.5


# --- load_dataset dispatch ----------------------------------------------


def test_load_dataset_rejects_unknown_dataset_id() -> None:
    """Non-ddxplus dataset ids hit the explicit NotImplementedError seam."""
    spec = DatasetSpec(id="custom_set", model_ids=["m"])
    model_spec = ModelSpec(
        id="m",
        algorithm_module="claritymed.ingest.symptoms.typed_basd",
        weights_subpath="m",
        manifest_sha256="0" * 64,
        maxstep=10,
    )
    with pytest.raises(NotImplementedError) as ei:
        load_dataset(spec, model_spec)
    msg = str(ei.value)
    assert "custom_set" in msg
    assert "ddxplus" in msg


# --- load_enabled_datasets ----------------------------------------------


def test_load_enabled_datasets_skips_disabled_specs() -> None:
    """Disabled datasets never enter the load_dataset hot path."""
    disabled_spec = SimpleNamespace(enabled=False, id="ddxplus", model_id="m")
    model_spec = SimpleNamespace(id="m")
    config = SimpleNamespace(datasets=[disabled_spec], models=[model_spec])
    # SimpleNamespace duck-types the SymptomsConfig surface the function
    # actually reads (``.datasets``, ``.models``) without paying the cost
    # of constructing the full pydantic tree (eligibility catalog +
    # init-matcher) that has nothing to do with the skip-disabled branch.
    out = load_enabled_datasets(config)  # type: ignore[arg-type]
    assert out == {}


# --- flush_mps_cache ----------------------------------------------------


def test_flush_mps_cache_never_raises() -> None:
    """Best-effort path — must swallow torch import / MPS unavailability."""
    # Smoke: on macOS-MPS hosts this hits the empty_cache call; elsewhere
    # the `except Exception` swallow keeps it a no-op. Either way, no
    # exception escapes — the caller (the inference loop) relies on that
    # to drop the call after every step without try/except.
    flush_mps_cache()
