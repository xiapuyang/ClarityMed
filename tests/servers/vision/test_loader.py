"""Loader + sha256 chain enforcement (Unit 4).

Mirrors the symptoms loader's chain semantics: the committed config
pins the manifest digest, the manifest pins the weights digest, and a
mismatch at either link aborts startup.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from claritymed.core.vision.schemas import DiseaseVisionModel, Manifest
from claritymed.servers.vision.loader import (
    ModelManifestMismatchError,
    WeightsManifestMismatchError,
    load_model_for_spec,
    registered_frameworks,
    sha256_file,
    verify_manifest_chain,
    resolve_model_dir,
)


# --- happy path -----------------------------------------------------------


def test_verify_manifest_chain_returns_parsed_manifest(
    make_vision_artifact, vision_models_root: Path
) -> None:
    spec, manifest_path, weights_path = make_vision_artifact()
    manifest = verify_manifest_chain(spec, manifest_path.parent)
    # Validated as a Manifest pydantic model, not just a dict.
    assert isinstance(manifest, Manifest)
    assert manifest.framework == "pytorch"
    # sha cross-references the file we just wrote.
    assert sha256_file(weights_path) == manifest.sha256_weights
    # The fixture wires a cancer-class model; mapping must cover labels.
    assert manifest.cancer_status_mapping is not None
    assert set(manifest.cancer_status_mapping) == {"benign", "malignant", "normal"}


def test_load_model_for_spec_returns_a_protocol_satisfying_adapter(
    make_vision_artifact, vision_models_root: Path
) -> None:
    spec, _, _ = make_vision_artifact()
    manifest, model = load_model_for_spec(spec, root=vision_models_root, device="cpu")
    assert isinstance(model, DiseaseVisionModel)  # runtime_checkable
    assert manifest.model_id == spec.id


# --- chain enforcement ---------------------------------------------------


def test_manifest_mismatch_raises_with_offender(
    make_vision_artifact, vision_models_root: Path
) -> None:
    spec, manifest_path, _ = make_vision_artifact(tamper="manifest")
    with pytest.raises(ModelManifestMismatchError) as ei:
        verify_manifest_chain(spec, manifest_path.parent)
    # Error names the file path + both expected/actual hashes — operators
    # need that to identify which artifact drifted.
    assert str(manifest_path) in str(ei.value)
    assert spec.manifest_sha256 in str(ei.value)


def test_weights_mismatch_raises(
    make_vision_artifact, vision_models_root: Path
) -> None:
    spec, manifest_path, weights_path = make_vision_artifact(tamper="weights")
    # The manifest was written with the *untampered* weights sha; the
    # extra byte appended after means actual digest now differs.
    with pytest.raises(WeightsManifestMismatchError) as ei:
        verify_manifest_chain(spec, manifest_path.parent)
    assert str(weights_path) in str(ei.value)


def test_missing_manifest_raises_file_not_found(
    make_vision_artifact, vision_models_root: Path
) -> None:
    spec, manifest_path, _ = make_vision_artifact()
    manifest_path.unlink()
    with pytest.raises(FileNotFoundError) as ei:
        verify_manifest_chain(spec, manifest_path.parent)
    # Hint pushes the operator at the training pipeline.
    assert "Unit 6" in str(ei.value) or "train" in str(ei.value)


def test_missing_weights_raises_file_not_found(
    make_vision_artifact, vision_models_root: Path
) -> None:
    spec, manifest_path, weights_path = make_vision_artifact()
    weights_path.unlink()
    with pytest.raises(FileNotFoundError) as ei:
        verify_manifest_chain(spec, manifest_path.parent)
    assert "weights.pt" in str(ei.value)


# --- cross-checks --------------------------------------------------------


def test_framework_drift_between_config_and_manifest_is_detected(
    make_vision_artifact, vision_models_root: Path, monkeypatch
) -> None:
    """Spec says pytorch but manifest somehow says onnx."""
    spec, manifest_path, _ = make_vision_artifact(framework="pytorch")
    # Rewrite the manifest to claim ONNX while keeping the same weights
    # sha so the *first* chain link still passes — only the framework
    # cross-check should fire. We need to also update the spec's pinned
    # manifest_sha256 to match the rewritten file.
    raw = manifest_path.read_text(encoding="utf-8")
    raw = raw.replace('"framework": "pytorch"', '"framework": "onnx"')
    manifest_path.write_text(raw, encoding="utf-8")
    new_sha = sha256_file(manifest_path)
    # Use private replacement via model_copy(update=) → spec is frozen.
    drifted_spec = spec.model_copy(update={"manifest_sha256": new_sha})
    with pytest.raises(ModelManifestMismatchError) as ei:
        verify_manifest_chain(drifted_spec, manifest_path.parent)
    assert "framework drift" in str(ei.value)


def test_modality_drift_between_config_and_manifest_is_detected(
    make_vision_artifact, vision_models_root: Path
) -> None:
    spec, manifest_path, _ = make_vision_artifact(accepted_modality="ultrasound")
    raw = manifest_path.read_text(encoding="utf-8")
    raw = raw.replace('"accepted_modality": "ultrasound"', '"accepted_modality": "ct"')
    manifest_path.write_text(raw, encoding="utf-8")
    new_sha = sha256_file(manifest_path)
    drifted_spec = spec.model_copy(update={"manifest_sha256": new_sha})
    with pytest.raises(ModelManifestMismatchError) as ei:
        verify_manifest_chain(drifted_spec, manifest_path.parent)
    assert "accepted_modality drift" in str(ei.value)


# --- adapter registry --------------------------------------------------


def test_registered_frameworks_includes_pytorch_and_onnx() -> None:
    """Importing the package registers both shipped adapters."""
    # The adapters import side-effect runs the moment loader is imported
    # via ``load_model_for_spec``; but for a pure-registry test we import
    # the package directly.
    import claritymed.servers.vision.adapters  # noqa: F401

    frameworks = set(registered_frameworks())
    assert "pytorch" in frameworks
    assert "onnx" in frameworks


def test_onnx_adapter_dispatch_fails_loudly_on_non_onnx_weights(
    make_vision_artifact, vision_models_root: Path
) -> None:
    """ONNX framework is wired; a manifest/file mismatch fails at session load.

    The fixture writes a ``weights.pt`` torch checkpoint. With
    ``framework="onnx"`` in the manifest, the loader dispatches to
    ``OnnxAdapter`` which feeds the file to ``onnxruntime`` — the
    parser surfaces ``InvalidProtobuf`` because the bytes aren't an
    ONNX proto. This is the failure path we want at startup if a
    misconfigured manifest claims ``framework: onnx`` but the artifact
    is actually a torch checkpoint.

    The real happy path (a genuine ``.onnx`` file flowing through the
    adapter) is covered by ``tests/servers/vision/test_onnx_adapter.py``,
    which builds a tiny real ONNX model via ``torch.onnx.export``.
    """
    import onnxruntime as ort  # noqa: PLC0415

    spec, _manifest_path, _ = make_vision_artifact(framework="onnx")
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidProtobuf):
        load_model_for_spec(spec, root=vision_models_root, device="cpu")


# --- path resolution -----------------------------------------------------


def test_resolve_model_dir_concatenates_weights_subpath(
    make_vision_artifact, vision_models_root: Path
) -> None:
    spec, manifest_path, _ = make_vision_artifact()
    resolved = resolve_model_dir(spec, root=vision_models_root)
    assert resolved == manifest_path.parent
    # Defensive: the spec subpath is relative — it must not escape the root.
    assert vision_models_root in resolved.parents


def test_sha256_file_streams_large_inputs(tmp_path: Path) -> None:
    """``sha256_file`` reads in chunks — verify it matches hashlib.sha256 on the same bytes."""
    blob = (b"\xab" * 1024) * 200  # ~200 KB
    path = tmp_path / "big.bin"
    path.write_bytes(blob)
    expected = hashlib.sha256(blob).hexdigest()
    assert sha256_file(path) == expected
