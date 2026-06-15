"""Shared fixtures for vision-server tests.

The model layout the loader expects is::

    <root>/<weights_subpath>/manifest.json
    <root>/<weights_subpath>/weights.pt

This conftest gives every test a freshly-minted (manifest, weights, spec)
triple under ``tmp_path`` so sha256 chain assertions stay tight without
shipping precomputed digests in the repo.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from claritymed.core.vision.schemas import ModelSpec


_DEFAULT_LABELS = ["benign", "malignant", "normal"]
_DEFAULT_LABELS_META: dict[str, dict[str, str]] = {
    "benign": {
        "description": "non-cancerous mass",
        "cancer_status": "benign",
        "clinical_action": "routine_followup",
    },
    "malignant": {
        "description": "cancerous lesion",
        "cancer_status": "malignant",
        "clinical_action": "urgent_specialist",
    },
    "normal": {
        "description": "no detectable lesion",
        "cancer_status": "normal",
        "clinical_action": "no_action",
    },
}


@pytest.fixture
def vision_models_root(tmp_path: Path) -> Path:
    """Root under which fixture models live; mirrors CLARITYMED_HOME/models/."""
    root = tmp_path / "models"
    root.mkdir()
    return root


@pytest.fixture
def make_vision_artifact(vision_models_root: Path):
    """Factory: writes a sha-consistent (weights, manifest) pair and returns a ModelSpec.

    Usage::

        spec, manifest_path, weights_path = make_vision_artifact()
        # or override defaults:
        spec, _, _ = make_vision_artifact(cancer_class=True, framework="pytorch")

    Returned ``spec`` references the same ``vision_models_root`` the loader
    will be told to use, so ``load_model_for_spec(spec, root=vision_models_root)``
    walks straight to the artifact.
    """

    def _make(
        *,
        model_id: str = "breast_busi_unet_v1",
        disease_id: str = "breast_cancer_ultrasound",
        framework: str = "pytorch",
        accepted_modality: str = "ultrasound",
        cancer_class: bool = True,
        labels: list[str] | None = None,
        labels_meta: dict[str, dict[str, str]] | None = None,
        tamper: str | None = None,  # "manifest" | "weights" | None
    ) -> tuple[ModelSpec, Path, Path]:
        weights_subpath = f"vision/{disease_id}/{model_id}"
        model_dir = vision_models_root / weights_subpath
        model_dir.mkdir(parents=True, exist_ok=True)

        # Write the weights first so the manifest can pin its sha. The
        # `.pt` is a real torch checkpoint (a dict of tensors) so the
        # Torch adapter's ``torch.load`` call exercises the real format
        # path — corrupt-format errors would be missed by a placeholder
        # byte string.
        import torch

        weights_path = model_dir / "weights.pt"
        torch.save(
            {"head.weight": torch.zeros(len(labels or _DEFAULT_LABELS), 8)},
            weights_path,
        )
        # Capture the untampered digest BEFORE applying any weight tamper:
        # the manifest must pin the pre-tamper sha so the chain check
        # actually catches the post-tamper drift.
        weights_sha = _sha256(weights_path)
        if tamper == "weights":
            with weights_path.open("ab") as fh:
                fh.write(b"\x00")

        used_labels = list(labels or _DEFAULT_LABELS)
        used_labels_meta = labels_meta or {
            k: _DEFAULT_LABELS_META[k] for k in used_labels
        }
        manifest_body: dict[str, Any] = {
            "model_id": model_id,
            "model_version": "v1.0.0",
            "framework": framework,
            "accepted_modality": accepted_modality,
            "sha256_weights": weights_sha,
            "task": "classification+segmentation" if cancer_class else "classification",
            "labels": used_labels,
            "labels_meta": used_labels_meta,
            "cancer_class": cancer_class,
            "supports_saliency": False,
            "supports_tta": False,
        }
        if cancer_class:
            manifest_body["cancer_status_mapping"] = {
                label: used_labels_meta[label]["cancer_status"] for label in used_labels
            }
            manifest_body["clinical_action_mapping"] = {
                label: used_labels_meta[label]["clinical_action"]
                for label in used_labels
            }
        manifest_path = model_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_body, indent=2), encoding="utf-8")
        if tamper == "manifest":
            # Append a comment line. JSON parse still succeeds for the
            # original content via ``manifest.json``, but the digest
            # drifts from the spec's ``manifest_sha256`` pin so the
            # first chain link fires.
            with manifest_path.open("a", encoding="utf-8") as fh:
                fh.write("\n")
        manifest_sha = _sha256(manifest_path)

        # When we tampered the manifest, the spec must *still* pin the
        # original (pre-tamper) sha so the chain check actually fires.
        # We re-compute after writing because we want to know the
        # current on-disk digest first; then if the test pinned a
        # tamper, we patch the spec to use the *expected* (untampered)
        # digest by reverse-engineering: re-write the untampered file
        # under a sibling path, hash it, and use that.
        if tamper == "manifest":
            untampered = model_dir / "manifest.original.json"
            untampered.write_text(json.dumps(manifest_body, indent=2), encoding="utf-8")
            spec_sha = _sha256(untampered)
            untampered.unlink()
        else:
            spec_sha = manifest_sha

        spec = ModelSpec(
            id=model_id,
            disease_id=disease_id,
            server_id="local_default",
            framework=framework,
            accepted_modality=accepted_modality,
            weights_subpath=weights_subpath,
            manifest_sha256=spec_sha,
            expected_ms=800,
        )
        return spec, manifest_path, weights_path

    return _make


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
