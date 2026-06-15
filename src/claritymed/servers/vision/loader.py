"""Vision model loading with two-level manifest sha256 chain (KTD-V7).

The chain mirrors the symptoms loader (``servers/symptoms/loader.py``):

1. ``configs/vision.yaml::models[*].manifest_sha256`` is the committed root.
2. Server hashes ``manifest.json`` and refuses to start on mismatch with (1).
3. The manifest's own ``sha256_weights`` pins the on-disk weights digest.
4. Server hashes the weights file and refuses to start on mismatch with (3).

Any tampered manifest pointing at substituted weights cannot survive both
links because (1) is rooted in the committed config.

Framework dispatch is keyed on ``manifest.framework`` (``pytorch`` / ``onnx`` /
``ultralytics``). Adapters declare themselves through a small registry so
adding a new framework is one file + one ``register_adapter`` call — no
edit to ``load_model_for_spec``.

The module is testable in isolation: every callable takes paths + specs as
arguments and returns either a ``DiseaseVisionModel`` or raises a clear
``RuntimeError``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Protocol

from claritymed import config as _cfg
from claritymed.core.vision.schemas import (
    DiseaseVisionModel,
    Manifest,
    ModelFramework,
    ModelSpec,
)

logger = logging.getLogger(__name__)

# Reasoning about path layout in one place: weights live under
# ``CLARITYMED_HOME/models/<weights_subpath>`` per ``ModelSpec``. Splitting
# the helper out of every loader callsite keeps the convention crisp.
WEIGHTS_FILENAME = "weights.pt"
MANIFEST_FILENAME = "manifest.json"


class ModelManifestMismatchError(RuntimeError):
    """Raised when the on-disk manifest digest doesn't match the config pin.

    Distinct exception type so the FastAPI lifespan handler can crash
    with a clear startup error code and Unit 5's ``VisionRegistry``
    can distinguish "wrong manifest" from "weights gone missing".
    """


class WeightsManifestMismatchError(RuntimeError):
    """Raised when the on-disk weights digest doesn't match the manifest's pin."""


def sha256_file(path: Path) -> str:
    """Return the lowercase-hex SHA-256 of ``path`` contents."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def models_root() -> Path:
    """Return ``CLARITYMED_HOME/models/`` (created lazily by callers).

    Vision models share the same root as symptoms — the per-feature
    subdirectory comes from ``ModelSpec.weights_subpath`` (e.g.
    ``vision/breast_cancer_ultrasound/breast_busi_unet_v1``).
    """
    return _cfg.CLARITYMED_HOME / "models"


def resolve_model_dir(spec: ModelSpec, root: Path | None = None) -> Path:
    """Resolve the on-disk directory holding ``manifest.json`` + ``weights.pt``."""
    base = root if root is not None else models_root()
    return base / spec.weights_subpath


def verify_manifest_chain(spec: ModelSpec, model_dir: Path) -> Manifest:
    """Verify (config sha) → manifest → weights chain. Returns parsed Manifest.

    Raises:
        FileNotFoundError: Manifest or weights file is missing.
        ModelManifestMismatchError: ``manifest.json`` digest differs from
            ``ModelSpec.manifest_sha256``.
        WeightsManifestMismatchError: ``weights.pt`` digest differs from
            ``manifest.sha256_weights``.
        ValueError: ``manifest.json`` is not valid JSON or fails
            ``Manifest`` schema validation.
    """
    manifest_path = model_dir / MANIFEST_FILENAME
    weights_path = model_dir / WEIGHTS_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json missing under {model_dir}; train the model "
            f"first (Unit 6 promotes a real BUSI artifact)."
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"weights.pt missing under {model_dir}; the manifest is "
            f"present but the checkpoint is not."
        )

    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != spec.manifest_sha256:
        raise ModelManifestMismatchError(
            f"manifest sha256 mismatch for model {spec.id!r}: "
            f"configs/vision.yaml pins {spec.manifest_sha256}, "
            f"on-disk {manifest_path} hashes to {actual_manifest_sha}. "
            f"Refusing to start — the manifest may have been tampered with."
        )

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = Manifest.model_validate(raw)

    actual_weights_sha = sha256_file(weights_path)
    if actual_weights_sha != manifest.sha256_weights:
        raise WeightsManifestMismatchError(
            f"weights sha256 mismatch for model {spec.id!r}: "
            f"manifest declares {manifest.sha256_weights}, on-disk "
            f"{weights_path} hashes to {actual_weights_sha}. "
            f"Refusing to start — the checkpoint does not match its manifest."
        )

    # Cross-check: ModelSpec.framework must agree with manifest.framework.
    # The config + the manifest both pin it; a drift between them is a
    # config-bug we want to catch at boot rather than half-load.
    if spec.framework != manifest.framework:
        raise ModelManifestMismatchError(
            f"framework drift for model {spec.id!r}: configs/vision.yaml "
            f"declares {spec.framework!r}, manifest declares {manifest.framework!r}"
        )
    if spec.accepted_modality != manifest.accepted_modality:
        raise ModelManifestMismatchError(
            f"accepted_modality drift for model {spec.id!r}: configs/vision.yaml "
            f"declares {spec.accepted_modality!r}, manifest declares "
            f"{manifest.accepted_modality!r}"
        )
    return manifest


# --- adapter registry ----------------------------------------------------


class _AdapterFactory(Protocol):
    """Signature each framework adapter satisfies."""

    def __call__(
        self, *, spec: ModelSpec, manifest: Manifest, weights_path: Path, device: str
    ) -> DiseaseVisionModel: ...


_ADAPTERS: dict[ModelFramework, _AdapterFactory] = {}


def register_adapter(framework: ModelFramework, factory: _AdapterFactory) -> None:
    """Register a framework adapter factory.

    Called once per adapter at import time (see ``adapters/__init__.py``).
    Re-registration overwrites — useful in tests that swap in a stub
    without monkey-patching the registry dict directly.
    """
    _ADAPTERS[framework] = factory


def load_model_for_spec(
    spec: ModelSpec, *, root: Path | None = None, device: str = "cpu"
) -> tuple[Manifest, DiseaseVisionModel]:
    """Verify integrity chain + load the model via its framework adapter.

    Args:
        spec: The ``ModelSpec`` from ``configs/vision.yaml``.
        root: Override for ``CLARITYMED_HOME/models``. Tests pass a
            ``tmp_path`` so fixture artifacts don't have to live in the
            user's real model store.
        device: ``"cpu"`` / ``"cuda"`` / ``"mps"``. Adapter decides what
            to do with it; the loader does not inspect.

    Returns:
        ``(manifest, model)`` — manifest surfaced separately so the
        FastAPI catalog handler can build ``CatalogModel`` entries
        without parsing JSON twice.
    """
    # Importing the adapters package registers every shipped framework
    # via side-effect. Lazy so the loader stays importable when the
    # vision extra isn't installed (the adapters import torch).
    from claritymed.servers.vision import adapters  # noqa: F401 — registers adapters

    model_dir = resolve_model_dir(spec, root=root)
    manifest = verify_manifest_chain(spec, model_dir)
    factory = _ADAPTERS.get(manifest.framework)
    if factory is None:
        raise RuntimeError(
            f"no adapter registered for framework {manifest.framework!r}; "
            f"available: {sorted(_ADAPTERS)!r}. Did you forget to install "
            f"the matching extra?"
        )
    weights_path = model_dir / WEIGHTS_FILENAME
    model = factory(
        spec=spec, manifest=manifest, weights_path=weights_path, device=device
    )
    if not isinstance(model, DiseaseVisionModel):
        # ``runtime_checkable`` on the Protocol catches stale adapters
        # whose method signatures drift away from the Protocol; without
        # this check the mismatch would surface as a confusing
        # AttributeError mid-request.
        raise RuntimeError(
            f"adapter {factory!r} returned an object that does not satisfy "
            f"DiseaseVisionModel Protocol for model {spec.id!r}"
        )
    logger.info(
        "loaded vision model id=%s disease=%s framework=%s device=%s",
        spec.id,
        spec.disease_id,
        manifest.framework,
        device,
    )
    return manifest, model


def registered_frameworks() -> tuple[ModelFramework, ...]:
    """Snapshot of which framework adapters are currently registered.

    Used by tests + by lifespan startup logging so an operator can see
    "we have torch but not onnx" without grep-ing the codebase.
    """
    return tuple(_ADAPTERS.keys())


__all__ = [
    "MANIFEST_FILENAME",
    "ModelManifestMismatchError",
    "WEIGHTS_FILENAME",
    "WeightsManifestMismatchError",
    "load_model_for_spec",
    "models_root",
    "register_adapter",
    "registered_frameworks",
    "resolve_model_dir",
    "sha256_file",
    "verify_manifest_chain",
]
