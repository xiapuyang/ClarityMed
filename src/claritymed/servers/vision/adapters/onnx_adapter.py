"""ONNX adapter — stub for v1; real implementation lands when a v1.x disease
genuinely needs it (per plan §"Deferred to Separate Tasks").

Filed alongside ``torch_adapter.py`` so the framework dispatch in
``loader.py`` already knows where to look — adding the real adapter is
one file edit, not a registry refactor. Until then, registering against
``"onnx"`` raises ``NotImplementedError`` so a stray ``configs/vision.yaml``
entry with ``framework: onnx`` fails fast at startup with a clear message
rather than silently using the torch path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from claritymed.core.vision.schemas import Manifest, ModelSpec
from claritymed.servers.vision.loader import register_adapter


class OnnxAdapter:
    """v1 placeholder — refuses to construct.

    Kept as a class (not just a raising factory) so the eventual real
    adapter can grow into the same name without churning the registry
    or the loader's type hints.
    """

    def __init__(
        self,
        *,
        spec: ModelSpec,
        manifest: Manifest,
        weights_path: Path,
        device: str,
    ) -> None:  # noqa: ARG002 — signature pinned by the registry contract
        raise NotImplementedError(
            f"ONNX adapter not implemented in v1; model {spec.id!r} "
            "declares framework='onnx'. Either retrain the model with "
            "framework='pytorch' or wait for v1.x to ship the real adapter."
        )


def _factory(
    *, spec: ModelSpec, manifest: Manifest, weights_path: Path, device: str
) -> Any:
    return OnnxAdapter(
        spec=spec, manifest=manifest, weights_path=weights_path, device=device
    )


register_adapter("onnx", _factory)


__all__ = ["OnnxAdapter"]
