"""Dataset + model weights loading with two-level manifest sha256 chain.

The chain (KTD-6 in the disease-prediction plan):

1. The committed ``configs/symptoms.yaml`` pins ``models[*].manifest_sha256``.
2. The server hashes the on-disk ``manifest.json`` against (1) at startup
   and refuses to start on mismatch.
3. The manifest's own ``sha256`` field pins the weights file digest.
4. The server hashes the on-disk ``weights.pt`` against (3) and refuses
   to start on mismatch.

A tampered manifest pointing at substituted weights cannot survive both
links because (1) is rooted in the committed config.

This module is testable in isolation: every callable takes paths +
specs as arguments and returns either a :class:`DatasetLoaded` or
raises a clear :class:`RuntimeError`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


from claritymed import config as _cfg
from claritymed.core.device import resolve_device
from claritymed.core.symptoms.schemas import DatasetSpec, ModelSpec, SymptomsConfig
from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_pidx,
)
from claritymed.ingest.symptoms.typed_basd import TypedEnv, build_basd
from claritymed.servers.symptoms.state import DatasetLoaded


def sha256_file(path: Path) -> str:
    """Return the lowercase-hex SHA-256 of ``path`` contents."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def models_root() -> Path:
    """Return ``CLARITYMED_HOME/models/symptoms/`` (created lazily by callers)."""
    return _cfg.CLARITYMED_HOME / "models" / "symptoms"


def data_root() -> Path:
    """Return ``CLARITYMED_HOME/data/symptoms/`` (per-dataset subdirs underneath)."""
    return _cfg.CLARITYMED_HOME / "data" / "symptoms"


def verify_manifest_chain(model_spec: ModelSpec, weights_dir: Path) -> dict:
    """Verify (committed config sha) → manifest → weights chain.

    Raises:
        FileNotFoundError: Manifest or weights file is missing.
        RuntimeError: SHA-256 mismatch at either link, or manifest is
            structurally wrong (missing ``sha256`` field).

    Returns:
        The parsed manifest dict — surfaced in startup logs + /health so
        an operator can see the eval numbers the deployed checkpoint
        was qualified at.
    """
    manifest_path = weights_dir / "manifest.json"
    weights_path = weights_dir / "weights.pt"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json missing under {weights_dir}; train the dataset "
            f"first via `claritymed-symptoms-train-ddxplus`."
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"weights.pt missing under {weights_dir}; the manifest is "
            f"present but the checkpoint is not."
        )

    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != model_spec.manifest_sha256:
        raise RuntimeError(
            f"manifest sha256 mismatch for model {model_spec.id!r}: "
            f"configs/symptoms.yaml pins {model_spec.manifest_sha256}, "
            f"on-disk {manifest_path} hashes to {actual_manifest_sha}. "
            f"Refusing to start — the manifest may have been tampered with."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_weights_sha = manifest.get("sha256")
    if not declared_weights_sha:
        raise RuntimeError(
            f"manifest {manifest_path} missing 'sha256' field — cannot "
            f"verify weights integrity."
        )
    actual_weights_sha = sha256_file(weights_path)
    if actual_weights_sha != declared_weights_sha:
        raise RuntimeError(
            f"weights sha256 mismatch for model {model_spec.id!r}: "
            f"manifest declares {declared_weights_sha}, on-disk "
            f"{weights_path} hashes to {actual_weights_sha}. "
            f"Refusing to start — the checkpoint does not match its manifest."
        )
    return manifest


def load_torch_agent(
    schema: dict,
    n_dis: int,
    weights_path: Path,
    device: str,
    *,
    stop_mode: str = "heuristic",
) -> Any:
    """Build a fresh :class:`Agent` and load weights from ``weights_path``.

    ``hidden``, ``stop_thres``, and ``temp`` are all read from the
    checkpoint so train-time and serve-time always stay in sync regardless
    of which hyperparameters the Optuna search settled on.
    """
    import torch

    state = torch.load(weights_path, map_location=device)
    hidden = state["trunk"]["0.weight"].shape[0]
    stop_thres = state.get("thres", 0.1)

    # ``build_basd`` reads env.S, env.n_ev, env.context_size, env.off —
    # it does NOT iterate env.patients, so an empty patient list is fine.
    seed_env = TypedEnv([], schema, n_dis)
    agent = build_basd(
        seed_env,
        n_dis=n_dis,
        hidden=hidden,
        lr=1e-4,
        device=device,
        stop_thres=stop_thres,
        stop_mode=stop_mode,
    )
    agent.trunk.load_state_dict(state["trunk"])
    agent.sym.load_state_dict(state["sym"])
    agent.patho.load_state_dict(state["patho"])
    if agent.stop is not None and state.get("stop") is not None:
        agent.stop.load_state_dict(state["stop"])
    agent.thres = state.get("thres", agent.thres)
    agent.temp = state.get("temp", agent.temp)
    return agent


def apply_model_overrides(agent, model_spec: "ModelSpec") -> None:
    """Override runtime-tunable agent params from the model yaml config.

    Called after weight loading so the yaml always wins over checkpoint values.
    """
    if model_spec.patho_temp is not None:
        agent.temp = model_spec.patho_temp
    if model_spec.stop_thres is not None:
        agent.thres = model_spec.stop_thres


def load_dataset(
    spec: DatasetSpec,
    model_spec: ModelSpec,
    *,
    device: str | None = None,
) -> DatasetLoaded:
    """Load schema, severity, agent, and weights for one dataset.

    Path conventions:

    * Schema files (``release_evidences.json`` / ``release_conditions.json``)
      live under ``CLARITYMED_HOME/data/symptoms/<dataset_id>/``.
    * Weights live under
      ``CLARITYMED_HOME/models/symptoms/<weights_subpath>/{manifest.json, weights.pt}``.

    For ``dataset_id != "ddxplus"`` the schema loaders here will need a
    factory branch — that's the seam a second dataset slots into.
    """
    device = device or resolve_device("auto")
    data_dir = data_root() / spec.id
    weights_dir = models_root() / model_spec.weights_subpath

    if spec.id != "ddxplus":
        raise NotImplementedError(
            f"dataset {spec.id!r}: only 'ddxplus' has a schema adapter in "
            f"v1. Add a branch in servers/symptoms/loader.py when the "
            f"second dataset lands."
        )

    schema = load_evidence_schema(data_dir)
    pidx, severity = load_pidx(data_dir)
    n_dis = len(pidx)
    condition_names = [name for name, _ in sorted(pidx.items(), key=lambda kv: kv[1])]

    manifest = verify_manifest_chain(model_spec, weights_dir)
    agent = load_torch_agent(
        schema,
        n_dis=n_dis,
        weights_path=weights_dir / "weights.pt",
        device=device,
    )
    apply_model_overrides(agent, model_spec)
    return DatasetLoaded(
        spec=spec,
        model_spec=model_spec,
        schema=schema,
        agent=agent,
        severity=severity,
        condition_names=condition_names,
        manifest=manifest,
    )


def load_enabled_datasets(config: SymptomsConfig) -> dict[str, DatasetLoaded]:
    """Load every dataset with ``enabled=True``; skip the rest.

    Returns a mapping keyed by ``dataset.id`` so the FastAPI router can
    look up by path parameter. Failure on any enabled dataset aborts —
    we never start half-loaded (the operator must see the error).
    """
    by_model_id = {m.id: m for m in config.models}
    out: dict[str, DatasetLoaded] = {}
    for spec in config.datasets:
        if not spec.enabled:
            continue
        model_spec = by_model_id[spec.model_id]
        out[spec.id] = load_dataset(spec, model_spec)
    return out


def flush_mps_cache() -> None:
    """Release MPS-cached pages — called after each agent inference step.

    Mirrors :func:`claritymed.servers.embedder._flush_mps_cache`. PyTorch's
    MPS allocator caches freed tensors in-process and won't return them
    to the OS unprompted; on long-running inference servers this shows
    up as ever-growing phys_footprint.
    """
    import gc

    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
