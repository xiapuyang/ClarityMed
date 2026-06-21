"""Dataset- and task-agnostic utilities used across every forge phase.

Everything here either takes no problem-specific state, or takes its
state as a parameter (e.g. ``check_floors`` reads a metric→floor map
from the caller). Nothing in this module imports a concrete dataset,
adapter, Task, or :mod:`spec` type — that keeps the dependency graph
flat: ``forge.framework`` is the only module that wires datasets to
this layer.

Categories:

* File / process helpers — :func:`sha256_file`, :func:`select_device`,
  :func:`version_tag`.
* Numpy helpers — :func:`softmax`, :func:`argmax_with_threshold`.
* Audit log — :func:`read_latest_entry`, :func:`append_latest_entry`,
  :func:`latest_jsonl_path`.
* ``configs/vision.yaml`` surgical patcher — :func:`patch_vision_yaml`,
  :func:`replace_model_fields`, :func:`vision_yaml_path`.
* Pipeline helpers — :data:`ALL_PHASES`, :func:`parse_phases`,
  :func:`check_floors`, :func:`staging_dir`, :func:`latest_staging_dir`,
  :func:`disease_root`.
* MLflow shim — :func:`log_metrics` (silent no-op when mlflow isn't
  installed).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from claritymed import config as _cfg

logger = logging.getLogger(__name__)

# The canonical phase order. Both pipelines share these four phases,
# and ``parse_phases`` re-orders any operator-typed subset to this
# order so a typo like ``deploy,tune`` runs as ``tune,deploy``.
ALL_PHASES: tuple[str, ...] = ("search", "train", "tune", "deploy")

# Vision feature key used in MLflow experiment names + Optuna study
# names. Forge is vision-only for now; if a sibling forge for symptoms
# lands later it will be parameterised here.
FEATURE = "vision"


# --- file + device --------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return lowercase-hex SHA-256 of ``path`` contents."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_device(torch, *, require_gpu: bool = False):
    """Pick MPS > CUDA > CPU.

    ``require_gpu=True`` raises :class:`SystemExit` on CPU-only boxes
    so a training run (where CPU is unworkable) fails fast. Tune /
    inference paths pass the default to allow CPU fallback.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise SystemExit(
            "no MPS or CUDA device detected. Training on CPU is unworkable — "
            "point this script at a machine with a GPU."
        )
    return torch.device("cpu")


def version_tag() -> str:
    """UTC timestamp used as the immutable version suffix on artifacts."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# --- numpy helpers --------------------------------------------------------


def softmax(logits):
    """Row-wise softmax over the last axis (numpy only, batch-safe)."""
    import numpy as np

    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def argmax_with_threshold(
    probs,
    *,
    critical_indices: Iterable[int],
    threshold: float,
):
    """Pick argmax but require any critical class to clear ``threshold``.

    For each row whose raw argmax is in ``critical_indices`` but whose
    probability for that class is below ``threshold``, fall back to the
    best non-critical class. Mirrors the
    ``BUSI _argmax_with_threshold`` semantics generalised over an
    arbitrary set of "critical" class indices (BUSI: ``{malignant}``;
    chest CT: ``{adeno, large, squamous}``).
    """
    import numpy as np

    critical = list(critical_indices)
    raw = probs.argmax(axis=1)
    if not critical:
        return raw

    row_idx = np.arange(len(probs))
    is_critical_arg = np.isin(raw, critical)
    crit_probs = probs[row_idx, raw]
    mask = is_critical_arg & (crit_probs < threshold)
    if not mask.any():
        return raw

    # Zero out every critical class for masked rows so the fallback
    # picks the best NON-critical class. Within a row this is
    # symmetric across critical classes (if the top critical class
    # didn't clear the bar, neither did the next).
    alt = probs.copy()
    for ci in critical:
        alt[mask, ci] = -1.0
    fallback = alt.argmax(axis=1)
    final = raw.copy()
    final[mask] = fallback[mask]
    return final


# --- staging + disease roots ---------------------------------------------


def disease_root(dataset_id: str) -> Path:
    """``~/.claritymed/models/vision/<dataset_id>/`` (parent of staging + stable)."""
    return _cfg.CLARITYMED_HOME / "models" / "vision" / dataset_id


def staging_dir(*, dataset_id: str, model_id: str) -> Path:
    """Fresh timestamped staging dir for one train run."""
    return disease_root(dataset_id) / "run" / f"{model_id}_{version_tag()}"


def latest_staging_dir(*, dataset_id: str, model_id: str) -> Path:
    """Most recent staging dir for ``(dataset_id, model_id)``.

    Used by tune + deploy when an operator omits ``--staging-dir`` and
    wants to act on the freshest run. Sort by directory name (UTC
    timestamp is lex-sortable).
    """
    run_dir = disease_root(dataset_id) / "run"
    candidates = sorted(
        (p for p in run_dir.glob(f"{model_id}_*") if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            f"no staging dirs under {run_dir} — run train.py first or pass --staging-dir."
        )
    return candidates[0]


# --- LATEST.jsonl audit log ----------------------------------------------


def latest_jsonl_path(dataset_id: str) -> Path:
    """Append-only deploy audit log; one file per dataset.

    Multiple models for the same dataset share the file; consumers
    that care about a single model filter rows by ``entry["model_id"]``.
    """
    return disease_root(dataset_id) / "LATEST.jsonl"


def read_latest_entry(
    path: Path,
    *,
    model_id: str | None = None,
) -> dict[str, Any] | None:
    """Return the last non-empty JSON object in ``path``, or ``None``.

    When ``model_id`` is given, return the last entry matching that
    model_id (so the regression gate compares apples-to-apples when
    one dataset has multiple models promoted independently). When
    ``model_id`` is ``None``, return the absolute last row regardless
    of model — kept for backwards-compat with the single-model
    busi tests.
    """
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if model_id is not None and entry.get("model_id") != model_id:
            continue
        last = entry
    return last


def append_latest_entry(path: Path, entry: dict[str, Any]) -> None:
    """Append one JSON line; create the file + parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# --- configs/vision.yaml surgical patch ----------------------------------


def vision_yaml_path() -> Path:
    """Path to the committed ``configs/vision.yaml``."""
    return _cfg.PROJECT_ROOT / "configs" / "vision.yaml"


def replace_model_fields(
    text: str,
    *,
    model_id: str,
    new_weights_subpath: str,
    new_manifest_sha: str,
) -> str:
    """Surgical line-based replace of two fields under the matching model entry.

    Preserves all comments + ordering (PyYAML roundtrip would otherwise
    strip them). The caller is expected to re-parse the result via
    :class:`~claritymed.core.vision.schemas.VisionConfig` to confirm
    schema validity before committing the edit to disk.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_target_block = False
    block_indent = 0
    replaced_weights = False
    replaced_sha = False

    id_pattern = re.compile(r"^(\s*-\s*id:\s*)([A-Za-z0-9_\-]+)\s*$")

    for line in lines:
        match = id_pattern.match(line)
        if match:
            entry_id = match.group(2)
            if entry_id == model_id:
                in_target_block = True
                block_indent = len(match.group(1)) - len("- id: ")
            elif in_target_block:
                in_target_block = False
            out.append(line)
            continue
        if in_target_block:
            stripped = line.lstrip(" ")
            indent = len(line) - len(stripped)
            if stripped.startswith(
                ("servers:", "models:", "tool:", "ocr_report:", "diseases:")
            ):
                in_target_block = False
            elif indent > block_indent:
                if stripped.startswith("weights_subpath:"):
                    line = " " * indent + f"weights_subpath: {new_weights_subpath}\n"
                    replaced_weights = True
                elif stripped.startswith("manifest_sha256:"):
                    line = " " * indent + f'manifest_sha256: "{new_manifest_sha}"\n'
                    replaced_sha = True
        out.append(line)

    if not replaced_weights:
        raise SystemExit(
            f"configs/vision.yaml: could not find weights_subpath line under "
            f"model id={model_id!r}"
        )
    if not replaced_sha:
        raise SystemExit(
            f"configs/vision.yaml: could not find manifest_sha256 line under "
            f"model id={model_id!r}"
        )
    return "".join(out)


def assert_vision_yaml_has_model(model_id: str) -> None:
    """Fail loud if ``configs/vision.yaml`` has no ``models[].id == model_id``.

    Called by ``_run_deploy_phase`` BEFORE any filesystem side effect
    (copytree, symlink swap, LATEST.jsonl append). Without this early
    gate a registry-side miss only surfaces after the half-promoted
    artifact is already on disk and the stable ``<model_id>`` symlink
    already points at it — leaving the vision-server pointed at a model
    the config doesn't know about. The remediation hint here doubles as
    the operator-facing instruction for the chest_xray_pneumonia-style
    "trained the model but never wired the registry" case.
    """
    yaml_path = vision_yaml_path()
    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    models = parsed.get("models", [])
    if not any(m.get("id") == model_id for m in models):
        raise SystemExit(
            f"configs/vision.yaml has no model entry with id={model_id!r}. "
            f"Add a `- id: {model_id}` block under `models:` (mirror an "
            f"existing entry's shape — disease_id, server_id, framework, "
            f"accepted_modality, weights_subpath, manifest_sha256, "
            f"expected_ms) and re-run deploy. Placeholder values for "
            f"weights_subpath / manifest_sha256 are fine; deploy overwrites "
            f"them with the real version_tag and hash."
        )


def patch_vision_yaml(
    *,
    model_id: str,
    new_weights_subpath: str,
    new_manifest_sha: str,
) -> None:
    """Atomically patch ``configs/vision.yaml`` for ``model_id``.

    Re-parses the result via ``VisionConfig`` so a malformed edit aborts
    before the file is committed.
    """
    assert_vision_yaml_has_model(model_id)

    yaml_path = vision_yaml_path()
    original = yaml_path.read_text(encoding="utf-8")

    edited = replace_model_fields(
        original,
        model_id=model_id,
        new_weights_subpath=new_weights_subpath,
        new_manifest_sha=new_manifest_sha,
    )

    # Schema check happens last so a broken edit doesn't touch disk.
    from claritymed.core.vision.schemas import VisionConfig

    re_parsed = yaml.safe_load(edited)
    VisionConfig.model_validate(re_parsed)

    tmp_path = yaml_path.with_suffix(yaml_path.suffix + ".tmp")
    tmp_path.write_text(edited, encoding="utf-8")
    tmp_path.replace(yaml_path)


# --- pipeline / floor helpers --------------------------------------------


def parse_phases(
    raw: str, *, all_phases: tuple[str, ...] = ALL_PHASES
) -> tuple[str, ...]:
    """Validate ``--phases``, drop dups, normalise to canonical order.

    A typo surfaces as a clear ``SystemExit`` naming the unknown phase
    rather than a silent skip — the orchestrator's contract with the
    operator is "what you asked for is what runs".
    """
    items = tuple(x.strip() for x in raw.split(",") if x.strip())
    unknown = [item for item in items if item not in all_phases]
    if unknown:
        raise SystemExit(
            f"--phases got unknown {unknown!r}; valid: {list(all_phases)!r}"
        )
    return tuple(p for p in all_phases if p in items)


def check_floors(
    breakdown: dict[str, float],
    floors_map: dict[str, float],
) -> None:
    """Raise :class:`SystemExit` listing every metric below its floor.

    Reports every violation in one pass so the operator doesn't have
    to re-run deploy three times to find every miss.
    """
    fails: list[str] = []
    for name, min_val in floors_map.items():
        actual = breakdown.get(name, 0.0)
        if actual < min_val:
            fails.append(f"{name}={actual:.3f} < floor {min_val}")
    if fails:
        raise SystemExit("floor gate failed: " + "; ".join(fails))


def scalar_only(breakdown: dict[str, Any]) -> dict[str, float]:
    """Drop non-scalar entries from a breakdown dict.

    Breakdowns produced by classification tasks now nest per-class stats
    under ``per_class`` (a dict). Consumers that need flat ``{name: float}``
    pairs — MLflow's ``log_metric``, the per-epoch ``training_curve.json``
    rows, ``check_floors``, ``feasibility_aware_score`` — call this to
    strip the nested block in one place instead of each site
    open-coding the filter.
    """
    return {k: float(v) for k, v in breakdown.items() if isinstance(v, (int, float))}


def class_weight_tensor_from_splits(
    splits, num_classes: int, scheme: str, *, device=None
):
    """Build the per-class weight tensor for ``F.cross_entropy(weight=...)``.

    Schemes:

    * ``"none"`` → returns ``None`` (caller passes that straight to
      ``F.cross_entropy`` for plain CE).
    * ``"inverse_freq"`` → ``w_c ∝ 1/N_c``, mean-normalised to ~1 so the
      logged loss magnitude stays comparable to the unweighted run.
    * ``"sqrt_inv_freq"`` → ``w_c ∝ 1/√N_c``, the gentler variant.

    Counts read from ``splits.train._items``; no image decode. Empty /
    missing classes fall back to ``N=1`` so a tiny dataset with an
    unrepresented class in train doesn't divide by zero.
    """
    if scheme == "none":
        return None
    if scheme not in ("inverse_freq", "sqrt_inv_freq"):
        raise ValueError(f"unknown class_weight scheme: {scheme!r}")

    from collections import Counter

    import torch  # local — avoid torch import at module load time

    counts = Counter(item.label for item in splits.train._items)  # noqa: SLF001
    raw: list[float] = []
    for c in range(num_classes):
        n = max(counts.get(c, 0), 1)
        raw.append(1.0 / n if scheme == "inverse_freq" else 1.0 / (n**0.5))
    mean = sum(raw) / len(raw)
    normalised = [r / mean for r in raw]
    return torch.tensor(normalised, dtype=torch.float32, device=device)


def dataset_stats_from_splits(splits, labels: tuple[str, ...]) -> dict[str, Any]:
    """Per-split class counts + ratios + imbalance ratio.

    Pure function over the train / val / test torch Datasets returned by
    ``DatasetSpec.build_splits()``. Reads each Dataset's ``_items``
    sample list to pull labels without triggering ``__getitem__`` (i.e.
    no image decode). All datasets in this module wrap their sample list
    on ``_items``; the coupling is intentional and project-local.

    Returns a JSON-friendly dict written into ``eval_metrics.json`` so
    every deployed model carries the training-time class distribution
    next to its eval metrics.
    """
    from collections import Counter

    out_splits: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        ds = getattr(splits, split_name)
        counts = Counter(item.label for item in ds._items)  # noqa: SLF001
        total = sum(counts.values())
        per_class = {
            labels[idx]: {
                "count": int(counts.get(idx, 0)),
                "ratio": (counts.get(idx, 0) / total) if total > 0 else 0.0,
            }
            for idx in range(len(labels))
        }
        # imbalance_ratio over present classes (skip absent so a class
        # missing from one split — possible if a tiny dataset's hash
        # split lands no samples there — doesn't divide by zero).
        present_counts = [c for c in counts.values() if c > 0]
        imbalance = max(present_counts) / min(present_counts) if present_counts else 0.0
        out_splits[split_name] = {
            "total": total,
            "per_class": per_class,
            "imbalance_ratio": imbalance,
        }
    return {"labels": list(labels), "splits": out_splits}


# --- MLflow shim ---------------------------------------------------------


def log_metrics(
    metrics: dict[str, float],
    *,
    step: int | None = None,
) -> None:
    """Log a metrics dict to the active MLflow run, no-op without mlflow."""
    try:
        import mlflow
    except ImportError:
        return
    if step is None:
        mlflow.log_metrics(metrics)
    else:
        mlflow.log_metrics(metrics, step=step)


__all__ = [
    "ALL_PHASES",
    "FEATURE",
    "append_latest_entry",
    "argmax_with_threshold",
    "assert_vision_yaml_has_model",
    "check_floors",
    "class_weight_tensor_from_splits",
    "dataset_stats_from_splits",
    "disease_root",
    "latest_jsonl_path",
    "latest_staging_dir",
    "log_metrics",
    "parse_phases",
    "patch_vision_yaml",
    "read_latest_entry",
    "replace_model_fields",
    "scalar_only",
    "select_device",
    "sha256_file",
    "softmax",
    "staging_dir",
    "version_tag",
    "vision_yaml_path",
]
