"""DDXPlus implementation of :class:`DatasetAdapter`.

Reads the native release JSONs from ``CLARITYMED_HOME/data/symptoms/ddxplus/``
and loads the typed-BASD checkpoints declared in :attr:`DatasetSpec.model_ids`.
Returns a :class:`LoadedDataset` whose canonical layer hides every
DDXPlus-specific shape from the server, the questions translator, and
the differential formatter.

Self-registers on import via
:func:`claritymed.core.symptoms.datasets.registry.register_adapter`.
The server's loader doesn't import this module directly — it imports
:mod:`claritymed.ingest.symptoms.ddxplus` (whose ``__init__`` triggers
this module) and calls :func:`build_dataset`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar

from claritymed import config as _cfg
from claritymed.core.symptoms.datasets.canonical import (
    CanonicalCondition,
    CanonicalDataset,
    CanonicalEvidence,
    CanonicalValue,
    LoadedDataset,
    LoadedModel,
    slugify_condition,
)
from claritymed.core.symptoms.schemas import DatasetSpec, ModelSpec
from claritymed.ingest.symptoms.ddxplus.schema import (
    DDXPLUS_CONDITIONS_JSON,
    DDXPLUS_EVIDENCES_JSON,
    DdxplusSchemaError,
    load_evidence_schema,
    load_pidx,
)
from claritymed.ingest.symptoms.typed_basd import TypedEnv, build_basd

DDXPLUS_DATASET_ID = "ddxplus"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_dir() -> Path:
    return _cfg.CLARITYMED_HOME / "data" / "symptoms" / DDXPLUS_DATASET_ID


def _models_root() -> Path:
    return _cfg.CLARITYMED_HOME / "models" / "symptoms"


def _verify_manifest_chain(model_spec: ModelSpec, weights_dir: Path) -> dict:
    manifest_path = weights_dir / "manifest.json"
    weights_path = weights_dir / "weights.pt"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json missing under {weights_dir}; train the dataset "
            f"first via `claritymed-symptoms-train-ddxplus`."
        )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"weights.pt missing under {weights_dir}; manifest present but "
            f"checkpoint is not."
        )
    actual_manifest_sha = _sha256_file(manifest_path)
    if actual_manifest_sha != model_spec.manifest_sha256:
        raise RuntimeError(
            f"manifest sha256 mismatch for model {model_spec.id!r}: "
            f"configs/symptoms.yaml pins {model_spec.manifest_sha256}, "
            f"on-disk {manifest_path} hashes to {actual_manifest_sha}. "
            f"Refusing to load — the manifest may have been tampered with."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("sha256")
    if not declared:
        raise RuntimeError(
            f"manifest {manifest_path} missing 'sha256' field — cannot "
            f"verify weights integrity."
        )
    actual_weights_sha = _sha256_file(weights_path)
    if actual_weights_sha != declared:
        raise RuntimeError(
            f"weights sha256 mismatch for model {model_spec.id!r}: "
            f"manifest declares {declared}, on-disk {weights_path} hashes "
            f"to {actual_weights_sha}. Refusing to load."
        )
    return manifest


def _load_torch_agent(schema: dict, n_dis: int, weights_path: Path, device: str):
    import torch

    state = torch.load(weights_path, map_location=device)
    # Derive hidden from the checkpoint's first Linear so the loader stays
    # in sync with whatever the Optuna search picked at train time
    # (sister implementation: src/claritymed/servers/symptoms/loader.py).
    hidden = state["trunk"]["0.weight"].shape[0]
    stop_thres = state.get("thres", 0.1)
    seed_env = TypedEnv([], schema, n_dis)
    agent = build_basd(
        seed_env,
        n_dis=n_dis,
        hidden=hidden,
        lr=1e-4,
        device=device,
        stop_thres=stop_thres,
        stop_mode="heuristic",
    )
    agent.trunk.load_state_dict(state["trunk"])
    agent.sym.load_state_dict(state["sym"])
    agent.patho.load_state_dict(state["patho"])
    if agent.stop is not None and state.get("stop") is not None:
        agent.stop.load_state_dict(state["stop"])
    agent.thres = state.get("thres", agent.thres)
    agent.temp = state.get("temp", agent.temp)
    return agent


def _build_canonical(spec: DatasetSpec, data_dir: Path) -> CanonicalDataset:
    """Read DDXPlus JSONs + raw evidence text and produce the canonical form."""
    schema = load_evidence_schema(data_dir)
    pidx, severity = load_pidx(data_dir)

    # Raw evidences carry question_en + value_meaning; pull both into
    # CanonicalEvidence for fallback rendering. ``schema["evs"]`` is the
    # build_layout-ordered list (sorted by name) — algorithm idx is its index.
    evidences_path = data_dir / DDXPLUS_EVIDENCES_JSON
    raw_evs = json.loads(evidences_path.read_text(encoding="utf-8"))
    if isinstance(raw_evs, dict):
        raw_by_name = {(e.get("name") or e.get("code")): e for e in raw_evs.values()}
    else:
        raw_by_name = {(e.get("name") or e.get("code")): e for e in raw_evs}

    canonical_evs: list[CanonicalEvidence] = []
    high_specificity = set(spec.severity_high_specificity_evidence_ids)
    for ev_idx, ev_block in enumerate(schema["evs"]):
        ev_id = ev_block["name"]
        native = raw_by_name.get(ev_id) or {}
        # Per-value canonical entries; local_idx matches schema["vmap"].
        vmap = schema["vmap"][ev_idx]
        values = [
            CanonicalValue(raw=raw, local_idx=local)
            for raw, local in sorted(vmap.items(), key=lambda kv: kv[1])
        ]
        # Native question text per language (DDXPlus ships ``en`` + ``fr``).
        native_q = {
            lang: text
            for lang in ("en", "fr")
            if (text := native.get(f"question_{lang}"))
        }
        # value_meaning[raw][lang] → flatten to {raw: {lang: text}}.
        native_v: dict[str, dict[str, str]] = {}
        meaning = native.get("value_meaning") or {}
        for raw, by_lang in meaning.items():
            if not isinstance(by_lang, dict):
                continue
            cleaned = {
                lang: text
                for lang, text in by_lang.items()
                if isinstance(text, str) and text
            }
            if cleaned:
                native_v[str(raw)] = cleaned
        canonical_evs.append(
            CanonicalEvidence(
                id=ev_id,
                idx=ev_idx,
                dtype=ev_block["dtype"],
                values=values,
                native_question_text=native_q,
                native_value_labels=native_v,
                is_high_specificity=ev_id in high_specificity,
            )
        )

    # Conditions: DDXPlus ``release_conditions.json`` keyed by display name.
    conditions_path = data_dir / DDXPLUS_CONDITIONS_JSON
    if not conditions_path.exists():
        raise FileNotFoundError(
            f"{conditions_path} missing — load_pidx would already have "
            f"failed, but reading raw conditions to extract ICD codes."
        )
    raw_conds = json.loads(conditions_path.read_text(encoding="utf-8"))
    raw_cond_items = (
        list(raw_conds.values()) if isinstance(raw_conds, dict) else raw_conds
    )
    raw_by_display: dict[str, dict] = {}
    for entry in raw_cond_items:
        display = entry.get("condition_name") or entry.get("cond-name-eng")
        if display:
            raw_by_display[display] = entry

    canonical_conds: list[CanonicalCondition] = []
    used_slugs: set[str] = set()
    for display, idx in sorted(pidx.items(), key=lambda kv: kv[1]):
        slug = slugify_condition(display)
        # Deduplicate slug collisions (rare; DDXPlus condition_names are
        # unique-ish but slugification can fold edge cases).
        original_slug = slug
        suffix = 2
        while slug in used_slugs:
            slug = f"{original_slug}_{suffix}"
            suffix += 1
        used_slugs.add(slug)
        raw_entry = raw_by_display.get(display) or {}
        icd10 = raw_entry.get("icd10-id") or raw_entry.get("icd10_id")
        native_name: dict[str, str] = {"en": display}
        fr_name = raw_entry.get("cond-name-fr")
        if fr_name:
            native_name["fr"] = fr_name
        sev_value = int(round(float(severity[idx])))
        if sev_value < 1 or sev_value > 5:
            raise DdxplusSchemaError(
                f"condition {display!r} has out-of-range severity "
                f"{sev_value}; expected 1-5."
            )
        canonical_conds.append(
            CanonicalCondition(
                id=slug,
                idx=idx,
                severity=sev_value,
                icd10=icd10,
                native_name=native_name,
            )
        )

    return CanonicalDataset.build(
        id=spec.id,
        evidences=canonical_evs,
        conditions=canonical_conds,
        layout=schema,
        severity_vector=severity,
    )


class DDXPlusAdapter:
    """:class:`DatasetAdapter` implementation for DDXPlus."""

    dataset_id: ClassVar[str] = DDXPLUS_DATASET_ID

    @classmethod
    def load(
        cls,
        spec: DatasetSpec,
        model_specs: dict[str, ModelSpec],
        *,
        device: str,
    ) -> LoadedDataset:
        data_dir = _data_dir()
        canonical = _build_canonical(spec, data_dir)
        n_dis = canonical.n_conditions

        models: dict[str, LoadedModel] = {}
        for model_id in spec.model_ids:
            model_spec = model_specs[model_id]
            weights_dir = _models_root() / model_spec.weights_subpath
            manifest = _verify_manifest_chain(model_spec, weights_dir)
            agent = _load_torch_agent(
                canonical.layout,
                n_dis=n_dis,
                weights_path=weights_dir / "weights.pt",
                device=device,
            )
            models[model_id] = LoadedModel(
                spec=model_spec, agent=agent, manifest=manifest
            )
        return LoadedDataset(spec=spec, canonical=canonical, models=models)
