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
    whitelist = set(spec.disease_whitelist) if spec.disease_whitelist else None
    pidx, severity = load_pidx(data_dir, whitelist=whitelist)

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
                is_antecedent=bool(native.get("is_antecedent", False)),
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
        init_matcher=None,  # type: InitMatcherEmbedder | None
    ) -> LoadedDataset:
        data_dir = _data_dir()
        canonical = _build_canonical(spec, data_dir)
        n_dis = canonical.n_conditions

        models: dict[str, LoadedModel] = {}
        for model_id in spec.model_ids:
            model_spec = model_specs[model_id]
            weights_dir = _models_root() / model_spec.weights_subpath
            manifest = _verify_manifest_chain(model_spec, weights_dir)
            _assert_whitelist_matches_manifest(spec, model_spec, manifest)
            agent = _load_torch_agent(
                canonical.layout,
                n_dis=n_dis,
                weights_path=weights_dir / "weights.pt",
                device=device,
            )
            models[model_id] = LoadedModel(
                spec=model_spec, agent=agent, manifest=manifest
            )

        init_catalog = _maybe_build_init_catalog(spec, canonical, init_matcher)
        return LoadedDataset(
            spec=spec,
            canonical=canonical,
            models=models,
            init_catalog=init_catalog,
        )


def _assert_whitelist_matches_manifest(
    spec: DatasetSpec, model_spec: ModelSpec, manifest: dict
) -> None:
    """Refuse to load when ``spec.disease_whitelist`` doesn't match the
    checkpoint's ``manifest.diseases_trained``.

    The patho head's class-idx contract is baked into the trained weights
    via the alphabetical pidx built from the whitelist at train time. If
    the config's whitelist and the manifest's diseases_trained disagree,
    the pidx built at load time will assign different diseases to the
    same output indices — the model will confidently emit wrong
    differentials in production. Fail loud instead.
    """
    if spec.disease_whitelist is None:
        return
    trained = manifest.get("diseases_trained")
    if trained is None:
        raise RuntimeError(
            f"dataset {spec.id!r} declares disease_whitelist but model "
            f"{model_spec.id!r}'s manifest has no 'diseases_trained' field. "
            f"Either the checkpoint predates the subset-training feature "
            f"(retrain with the current train.py) or the manifest was "
            f"hand-edited. Refusing to load."
        )
    if set(trained) != set(spec.disease_whitelist):
        only_config = sorted(set(spec.disease_whitelist) - set(trained))
        only_manifest = sorted(set(trained) - set(spec.disease_whitelist))
        raise RuntimeError(
            f"dataset {spec.id!r} vs model {model_spec.id!r}: "
            f"disease_whitelist ↔ manifest.diseases_trained mismatch. "
            f"config-only: {only_config}; manifest-only: {only_manifest}. "
            f"Retrain the model with the current whitelist, or point the "
            f"config at a matching checkpoint. Refusing to load — class "
            f"indices would silently misalign."
        )


def _maybe_build_init_catalog(spec, canonical, init_matcher):
    """Build the init-symptom catalog when the spec opts in and the
    matcher is available. ``None`` on any disabled-or-failure path."""
    if not spec.use_initial_symptom_flag:
        return None
    if init_matcher is None:
        return None
    # Local import to avoid pulling sentence-transformers types into the
    # adapter module's import graph when the matcher is disabled.
    from claritymed.core.symptoms.init_matcher import build_catalog

    # Default threshold lives on InitMatcherConfig and is surfaced via
    # ``InitMatcherEmbedder.default_threshold``. Per-dataset override
    # is a future extension on ``DatasetSpec`` — until then, the
    # singleton's value is the single source of truth.
    return build_catalog(
        canonical.evidences,
        spec.init_symptom_filter,
        init_matcher,
        init_matcher.default_threshold,
    )
