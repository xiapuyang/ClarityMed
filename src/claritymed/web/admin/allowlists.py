"""Allowlists for admin write paths.

Centralized so a future audit can grep one file to enumerate the entire
admin write surface. Three allowlists live here:

* :data:`EDITABLE_CONFIGS` — YAML files the admin UI may write. Used by
  :func:`claritymed.web.admin.yaml_io.save_yaml` to refuse writes to any
  other path.
* :data:`EDITABLE_KEYS_BY_CONFIG` — per-config dotted-path allowlist for
  the System Config module. Each value is a tuple of dotted paths the
  ``PATCH /admin/configs/{name}`` endpoint will accept; everything else
  raises 400.
* :data:`EDITABLE_MODEL_CATALOGS` — catalog files the Models module may
  full-document overwrite (`models.yaml`, `vision.yaml`, etc.).

The dotted-path entries are matched against the body's ``path`` field via
exact string equality (no glob). Each module updates this file in its own
unit; foundation ships empty tuples so a misfire 400s rather than 500s.
"""

from __future__ import annotations

# System Config module (U5) — extended with concrete per-config keys at
# that unit's landing. Foundation seeds the 6 expected file names so the
# routing layer can fail-loud on unknown names from day one.
EDITABLE_CONFIGS: frozenset[str] = frozenset(
    [
        "app.yaml",
        "retrieval.yaml",
        "safety.yaml",
        "ocr.yaml",
        "uncertainty.yaml",
        "evals.yaml",
    ]
)

# Filled by U5. Empty tuples here mean "no key is editable yet" — the
# endpoint refuses every PATCH until the unit lands its concrete list.
EDITABLE_KEYS_BY_CONFIG: dict[str, tuple[str, ...]] = {
    "app.yaml": (
        "i18n.default_lang",
        "tracing.enabled",
        "paste.max_text_chars",
        "upload.dedupe_cosine_threshold",
    ),
    "retrieval.yaml": (
        "user_rag.top_k",
        "user_rag.rerank_k",
    ),
    "safety.yaml": (
        "phi.privacy_filter.enabled",
        "phi.on_deny",
    ),
    "ocr.yaml": (
        "phi_policy",
        "llm.provider_id",
    ),
    "uncertainty.yaml": ("abstain.threshold",),
    "evals.yaml": ("judge_provider_id",),
}

# Models module (U9) — registered here so the catalog endpoint refuses
# names that aren't in this set.
EDITABLE_MODEL_CATALOGS: frozenset[str] = frozenset(
    [
        "models.yaml",
        "vision.yaml",
        "medical_clip.yaml",
        "symptoms.yaml",
    ]
)


def is_editable_config(name: str) -> bool:
    return name in EDITABLE_CONFIGS


def is_editable_model_catalog(name: str) -> bool:
    return name in EDITABLE_MODEL_CATALOGS


def is_editable_config_key(name: str, dotted_path: str) -> bool:
    return dotted_path in EDITABLE_KEYS_BY_CONFIG.get(name, ())
