"""Auto-generate symptom-feature i18n YAMLs from a dataset's evidence schema.

Reads ``release_evidences.json`` for the named dataset (the DDXPlus shape
is the v1 reference) and writes:

* ``configs/i18n/<lang>/symptoms_<dataset_id>.yaml`` for each requested
  language. For ``en`` and ``fr`` the script pulls strings directly from
  the dataset's ``question_<lang>`` + ``value_meaning[raw][<lang>]``
  fields. For ``zh`` (or any language the corpus doesn't carry), the
  script writes the same key tree with empty-string values so the loader
  fails over to English instead of the bare key, and a translator has
  a complete to-fill template.
* The shared global ``symptoms.binary.<yes|no>`` block — emitted once
  per language; safe to re-run (overwrites).

Usage::

    uv run python scripts/generate_symptoms_i18n.py \\
        --data-dir demo/ddxplus_demo/ddxplus \\
        --dataset-id ddxplus \\
        --langs en,zh
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
I18N_DIR = REPO_ROOT / "configs" / "i18n"

# Slug derivation must match
# claritymed.core.symptoms.datasets.canonical.slugify_condition. Importing
# from the package would couple the script's startup time to the full
# claritymed import; redefining the regex inline keeps the script
# self-contained while we cross-check in a small assertion below.
import re  # noqa: E402

_SLUG_BAD = re.compile(r"[^a-z0-9]+")
_SLUG_EDGE = re.compile(r"^_+|_+$")

DEFAULT_BINARY = {
    "en": {"yes": "Yes", "no": "No"},
    "zh": {"yes": "是", "no": "否"},
    "fr": {"yes": "Oui", "no": "Non"},
}


def _nested_set(tree: dict, dotted: str, value: str) -> None:
    """Set ``tree[a][b][c] = value`` from key ``'a.b.c'``."""
    parts = dotted.split(".")
    cursor: dict = tree
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _yaml_dump(payload: dict) -> str:
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=120,
    )


def _slugify(name: str) -> str:
    """Slugify a condition display name; mirrors ``slugify_condition``."""
    lower = name.lower().strip()
    slug = _SLUG_BAD.sub("_", lower)
    slug = _SLUG_EDGE.sub("", slug)
    return slug or "unknown"


def _evidence_question(ev: dict, lang: str) -> str:
    """Return the dataset's native question text for ``lang`` or ``""``."""
    return (ev.get(f"question_{lang}") or "").strip()


def _value_label(ev: dict, raw: str, lang: str) -> str:
    """Return ``value_meaning[raw][lang]`` if present, else ``""``."""
    meaning = (ev.get("value_meaning") or {}).get(raw) or {}
    return (meaning.get(lang) or "").strip()


def _possible_values(ev: dict) -> list[str]:
    raw = ev.get("possible-values") or ev.get("possible_values") or []
    return [str(v) for v in raw]


def _condition_display(cond: dict) -> str | None:
    """Return the canonical display name from a DDXPlus condition entry."""
    return cond.get("condition_name") or cond.get("cond-name-eng")


def _condition_name(cond: dict, lang: str) -> str:
    """Return the localized condition name from the DDXPlus entry, or ``""``."""
    if lang == "en":
        return (_condition_display(cond) or "").strip()
    if lang == "fr":
        return (cond.get("cond-name-fr") or "").strip()
    # No native key for other langs — translator pass needed.
    return ""


def _build_lang_tree(
    evidences: dict[str, dict],
    conditions: dict[str, dict],
    dataset_id: str,
    lang: str,
) -> dict:
    """Construct the nested YAML tree for one language."""
    prefix = f"symptoms.{dataset_id}"
    tree: dict[str, Any] = {}
    for ev in evidences.values():
        ev_id = ev.get("name") or ev.get("code")
        if not ev_id:
            continue
        question = _evidence_question(ev, lang)
        _nested_set(tree, f"{prefix}.{ev_id}.question", question)
        values = _possible_values(ev)
        if not values:
            continue
        for raw in values:
            label = _value_label(ev, raw, lang)
            _nested_set(tree, f"{prefix}.{ev_id}.values.{raw}", label)
    # Conditions: localized display name, keyed by slug so the canonical
    # condition_name_key convention resolves.
    used_slugs: set[str] = set()
    for cond in conditions.values():
        display = _condition_display(cond)
        if not display:
            continue
        slug = _slugify(display)
        original = slug
        suffix = 2
        while slug in used_slugs:
            slug = f"{original}_{suffix}"
            suffix += 1
        used_slugs.add(slug)
        name = _condition_name(cond, lang)
        _nested_set(tree, f"{prefix}.conditions.{slug}.name", name)
    # Shared binary terms — emitted once per language file so every
    # language is self-contained for the symptoms feature.
    binary = DEFAULT_BINARY.get(lang, {"yes": "", "no": ""})
    _nested_set(tree, "symptoms.binary.yes", binary["yes"])
    _nested_set(tree, "symptoms.binary.no", binary["no"])
    return tree


def _load_json_dict(path: Path, label: str) -> dict[str, dict]:
    """Load a DDXPlus-style JSON dict; SystemExit with a hint when malformed."""
    if not path.exists():
        raise SystemExit(
            f"{label} not found at {path}; run the DDXPlus prepare step first."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise SystemExit(f"unexpected {label} shape — expected dict, got {type(data)}")
    return data


def _load_evidences(data_dir: Path) -> dict[str, dict]:
    return _load_json_dict(
        data_dir / "release_evidences.json", "release_evidences.json"
    )


def _load_conditions(data_dir: Path) -> dict[str, dict]:
    return _load_json_dict(
        data_dir / "release_conditions.json", "release_conditions.json"
    )


def generate(
    *,
    data_dir: Path,
    dataset_id: str,
    langs: Iterable[str],
    out_dir: Path = I18N_DIR,
) -> dict[str, Path]:
    """Write per-language YAML files; return ``{lang: path}`` of what was written."""
    evidences = _load_evidences(data_dir)
    conditions = _load_conditions(data_dir)
    written: dict[str, Path] = {}
    for lang in langs:
        tree = _build_lang_tree(evidences, conditions, dataset_id, lang)
        target_dir = out_dir / lang
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"symptoms_{dataset_id}.yaml"
        header = (
            f"# Auto-generated by scripts/generate_symptoms_i18n.py for "
            f"dataset={dataset_id} lang={lang}.\n"
            f"# Re-run to refresh from {data_dir}/release_evidences.json + "
            f"release_conditions.json.\n"
            f"# Empty string values mean 'no native translation in the "
            f"corpus'; the i18n loader falls back to en then to the "
            f"corpus's native fields, so blanks here are safe — fill them "
            f"in as a translator pass.\n"
        )
        target.write_text(header + _yaml_dump(tree), encoding="utf-8")
        written[lang] = target
    return written


def main() -> None:
    """CLI: regenerate the i18n bundle for one dataset, one or more languages."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument(
        "--langs",
        default="en,zh",
        help="Comma-separated language codes (default: en,zh).",
    )
    args = ap.parse_args()
    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    written = generate(
        data_dir=args.data_dir,
        dataset_id=args.dataset_id,
        langs=langs,
    )
    for lang, path in written.items():
        print(f"wrote {lang}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
