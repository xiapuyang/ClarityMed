"""Curated per-condition medical content for the symptoms card renderer.

Backing store is ``configs/i18n/<lang>/symptoms_conditions.yaml`` — one
file per language, keyed by the canonical condition slug (matching
``wire.DifferentialRow.condition_id``). The i18n loader also globs
these files into its flat dict, but that path is not authoritative —
:class:`SymptomsConditionsCatalog` reads the YAML directly and returns
structured :class:`ConditionEntry` models so the plugin can hydrate
:class:`DifferentialCard` payloads without ``t()`` gymnastics.

Startup validator ``validate_conditions_catalog`` asserts full coverage
of every ``condition_id`` in every enabled ``DatasetRegistry`` entry
across both languages. A miss raises
:class:`~claritymed.errors.SymptomsCatalogValidationError` at plugin
construct, matching the fail-loud posture of
``_validate_symptoms_prompts``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Iterable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from claritymed.config import I18N_DIR
from claritymed.errors import SymptomsCatalogValidationError

if TYPE_CHECKING:
    from claritymed.core.symptoms.schemas import DatasetSpec

logger = logging.getLogger(__name__)

_FILENAME = "symptoms_conditions.yaml"

Language = Literal["en", "zh"]
SUPPORTED_LANGUAGES: tuple[Language, ...] = ("en", "zh")


class ConditionEntry(BaseModel):
    """Per-condition curated content the card renderer consumes.

    ``suggestion`` is not rendered on the card itself — the card carries
    only ``display_name`` + ``report``. Suggestions from top-ranked
    conditions are aggregated into the summary block below the cards
    (see ``symptoms_card_builder.build_differential_summary``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    display_name: str = Field(min_length=1, max_length=128)
    report: str = Field(min_length=1)
    suggestion: str = Field(min_length=1)


class SymptomsConditionsCatalog:
    """Mtime-cached loader for the per-language conditions catalog.

    Load semantics mirror the i18n loader: entries are cached on
    ``(lang, mtime)`` so admins can edit YAML without restarting the
    orchestrator. Missing files are surfaced as an empty language dict
    — the validator turns that into a fail-loud error at plugin
    construct so a first-run empty file cannot silently ship without
    catalog coverage.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else Path(I18N_DIR)
        self._by_lang: dict[str, dict[str, ConditionEntry]] = {}
        self._mtime: dict[str, float] = {}
        self._lock = Lock()

    def _path(self, lang: Language) -> Path:
        return self._base_dir / lang / _FILENAME

    def _current_mtime(self, path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return -1.0

    def _load_language(self, lang: Language) -> dict[str, ConditionEntry]:
        path = self._path(lang)
        current = self._current_mtime(path)
        with self._lock:
            cached_mtime = self._mtime.get(lang)
            if cached_mtime == current and lang in self._by_lang:
                return self._by_lang[lang]
            entries = self._parse_file(path, lang)
            self._by_lang[lang] = entries
            self._mtime[lang] = current
            return entries

    def _parse_file(self, path: Path, lang: Language) -> dict[str, ConditionEntry]:
        if not path.exists():
            logger.warning(
                "symptoms conditions catalog missing for lang=%s at %s", lang, path
            )
            return {}
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        conditions = raw.get("conditions") if isinstance(raw, dict) else None
        if not isinstance(conditions, dict):
            raise SymptomsCatalogValidationError(
                f"{path}: expected a top-level 'conditions' mapping, "
                f"got {type(conditions).__name__}"
            )
        entries: dict[str, ConditionEntry] = {}
        for cid, payload in conditions.items():
            try:
                entries[str(cid)] = ConditionEntry.model_validate(payload)
            except ValidationError as exc:
                raise SymptomsCatalogValidationError(
                    f"{path}: conditions.{cid} failed validation: {exc}"
                ) from exc
        return entries

    def get(self, condition_id: str, language: Language) -> ConditionEntry | None:
        """Return the curated entry or ``None`` if uncovered.

        Callers that require coverage (i.e. after a successful
        :func:`validate_conditions_catalog` at construct) can assume
        this returns a non-None value for every ``condition_id`` the
        dataset registry knows about.
        """
        return self._load_language(language).get(condition_id)

    def language_entries(self, language: Language) -> dict[str, ConditionEntry]:
        """Return every entry for ``language`` (mtime-cached view)."""
        return dict(self._load_language(language))


def enumerate_dataset_condition_ids(
    dataset: "DatasetSpec",
    i18n_dir: Path | None = None,
) -> set[str]:
    """Read the dataset's own i18n YAML files to enumerate its condition slugs.

    The plugin doesn't hold a canonical dataset in memory (that lives on
    the symptoms server), so the runtime authoritative list of
    ``condition_id`` slugs is the operator-owned i18n file — e.g.
    ``configs/i18n/en/symptoms_ddxplus.yaml`` under
    ``symptoms.ddxplus.conditions.<slug>.name``. This helper walks the
    known i18n files, uses :meth:`DatasetSpec.resolved_i18n_prefix` to
    locate the ``conditions:`` sub-tree, and returns the union of slugs
    across both languages so a slug present in only one language still
    counts (bilingual asymmetry surfaces as a catalog gap in the other
    language during ``validate_conditions_catalog``).
    """
    base = Path(i18n_dir) if i18n_dir is not None else Path(I18N_DIR)
    ids: set[str] = set()
    prefix_parts = dataset.resolved_i18n_prefix().split(".")
    for lang in SUPPORTED_LANGUAGES:
        domain_dir = base / lang
        if not domain_dir.is_dir():
            continue
        for path in sorted(domain_dir.glob("*.yaml")):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    raw = yaml.safe_load(fh) or {}
            except (OSError, yaml.YAMLError) as exc:
                logger.warning(
                    "conditions_catalog: could not parse %s while enumerating "
                    "dataset condition ids: %s",
                    path,
                    exc,
                )
                continue
            node: Any = raw
            for part in prefix_parts:
                if not isinstance(node, dict):
                    node = None
                    break
                node = node.get(part)
            if isinstance(node, dict):
                conds = node.get("conditions")
                if isinstance(conds, dict):
                    ids.update(str(k) for k in conds.keys())
    return ids


def validate_conditions_catalog(
    catalog: SymptomsConditionsCatalog,
    condition_ids: Iterable[str],
) -> None:
    """Assert every ``condition_id`` resolves in both ``en`` and ``zh``.

    Called from ``SymptomsFeature.__init__`` after
    ``_validate_symptoms_prompts`` and ``_validate_safety_keywords``.
    Fails loud with a bilingual missing-key report rather than
    per-language so operators can fix in one pass.
    """
    ids = list(condition_ids)
    if not ids:
        return
    missing: list[str] = []
    for lang in SUPPORTED_LANGUAGES:
        entries = catalog.language_entries(lang)
        for cid in ids:
            if cid not in entries:
                missing.append(f"{lang}:{cid}")
    if missing:
        raise SymptomsCatalogValidationError(
            "Missing symptoms conditions catalog entries — every "
            "dataset condition_id must resolve in both en and zh. "
            "Missing: " + ", ".join(sorted(missing))
        )
