"""Local UMLS + CMeKG terminology lookup.

Loads a pre-normalized JSONL file (operator prepares this from UMLS /
CMeKG offline — we never redistribute UMLS) into memory at construction
time. v1 expectation: ~50-200k concepts × a few aliases each, well under
1GB; in-memory dict lookup is O(1) per surface form.

For lookup we lowercase-strip the surface form. CJK characters are
unchanged by ``.lower()`` so the same code path handles Chinese.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from claritymed.core.rag.terms.base import (
    Alias,
    ConceptHit,
    ConceptLanguage,
    ConceptRecord,
    TermService,
)

logger = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return s.strip().lower()


class UmlsCmekgLocalService(TermService):
    """In-memory term service backed by ``concepts.jsonl``."""

    def __init__(self, jsonl_path: Path) -> None:
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"UmlsCmekgLocalService needs {jsonl_path} — prepare a UMLS+CMeKG "
                f"export per docs (see core/rag/terms/__init__.py module docstring)"
            )
        self._records: dict[str, ConceptRecord] = {}
        # surface (lowered) + language → list of concept_ids
        self._index: dict[tuple[str, str], list[str]] = {}
        self._load(jsonl_path)

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "terminology line %d skipped (bad JSON): %s", line_no, exc
                    )
                    continue
                record = self._build_record(obj)
                if record is None:
                    continue
                self._records[record.concept_id] = record
                for alias in record.aliases:
                    key = (_norm(alias.text), alias.language)
                    bucket = self._index.setdefault(key, [])
                    if record.concept_id not in bucket:
                        bucket.append(record.concept_id)

    @staticmethod
    def _build_record(obj: dict) -> ConceptRecord | None:
        try:
            aliases = tuple(
                Alias(text=a["text"], language=a["language"], source=a["source"])
                for a in obj["aliases"]
            )
            return ConceptRecord(
                concept_id=obj["concept_id"],
                type=obj["type"],
                aliases=aliases,
            )
        except (KeyError, TypeError) as exc:
            logger.warning("terminology record skipped (bad shape): %s", exc)
            return None

    # --- TermService -----------------------------------------------------

    def lookup(self, surface: str, language: ConceptLanguage) -> list[ConceptHit]:
        if not surface or not surface.strip():
            return []
        key = (_norm(surface), language)
        concept_ids = self._index.get(key, [])
        hits: list[ConceptHit] = []
        for cid in concept_ids:
            record = self._records[cid]
            hits.append(
                ConceptHit(
                    concept_id=record.concept_id,
                    surface=surface,
                    language=language,
                    score=1.0,
                    type=record.type,
                    aliases=record.aliases,
                )
            )
        return hits

    def cross_lingual_aliases(self, concept_id: str) -> list[Alias]:
        record = self._records.get(concept_id)
        if record is None:
            return []
        return list(record.aliases)


class NoOpTermService(TermService):
    """Identity passthrough — used when no terminology export is configured."""

    def lookup(self, surface: str, language: ConceptLanguage) -> list[ConceptHit]:
        return []

    def cross_lingual_aliases(self, concept_id: str) -> list[Alias]:
        return []
