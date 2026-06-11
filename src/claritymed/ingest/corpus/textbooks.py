"""MedRAG medical textbooks source adapter.

Reads the MedRAG textbook corpus distributed as one JSONL per book under a
``chunk/`` subdirectory. Each line is a JSON object with:

    id       — unique chunk ID, e.g. ``First_Aid_Step1_0``
    title    — book title, e.g. ``First_Aid_Step1``
    content  — chunk text
    contents — redundant (title + content); ignored here

The adapter yields one ``RawDocument`` per JSONL line. The Qdrant collection
is ``textbooks_en`` and matches the entry in ``configs/retrieval.yaml``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from claritymed.core.rag.chunking.base import RawDocument

logger = logging.getLogger(__name__)

# FROZEN: this string is the Qdrant collection name on disk. Renaming it
# requires a data migration — do not change it without migrating the index.
COLLECTION_NAME = "textbooks_en"
LANGUAGE = "en"


class TextbooksSource:
    """Iterate MedRAG medical textbook chunks from a local directory."""

    name: str = COLLECTION_NAME

    def __init__(self, root: Path) -> None:
        if not root.exists():
            raise FileNotFoundError(
                f"Textbooks root {root} does not exist — symlink "
                f"~/.claritymed/shared/knowledge/raw/textbooks to the "
                f"downloaded textbooks directory first."
            )
        self._root = root

    def iter_raw_docs(self) -> Iterator[RawDocument]:
        # MedRAG layout: root/chunk/*.jsonl — also handles flattened root.
        jsonl_files = sorted(self._root.rglob("*.jsonl"))
        if not jsonl_files:
            logger.warning("No .jsonl files found under %s", self._root)
            return
        for path in jsonl_files:
            yield from self._iter_jsonl(path)

    def _iter_jsonl(self, path: Path) -> Iterator[RawDocument]:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("%s:%d skipped (bad JSON)", path.name, line_no)
                    continue
                doc = _doc_from_json(obj)
                if doc:
                    yield doc
                else:
                    logger.warning(
                        "%s:%d skipped (missing id/content)", path.name, line_no
                    )


def _doc_from_json(obj: dict) -> RawDocument | None:
    doc_id = obj.get("id")
    text = obj.get("content")
    if not doc_id or not text:
        return None
    title = obj.get("title", "")
    merged_text = f"{title}\n\n{text}" if title else text
    return RawDocument(
        doc_id=str(doc_id),
        text=merged_text,
        language=LANGUAGE,
        metadata={
            "doc_title": title,
            "source_uri": None,
            "collection": COLLECTION_NAME,
        },
    )
