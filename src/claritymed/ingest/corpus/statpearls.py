"""StatPearls source adapter.

StatPearls is published as PMC JATS XML through NCBI Bookshelf. Two input
shapes are supported so the runner does not have to special-case
downloads:

* **NXML / JATS XML** — one article per ``.nxml`` file. The adapter pulls
  ``article-title`` + concatenated paragraph text from the body and emits
  one ``RawDocument`` per file.
* **JSONL** — one record per line. Each line must be a JSON object with
  ``doc_id`` + ``title`` + ``text`` keys. This is the normalized form
  produced by ``--dry-run`` so a follow-up run can skip XML parsing.

Adapter is sync (file IO only); the async embed + Qdrant write happens
in ``ingest_corpus``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

from claritymed.core.rag.chunking.base import RawDocument

logger = logging.getLogger(__name__)

COLLECTION_NAME = "statpearls_en"
LANGUAGE = "en"
SOURCE_URI_PREFIX = "https://www.ncbi.nlm.nih.gov/books/"


def _nbk_uri(doc_id: str) -> str | None:
    """Return a valid NCBI Bookshelf URL only for NBK-prefixed IDs.

    MedRAG StatPearls data uses ``article-XXXXX`` internal IDs that do not
    map to public NCBI URLs. Only ``NBK``-prefixed IDs have real bookshelf
    pages.
    """
    if doc_id.upper().startswith("NBK"):
        return f"{SOURCE_URI_PREFIX}{doc_id}/"
    return None


class StatPearlsSource:
    """Iterate StatPearls articles from a local directory."""

    name: str = COLLECTION_NAME

    def __init__(self, root: Path) -> None:
        if not root.exists():
            raise FileNotFoundError(
                f"StatPearls root {root} does not exist — download to "
                f"data/knowledge/raw/statpearls/ first (README has the URL)"
            )
        self._root = root

    def iter_raw_docs(self) -> Iterator[RawDocument]:
        yield from self._iter_jsonl()
        yield from self._iter_nxml()

    # --- JSONL path (normalized output) -------------------------------

    def _iter_jsonl(self) -> Iterator[RawDocument]:
        for path in sorted(self._root.rglob("*.jsonl")):
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
                    doc = self._doc_from_json(obj)
                    if doc:
                        yield doc

    @staticmethod
    def _doc_from_json(obj: dict) -> RawDocument | None:
        doc_id = obj.get("doc_id")
        text = obj.get("text")
        if not doc_id or not text:
            return None
        doc_id = str(doc_id)
        title = obj.get("title", "")
        merged_text = f"{title}\n\n{text}" if title else text
        return RawDocument(
            doc_id=doc_id,
            text=merged_text,
            language=LANGUAGE,
            metadata={
                "doc_title": title,
                "source_uri": obj.get("source_uri") or _nbk_uri(doc_id),
                "collection": COLLECTION_NAME,
            },
        )

    # --- NXML / JATS path ---------------------------------------------

    def _iter_nxml(self) -> Iterator[RawDocument]:
        for path in sorted(self._root.rglob("*.nxml")):
            try:
                doc = self._doc_from_nxml(path)
            except ET.ParseError:
                logger.warning("%s skipped (bad XML)", path.name)
                continue
            if doc:
                yield doc

    @staticmethod
    def _doc_from_nxml(path: Path) -> RawDocument | None:
        tree = ET.parse(path)
        root = tree.getroot()
        title = (root.findtext(".//article-title") or "").strip()
        # Concatenate every <p> in <body>; preserve paragraph breaks.
        body_paragraphs = [
            (p.text or "").strip()
            for p in root.findall(".//body//p")
            if (p.text or "").strip()
        ]
        if not body_paragraphs:
            return None
        text = ("\n\n".join(body_paragraphs)).strip()
        if not text:
            return None
        doc_id = path.stem
        merged_text = f"{title}\n\n{text}" if title else text
        return RawDocument(
            doc_id=doc_id,
            text=merged_text,
            language=LANGUAGE,
            metadata={
                "doc_title": title,
                "source_uri": _nbk_uri(doc_id),
                "collection": COLLECTION_NAME,
            },
        )
