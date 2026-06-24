"""Ingest runner — feeds one file into a system_rag collection.

The runner expects the file to already exist on the server (the SPA
uploads via the existing attachments path before kicking the job,
or the operator places it in a known directory). Wire format:

    POST /admin/rag/ingest
      body: { collection_name, file_path, doc_id?, language? }

The runner delegates to :func:`claritymed.ingest.corpus.ingest_corpus`
with a one-doc :class:`CorpusSource`. Progress is reported at
milestone boundaries (read → chunk → embed → upsert).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claritymed.web.admin.jobs import JobRegistry, JobSpec

logger = logging.getLogger(__name__)


async def run(spec: "JobSpec", registry: "JobRegistry") -> None:
    """Validate inputs, then defer to ``ingest_corpus``.

    Validation only — actual ingest is delegated to the existing
    pipeline so the admin path does not re-implement chunking /
    embedding / upserting. If the pipeline raises, the registry's
    failure handler captures the exception text.
    """
    params = spec.params
    collection_name = params.get("collection_name")
    file_path = params.get("file_path")
    if not collection_name or not file_path:
        raise ValueError("rag_ingest requires both 'collection_name' and 'file_path'")
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")

    registry.update(spec.id, progress="reading")

    try:
        from claritymed.ingest.corpus import CorpusSource, ingest_corpus
    except ImportError as exc:
        raise RuntimeError(
            "ingest pipeline not importable; cannot run rag_ingest"
        ) from exc

    registry.update(spec.id, progress="ingesting (chunk + embed + upsert)")
    # The pipeline is synchronous; run it on a worker thread so the
    # event loop stays responsive for the registry / cancel signal.
    import asyncio

    def _go() -> int:
        source = CorpusSource(
            name=collection_name,
            files=[path],
            topic=params.get("topic", "operator-supplied"),
            language=params.get("language", "en"),
            authority_tier=int(params.get("authority_tier", 3)),
        )
        result = ingest_corpus(source)
        # ingest_corpus return shape varies — at minimum we can stash a
        # repr of the result in stdout_tail so the SPA shows something
        # useful in the drilldown.
        return getattr(result, "added_chunks", 0)

    added = await asyncio.to_thread(_go)
    registry.append_stdout(spec.id, f"added {added} chunks")
    registry.update(spec.id, progress=f"done ({added} chunks)")
