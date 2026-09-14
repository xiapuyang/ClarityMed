"""Ingest runner — feeds operator-uploaded files into a system_rag collection.

Wire format:

    POST /admin/rag/collections/upsert
      body: multipart (files=…) + JSON metadata
      job params:
        {
          "name": "ats_idsa_cap_en",
          "file_paths": ["/abs/path/to/upload1.pdf", "/abs/path/to/upload2.pdf"],
          "topics": ["pneumonia"],
          "language": "en",                # optional — inherited on append
          "cross_lingual": false,          # optional
          "authority_tier": 1,             # optional — inherited on append
          "license": "ATS/IDSA",           # optional
          "dedupe_cosine_threshold": 0.0,  # optional
        }

The runner delegates to ``ingest_system_rag`` so the chunk + embed +
dedup + upsert + centroid-refresh path matches the
``scripts/init_system_rag.py`` CLI exactly — one implementation, two
entry points.

After a successful ingest of a *new* collection, the runner auto-appends
the rendered YAML snippet to ``configs/retrieval.yaml``. This is the
opposite of the CLI script's print-and-paste posture: the SPA caller's
review trail is the Jobs UI, not the git diff. Uploaded files are
cleaned up after the job terminates regardless of outcome.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from claritymed.ingest.system_rag import (
    SystemRagIngestRequest,
    ingest_system_rag,
)
from claritymed.web.admin.retrieval_yaml import append_system_rag_collection

if TYPE_CHECKING:
    from claritymed.web.admin.jobs import JobRegistry, JobSpec

logger = logging.getLogger(__name__)


def _request_from_params(params: dict) -> SystemRagIngestRequest:
    """Translate the JSON job params into a ``SystemRagIngestRequest``.

    Raises ``ValueError`` for missing required fields so the registry's
    failure handler captures a clear message in ``error``.
    """
    name = params.get("name")
    file_paths = params.get("file_paths") or []
    if not name:
        raise ValueError("rag_ingest requires 'name'")
    if not file_paths:
        raise ValueError("rag_ingest requires at least one file in 'file_paths'")

    files = [Path(p) for p in file_paths]

    return SystemRagIngestRequest(
        name=name,
        files=files,
        topics=list(params.get("topics") or []),
        language=params.get("language"),
        cross_lingual=bool(params.get("cross_lingual", False)),
        authority_tier=params.get("authority_tier"),
        license=params.get("license"),
        dedupe_cosine_threshold=float(params.get("dedupe_cosine_threshold", 0.0)),
        dry_run=bool(params.get("dry_run", False)),
    )


def _cleanup_uploads(upload_dir: str | None) -> None:
    """Best-effort removal of the per-job uploads directory.

    Called from the runner's finally clause so a successful ingest, a
    failed ingest, and a cancelled run all leave a clean ``data/jobs/``.
    """
    if not upload_dir:
        return
    path = Path(upload_dir)
    try:
        if path.exists():
            shutil.rmtree(path)
    except OSError:
        logger.warning("rag_ingest: failed to clean up %s", path, exc_info=True)


async def run(spec: "JobSpec", registry: "JobRegistry") -> None:
    """Run the shared ingest pipeline; auto-append YAML for new collections."""
    upload_dir = spec.params.get("upload_dir")
    try:
        req = _request_from_params(spec.params)
        registry.update(
            spec.id, progress=f"ingest: {req.name} ({len(req.files)} files)"
        )

        def _on_progress(line: str) -> None:
            registry.append_stdout(spec.id, line)

        result = await ingest_system_rag(req, on_progress=_on_progress)
        stats = result.stats
        registry.append_stdout(
            spec.id,
            (
                f"summary: {stats.docs_processed} docs / "
                f"{stats.parents_written} parents / "
                f"{stats.children_written} children "
                f"(skipped {stats.docs_skipped}, "
                f"resumed {stats.docs_resumed}, "
                f"deduped {stats.children_deduped})"
            ),
        )
        if result.centroid_refreshed:
            registry.append_stdout(spec.id, f"centroid refreshed: {req.name}")

        # Auto-append the snippet only for genuinely new collections
        # AND only when the ingest actually wrote children — appending
        # an empty entry would expose a collection the router can score
        # against zero centroid mass.
        if result.is_new_collection and stats.children_written > 0:
            try:
                appended = append_system_rag_collection(
                    result.yaml_snippet, name=req.name
                )
                if appended:
                    registry.append_stdout(
                        spec.id,
                        f"retrieval.yaml: appended new collection {req.name!r}",
                    )
                else:
                    registry.append_stdout(
                        spec.id,
                        f"retrieval.yaml: {req.name!r} already present; not re-appended",
                    )
            except Exception as exc:  # noqa: BLE001 — surfacing in stdout is enough
                logger.exception("rag_ingest: yaml append failed")
                registry.append_stdout(
                    spec.id, f"retrieval.yaml: append failed: {exc!r}"
                )
        elif result.is_new_collection:
            registry.append_stdout(
                spec.id,
                "retrieval.yaml: new collection had no children written; "
                "skipping yaml append (re-run with real files to register).",
            )

        registry.update(
            spec.id,
            progress=(
                f"done ({stats.children_written} children, "
                f"{stats.children_deduped} deduped)"
            ),
        )
    finally:
        _cleanup_uploads(upload_dir)
