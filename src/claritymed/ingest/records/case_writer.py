"""Apply one ``CaseEntry`` to disk: blob bytes + manifest.yaml.

Slug formula (R10, deepening-revised — see plan §"Slug entropy source"):

    slug = f"{event_date.isoformat()}-tmpl-{sha256(case_id)[:10]}"

40 bits from the *full* case_id (not a prefix) avoid the
`notion-0db3d4f9-page` vs `notion-0db3d4f9-extract` prefix collision
the original ``[:8]`` formula was susceptible to.

The three dedup layers compose here without overlap:

1. **Record-level:** ``ManifestStore.create`` raises ``FileExistsError``
   on slug collision — works even if ``rows.jsonl`` is gone.
2. **Attachment-level:** ``BlobStore.store`` is content-addressed —
   re-feeding the same bytes is free.
3. **Chunk-level (third layer):** Unit 2's ``UserPhiRagStore.add_record``
   handles it — not this module's concern.

OCR provenance: if the skill already ran ``ocr-extract`` for an
attachment, ``BlobStore.ocr_done(sha)`` is True; we write
``ocr_status="done"`` on the ``Attachment`` so the lazy ``OcrWorker``
skips it. Otherwise we write ``ocr_status="pending"`` and the worker
picks it up post-import. Each attachment is OCR'd at most once across
the whole pipeline — see Key Decision §OCR pipeline reuse.

``embed_status="pending_retry"`` is the documented "needs embedding"
sentinel the reconcile worker scans for. Writing ``"ok"`` would orphan
every imported record from the chat agent's retrieval path. See plan
§Case writer writes ``embed_status="pending_retry"``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from claritymed.core.schemas.records import Attachment
from claritymed.ingest.records.template_schema import CaseAttachment, CaseEntry
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.manifest_store import ManifestStore

SLUG_HASH_LEN = 10  # 40 bits — see module docstring.

ApplyStatus = Literal["done", "skipped", "error"]


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of one ``apply_case`` call.

    The orchestrator (Unit 8) translates this into a ``RowRecord`` plus
    the WAL distinction between ``done(recovered)`` and
    ``skipped(already_imported)`` — this module stays state-machine-
    naive on purpose so it can be unit-tested without the WAL fixture.
    """

    status: ApplyStatus
    slug: str
    error_detail: str | None = None


def compute_slug(case: CaseEntry) -> str:
    """Deterministic slug: ``<event_date>-tmpl-<sha10>``.

    The hash domain is the full ``case_id`` string. Two skill outputs
    that share a long prefix (``notion-0db3d4f9-page`` and
    ``notion-0db3d4f9-extract``) produce distinct slugs because the
    hash distinguishes them, not the first N characters.
    """
    digest = hashlib.sha256(case.case_id.encode("utf-8")).hexdigest()[:SLUG_HASH_LEN]
    return f"{case.event_date.isoformat()}-tmpl-{digest}"


def apply_case(user_id: str, case: CaseEntry) -> ApplyResult:
    """Materialize one case on disk. Idempotent under re-run.

    Failure modes:
    * ``FileExistsError`` (manifest already at this slug) →
      ``ApplyResult("skipped", slug, "already_imported")``. The
      orchestrator distinguishes "crash-recovered done" from "earlier
      run already wrote" by checking ``rows.jsonl`` for a ``pending``
      line on this row_id.
    * Any other exception (missing attachment bytes, store I/O error,
      pydantic validation) → ``ApplyResult("error", slug, repr(exc))``.
      The orchestrator decides whether to keep the template alive for
      resume; this function never raises.
    """
    if case.category is None:
        # Loader (Unit 4) guarantees backfill from _meta.default_category;
        # reaching this branch is a programming error in the orchestrator.
        raise ValueError(
            f"apply_case received CaseEntry with no category (loader bug?): "
            f"case_id={case.case_id!r}"
        )

    slug = compute_slug(case)

    try:
        attachments = _materialize_attachments(user_id, case.attachments)
    except FileNotFoundError as exc:
        return ApplyResult("error", slug, repr(exc))
    except Exception as exc:
        return ApplyResult("error", slug, repr(exc))

    payload = {
        "kind": case.kind,
        "category": case.category,
        "title": case.title,
        "date": case.event_date.isoformat(),
        "tags": list(case.tags),
        "body": case.body_md or None,
        "attachments": [a.model_dump() for a in attachments],
        "_embed_status": "pending_retry",
    }

    try:
        ManifestStore(user_id, scope="records").create(case.category, slug, payload)
    except FileExistsError:
        return ApplyResult("skipped", slug, "already_imported")
    except Exception as exc:
        return ApplyResult("error", slug, repr(exc))

    return ApplyResult("done", slug, None)


def _materialize_attachments(
    user_id: str,
    case_attachments: list[CaseAttachment],
) -> list[Attachment]:
    """Store each attachment in the user's BlobStore; build the manifest list.

    Reading attachment bytes lazily here (vs at template-build time)
    keeps the loader cheap on large templates and means a missing file
    between Unit 4 load and Unit 6 apply produces a per-case ``error``,
    not a fatal import abort.
    """
    blob = BlobStore(user_id)
    out: list[Attachment] = []
    for att in case_attachments:
        path = Path(att.path)
        content = path.read_bytes()
        ext = _safe_ext(att.original_filename)
        sha = blob.store(content, ext)
        ocr_status = "done" if blob.ocr_done(sha) else "pending"
        out.append(
            Attachment(
                sha256=sha,
                filename=att.original_filename,
                mime=att.mime,
                size=len(content),
                ocr_status=ocr_status,
            )
        )
    return out


def _safe_ext(filename: str) -> str:
    """Lowercase, validated extension (no dot). Falls back to ``bin``.

    BlobStore validates ``ext`` against ``[a-z0-9]{1,8}``; anything
    longer or with punctuation gets bucketed to ``bin`` so the store
    doesn't reject the write on a quirky filename.
    """
    suffix = Path(filename).suffix.lstrip(".").lower()
    if suffix and 1 <= len(suffix) <= 8 and suffix.isalnum():
        return suffix
    return "bin"
