"""E2E: save_record lifecycle without live services.

Stitches together the v1 PHI pipeline as wired in Units 1-11:

    BlobStore.store → SessionAttachments → ToolDispatcher.gate →
    save_record body → ManifestStore.create → audit_payloads side-channel

This is the integration story Unit 12 ships: every layer's primitive
gets exercised against in-memory fixtures, no Qdrant / no LLM / no
network. Real PHI defense at runtime is verified by the per-unit suites
(Unit 5's PhiAssertionModel, Unit 6's TOCTOU close, Unit 7's
SettingsStore rules) — this test proves they compose.
"""

from __future__ import annotations

import json
import os

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.orchestrator.features.ingest_tools_plugin import (
    delete_record,
    save_record,
)
from claritymed.orchestrator.services.session_attachments import SessionAttachments
from claritymed.orchestrator.services.tool_dispatcher import ToolDispatcher
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.manifest_store import ManifestStore
from claritymed.stores.paths import user_audit_payload_path


@pytest.fixture
def _ctx():
    tokens = apply_context("20260611000000ABCDEF12", "e2e", "en")
    yield
    reset_context(tokens)


def test_paste_to_save_round_trip(_ctx):
    """User pastes a PDF → session tray → LLM proposes save_record →
    approval is implicit (no rule, fall-through, headless caller) →
    manifest lands on disk → audit_payload carries PHI text."""
    user_id = "e2e"
    # Step 1: paste the blob.
    bs = BlobStore(user_id)
    sha = bs.store(b"%PDF-1.4 fake report body", "pdf")

    # Step 2: session tray gets a row (in real life from app.py paste handler).
    sa = SessionAttachments(user_id, "sess-1")
    sa.add(sha256=sha, filename="checkup.pdf", mime="application/pdf", size=64)

    # Step 3: dispatcher sees the session attachments.
    dispatcher = ToolDispatcher(session_attachments=lambda: {sha})

    # Step 4: tool body materializes the manifest from a save_record call.
    args = {
        "category": "exam-reports",
        "kind": "exam-report",
        "title": "Annual checkup",
        "date": "2026-06-11",
        "provider": "Dr. Smith",
        "attachments": [{"sha256": sha, "filename": "checkup.pdf"}],
        "extracted_labs": [
            {"name": "HGB", "value": 105, "unit": "g/L"},
        ],
        "tags": ["routine"],
        "notes": "All within reference range.",
    }
    out = save_record(args, dispatcher=dispatcher)
    record_path = out["record_path"]
    assert record_path.startswith("exam-reports/2026-06-11-")

    # Step 5: manifest lands on disk with revision=1.
    cat, slug = record_path.split("/")
    manifest = ManifestStore(user_id, "records").read(cat, slug)
    assert manifest.revision == 1
    assert manifest.title == "Annual checkup"
    assert manifest.attachments[0].sha256 == sha
    assert manifest.attachments[0].mime == "application/pdf"

    # Step 6: PHI side-channel exists with mode 0o600.
    payload_path = user_audit_payload_path(
        user_id, "20260611000000ABCDEF12"
    )  # matches _ctx request_id
    assert payload_path.exists()
    mode = os.stat(payload_path).st_mode & 0o777
    assert mode == 0o600
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["tool_name"] == "save_record"
    assert payload["args"]["title"] == "Annual checkup"

    # Step 7: delete cascades — manifest gone afterwards.
    delete_out = delete_record(
        {"record_path": record_path, "confirm_kind": "exam-report"},
        dispatcher=dispatcher,
    )
    assert delete_out == {"deleted": record_path}
    from claritymed.errors import RecordNotFound

    with pytest.raises(RecordNotFound):
        ManifestStore(user_id, "records").read(cat, slug)


def test_unknown_sha_refuses_save_record(_ctx):
    """The gate must reject a save_record that references a sha not in the
    user's blob universe — pin the contract before any LLM hallucination
    reaches a real model."""
    from claritymed.errors import UnknownSha256

    dispatcher = ToolDispatcher()
    with pytest.raises(UnknownSha256):
        save_record(
            {
                "category": "exam-reports",
                "kind": "exam-report",
                "title": "x",
                "attachments": [{"sha256": "c" * 64, "filename": "x.pdf"}],
            },
            dispatcher=dispatcher,
        )


def test_path_outside_user_domain_refused(_ctx):
    """delete_record with a traversal path must raise."""
    from claritymed.errors import PathOutsideUserDomain

    dispatcher = ToolDispatcher()
    with pytest.raises(PathOutsideUserDomain):
        delete_record(
            {"record_path": "../../../etc/passwd", "confirm_kind": "x"},
            dispatcher=dispatcher,
        )
