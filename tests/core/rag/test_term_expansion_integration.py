"""Integration tests for the production ``umls_cmekg_local`` path.

Unlike ``test_term_expansion.py`` (which constructs ``UmlsCmekgLocalService``
directly from a fixture path), these tests exercise the seam an operator
hits in production:

1. ``scripts/init_terminology.py --seed`` writes ``concepts.jsonl`` into
   ``SHARED_DIR/terminology/`` (the path resolved by
   ``shared_terminology_jsonl()``).
2. ``build_term_service()`` resolves the ``umls_cmekg_local`` catalog entry
   (no ``data_dir`` override) to the shared path, reads the file, and
   returns a working ``UmlsCmekgLocalService``.
3. ``expand_query`` produces the expected cross-lingual synonyms.

These tests construct an **explicit** ``TermServiceConfig(active=
"umls_cmekg_local")`` rather than reading ``configs/retrieval.yaml`` —
the YAML's ``active`` is operationally ``none`` (term service is shipped
off-by-default until operators provision a real export), but the
underlying code paths must keep working so a flip to ``umls_cmekg_local``
is a one-line change for them.

CI-safe: no live services required. The per-test ``_isolate_runtime``
fixture (see ``tests/conftest.py``) gives each test its own
``SHARED_DIR`` under ``tmp_path``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from claritymed.core.rag import load_retrieval_config
from claritymed.core.rag.schemas import TermServiceConfig, TermServiceEntry
from claritymed.core.rag.terms import (
    NoOpTermService,
    UmlsCmekgLocalService,
    expand_query,
)
from claritymed.core.rag.terms.factory import build_term_service
from claritymed.stores.paths import shared_terminology_jsonl

# Import the seed script's data + commands as a library — same path the
# e2e conftest uses, so we exercise the same code production operators run.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
try:
    from init_terminology import SEED_CONCEPTS, cmd_seed, cmd_validate  # type: ignore[import-not-found]
finally:
    sys.path.pop(0)


# --- helpers ------------------------------------------------------------


def _umls_local_config() -> TermServiceConfig:
    """Explicit `active=umls_cmekg_local` config (no `data_dir` override)."""
    return TermServiceConfig(
        active="umls_cmekg_local",
        catalog=[
            TermServiceEntry(id="umls_cmekg_local", kind="local"),
            TermServiceEntry(id="none", kind="noop"),
        ],
    )


# --- YAML structural invariants ----------------------------------------


def test_yaml_term_service_catalog_invariants():
    """Lock the architectural invariants without pinning the operational toggle.

    ``active`` is a runtime decision (operator may flip it across
    deploys, environments, or experiments) — pinning it here would
    make every legitimate flip a test failure. We assert only the
    structural facts that must hold regardless of which entry is
    active:

    1. Both catalog entries (``umls_cmekg_local`` + ``none``) are
       present, so flipping back to NoOp degradation is always one
       line away.
    2. ``active`` resolves to a catalog entry — pydantic enforces this
       at load time, but we assert it here as documentation.
    3. The ``umls_cmekg_local`` entry has no ``data_dir`` — the
       absence is load-bearing: it proves resolution still goes
       through ``shared_terminology_jsonl()`` (the test below covers
       the positive case).
    """
    cfg = load_retrieval_config()
    ts = cfg.term_service

    ids = {entry.id for entry in ts.catalog}
    assert ids == {"umls_cmekg_local", "none"}, (
        f"catalog drifted: expected {{'umls_cmekg_local', 'none'}}, got {ids}"
    )
    assert ts.active in ids

    umls_entry = next(e for e in ts.catalog if e.id == "umls_cmekg_local")
    assert umls_entry.data_dir is None, (
        "umls_cmekg_local catalog entry sprouted a data_dir — that breaks "
        "the shared/ resolution path tested in "
        "test_build_term_service_resolves_shared_dir_by_default"
    )


# --- production resolution path -----------------------------------------


def test_build_term_service_resolves_shared_dir_by_default():
    """No data_dir → factory falls back to shared/terminology/concepts.jsonl."""
    cmd_seed(shared_terminology_jsonl(), force=False, dry_run=False)
    svc = build_term_service(_umls_local_config())
    assert isinstance(svc, UmlsCmekgLocalService)


def test_missing_file_error_points_at_seed_script():
    """Operators reading the traceback should see the recovery command."""
    target = shared_terminology_jsonl()
    assert not target.exists(), "fresh tmp_path SHARED_DIR should be empty"
    with pytest.raises(FileNotFoundError) as exc:
        build_term_service(_umls_local_config())
    msg = str(exc.value)
    assert "scripts/init_terminology.py" in msg
    assert "--seed" in msg


def test_seeded_service_expands_known_concept():
    """End-to-end: seed → factory → expand_query produces synonyms."""
    cmd_seed(shared_terminology_jsonl(), force=False, dry_run=False)
    svc = build_term_service(_umls_local_config())
    out = expand_query("aspirin side effects", "en", svc)
    assert "acetylsalicylic acid" in out
    assert "阿司匹林" in out
    assert out.startswith("aspirin side effects")  # original preserved


def test_seeded_service_handles_chinese_query():
    """zh→en expansion is the cross-lingual recall payoff."""
    cmd_seed(shared_terminology_jsonl(), force=False, dry_run=False)
    svc = build_term_service(_umls_local_config())
    out = expand_query("血红蛋白 偏低", "zh", svc)
    # The "hemoglobin" concept in the seed pairs Chinese + English aliases;
    # cross-lingual expansion lets BGE-M3 match English-language sources.
    assert "hemoglobin" in out.lower()


# --- script round-trip --------------------------------------------------


def test_seed_writes_well_formed_jsonl():
    target = shared_terminology_jsonl()
    cmd_seed(target, force=False, dry_run=False)
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(SEED_CONCEPTS)
    for line in lines:
        obj = json.loads(line)
        assert obj["concept_id"]
        assert obj["type"] in {"drug", "disease", "symptom", "procedure", "other"}
        assert obj["aliases"]


def test_seed_refuses_to_overwrite_without_force():
    target = shared_terminology_jsonl()
    cmd_seed(target, force=False, dry_run=False)
    # Second call must fail loud — operators with a real export should not
    # lose it because they re-ran --seed by mistake.
    rc = cmd_seed(target, force=False, dry_run=False)
    assert rc == 1


def test_seed_force_overwrites():
    target = shared_terminology_jsonl()
    cmd_seed(target, force=False, dry_run=False)
    rc = cmd_seed(target, force=True, dry_run=False)
    assert rc == 0


def test_validate_reports_summary(capsys):
    target = shared_terminology_jsonl()
    cmd_seed(target, force=False, dry_run=False)
    rc = cmd_validate(target)
    assert rc == 0
    captured = capsys.readouterr().out
    assert f"concepts:       {len(SEED_CONCEPTS)}" in captured
    assert "by language:" in captured
    assert "by type:" in captured


# --- noop still works (back-compat) -------------------------------------


def test_noop_branch_does_not_require_jsonl():
    """Flipping back to ``none`` must not touch the filesystem.

    This is also the shipped default — without it, a fresh checkout
    couldn't run RAG until UMLS data was provisioned.
    """
    cfg = TermServiceConfig(
        active="none",
        catalog=[
            TermServiceEntry(id="umls_cmekg_local", kind="local"),
            TermServiceEntry(id="none", kind="noop"),
        ],
    )
    svc = build_term_service(cfg)
    assert isinstance(svc, NoOpTermService)
