"""CLI-specific tests for ``scripts/init_system_rag.py``.

The pipeline logic — slug, ocr_files, build_yaml_snippet, resolve_metadata
— lives in ``claritymed.ingest.system_rag`` and is covered by
``tests/unit/test_system_rag_pipeline.py``. This file pins only the CLI
surface: argparse validation and the legacy compatibility wrappers
(``_build_yaml_snippet``, ``_resolve_metadata``) that the script keeps
around so other callers can hand it an argparse Namespace.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module():
    """Load ``scripts/init_system_rag.py`` by path.

    ``scripts/`` is not a package on the import path (it ships as raw
    files alongside ``src/``), so the test loads the script via
    importlib instead of adjusting pythonpath project-wide.
    """
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "init_system_rag.py"
    spec = importlib.util.spec_from_file_location(
        "_init_system_rag_under_test", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_script_module()


# --- _parse_args --------------------------------------------------------


def test_parse_args_minimal_happy_path() -> None:
    args = mod._parse_args(
        ["--name", "pneumonia_en", "--topic", "pneumonia", "a.pdf", "b.pdf"]
    )
    assert args.name == "pneumonia_en"
    assert args.topic == ["pneumonia"]
    assert [p.name for p in args.paths] == ["a.pdf", "b.pdf"]
    # ``--language`` / ``--authority-tier`` default to None at parse
    # time so ``_resolve_metadata`` can either inherit from the yaml on
    # append or apply the en/tier-2 defaults for a new collection.
    assert args.language is None
    assert args.authority_tier is None
    assert args.cross_lingual is False
    assert args.license is None
    assert args.dry_run is False
    assert args.dedupe_cosine_threshold == 0.0


def test_parse_args_collects_repeated_topics() -> None:
    args = mod._parse_args(
        [
            "--name",
            "cap_en",
            "--topic",
            "pneumonia",
            "--topic",
            "respiratory infections",
            "--topic",
            "antibiotics",
            "x.pdf",
        ]
    )
    assert args.topic == ["pneumonia", "respiratory infections", "antibiotics"]


def test_parse_args_full_optional_set() -> None:
    args = mod._parse_args(
        [
            "--name",
            "cap_en",
            "--topic",
            "pneumonia",
            "--language",
            "zh",
            "--cross-lingual",
            "--authority-tier",
            "1",
            "--license",
            "ATS/IDSA (educational)",
            "--dedupe-cosine-threshold",
            "0.92",
            "--dry-run",
            "x.pdf",
        ]
    )
    assert args.language == "zh"
    assert args.cross_lingual is True
    assert args.authority_tier == 1
    assert args.license == "ATS/IDSA (educational)"
    assert args.dedupe_cosine_threshold == pytest.approx(0.92)
    assert args.dry_run is True


def test_parse_args_allows_missing_topic() -> None:
    """``--topic`` is now optional at parse time -- inherited on append."""
    args = mod._parse_args(["--name", "cap_en", "a.pdf"])
    assert args.topic == []


def test_parse_args_rejects_invalid_name() -> None:
    with pytest.raises(SystemExit):
        mod._parse_args(["--name", "BadName", "--topic", "x", "a.pdf"])


def test_parse_args_rejects_name_starting_with_digit() -> None:
    with pytest.raises(SystemExit):
        mod._parse_args(["--name", "2007_cap", "--topic", "x", "a.pdf"])


def test_parse_args_rejects_invalid_language() -> None:
    with pytest.raises(SystemExit):
        mod._parse_args(
            ["--name", "cap_en", "--topic", "x", "--language", "fr", "a.pdf"]
        )


def test_parse_args_rejects_invalid_authority_tier() -> None:
    with pytest.raises(SystemExit):
        mod._parse_args(
            [
                "--name",
                "cap_en",
                "--topic",
                "x",
                "--authority-tier",
                "5",
                "a.pdf",
            ]
        )


# --- _build_yaml_snippet (legacy CLI wrapper) ---------------------------


def _make_ns(**overrides) -> argparse.Namespace:
    defaults = dict(
        name="cap_en",
        topic=["pneumonia"],
        language="en",
        cross_lingual=False,
        authority_tier=2,
        license=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_yaml_snippet_legacy_wrapper_minimal() -> None:
    snippet = mod._build_yaml_snippet(_make_ns())
    assert "- name: cap_en" in snippet
    assert "language: en" in snippet
    assert "        - pneumonia" in snippet


# --- _resolve_metadata (legacy CLI wrapper) -----------------------------


def _stub_retrieval_config(collections: list) -> argparse.Namespace:
    return argparse.Namespace(
        system_rag=argparse.Namespace(collections=list(collections))
    )


def _existing_entry(**overrides) -> argparse.Namespace:
    defaults = dict(
        name="cap_en",
        topics=("community-acquired pneumonia", "respiratory infections"),
        language="en",
        cross_lingual=True,
        authority_tier=1,
        license="ATS/IDSA",
        source_uri_prefix="https://www.atsjournals.org/",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fresh_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        name="cap_en",
        topic=[],
        language=None,
        cross_lingual=False,
        authority_tier=None,
        license=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_resolve_metadata_cli_wrapper_inherits_yaml(monkeypatch) -> None:
    """The CLI wrapper still hydrates an argparse Namespace from yaml."""
    from claritymed.ingest import system_rag as pipeline

    existing = _existing_entry()
    monkeypatch.setattr(
        pipeline, "load_retrieval_config", lambda: _stub_retrieval_config([existing])
    )
    # The CLI's _warn_on_yaml_overrides does its own import — patch it too.
    monkeypatch.setattr(
        "claritymed.core.rag.load_retrieval_config",
        lambda: _stub_retrieval_config([existing]),
    )

    args = _fresh_args(name="cap_en")
    mod._resolve_metadata(args)

    assert args.topic == [
        "community-acquired pneumonia",
        "respiratory infections",
    ]
    assert args.language == "en"
    assert args.cross_lingual is True
    assert args.authority_tier == 1
    assert args.license == "ATS/IDSA"
    assert args.source_uri_prefix == "https://www.atsjournals.org/"


def test_resolve_metadata_cli_wrapper_warns_on_override(monkeypatch, capsys) -> None:
    """Explicit CLI flags win, but disagreement surfaces as a yellow warning."""
    from claritymed.ingest import system_rag as pipeline

    existing = _existing_entry()
    monkeypatch.setattr(
        pipeline, "load_retrieval_config", lambda: _stub_retrieval_config([existing])
    )
    monkeypatch.setattr(
        "claritymed.core.rag.load_retrieval_config",
        lambda: _stub_retrieval_config([existing]),
    )

    args = _fresh_args(
        name="cap_en",
        topic=["antibiotics"],
        language="zh",
        authority_tier=3,
        license="custom",
    )
    mod._resolve_metadata(args)

    captured = capsys.readouterr().out
    assert args.topic == ["antibiotics"]
    assert args.language == "zh"
    assert args.authority_tier == 3
    assert args.license == "custom"
    assert "warning" in captured.lower()
