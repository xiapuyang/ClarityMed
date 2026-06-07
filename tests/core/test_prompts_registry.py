"""Tests for ``claritymed.core.prompts.registry``."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from claritymed.context import language_ctx
from claritymed.core.prompts.registry import (
    PromptNotFound,
    PromptRegistry,
    PromptVersionNotFound,
)


def _write(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _two_lang(en: str = "hello", zh: str = "你好") -> dict[str, str]:
    return {"en": en, "zh": zh}


def _valid_payload(name: str = "sample") -> dict:
    return {
        "name": name,
        "description": "sample prompt for tests",
        "versions": [
            {
                "version": "v1",
                "created_at": date(2026, 6, 1),
                "notes": "initial",
                "languages": _two_lang(),
            }
        ],
    }


@pytest.fixture
def store(tmp_path: Path) -> Path:
    d = tmp_path / "store"
    d.mkdir()
    return d


def test_get_returns_template_for_language(store):
    _write(store / "sample.yaml", _valid_payload())
    reg = PromptRegistry(store_dir=store)
    assert reg.get("sample", language="en") == "hello"
    assert reg.get("sample", language="zh") == "你好"


def test_latest_picks_highest_created_at(store):
    payload = _valid_payload()
    payload["versions"].append(
        {
            "version": "v2",
            "created_at": date(2026, 6, 10),
            "notes": "added guardrail",
            "languages": _two_lang(en="hello v2", zh="你好 v2"),
        }
    )
    _write(store / "sample.yaml", payload)
    reg = PromptRegistry(store_dir=store)
    assert reg.get("sample", version="latest", language="en") == "hello v2"
    assert reg.get("sample", version="v1", language="en") == "hello"


def test_language_falls_through_to_language_ctx(store):
    _write(store / "sample.yaml", _valid_payload())
    reg = PromptRegistry(store_dir=store)
    token = language_ctx.set("zh")
    try:
        assert reg.get("sample") == "你好"
    finally:
        language_ctx.reset(token)


def test_list_returns_sorted_names(store):
    _write(store / "alpha.yaml", _valid_payload(name="alpha"))
    _write(store / "beta.yaml", _valid_payload(name="beta"))
    reg = PromptRegistry(store_dir=store)
    assert reg.list() == ["alpha", "beta"]


def test_prompt_not_found(store):
    reg = PromptRegistry(store_dir=store)
    with pytest.raises(PromptNotFound):
        reg.get("missing")


def test_prompt_version_not_found(store):
    _write(store / "sample.yaml", _valid_payload())
    reg = PromptRegistry(store_dir=store)
    with pytest.raises(PromptVersionNotFound):
        reg.get("sample", version="v99", language="en")


def test_missing_language_in_yaml_raises_on_load(store):
    bad = _valid_payload()
    bad["versions"][0]["languages"] = {"en": "only english"}
    _write(store / "sample.yaml", bad)
    with pytest.raises(Exception):  # ValidationError from pydantic
        PromptRegistry(store_dir=store)


def test_extra_top_level_keys_rejected(store):
    bad = _valid_payload()
    bad["unexpected_field"] = "should fail"
    _write(store / "sample.yaml", bad)
    with pytest.raises(Exception):
        PromptRegistry(store_dir=store)


def test_filename_must_match_prompt_name(store):
    payload = _valid_payload(name="actual_name")
    _write(store / "different_name.yaml", payload)
    with pytest.raises(ValueError, match="does not match"):
        PromptRegistry(store_dir=store)


def test_reload_picks_up_new_versions(store):
    payload = _valid_payload()
    _write(store / "sample.yaml", payload)
    reg = PromptRegistry(store_dir=store)
    payload["versions"].append(
        {
            "version": "v2",
            "created_at": date(2026, 6, 10),
            "notes": "new",
            "languages": _two_lang(en="v2 en", zh="v2 zh"),
        }
    )
    _write(store / "sample.yaml", payload)
    # Cache is still v1 until we reload.
    assert reg.get("sample", version="latest", language="en") == "hello"
    reg.reload()
    assert reg.get("sample", version="latest", language="en") == "v2 en"


def test_real_store_loads_without_error():
    """All store entries shipped in the repo must validate at startup."""
    reg = PromptRegistry()
    assert "patient_qa_system_prompt" in reg.list()
    # And no version is missing a language.
    en = reg.get("patient_qa_system_prompt", language="en")
    zh = reg.get("patient_qa_system_prompt", language="zh")
    assert en.strip()
    assert zh.strip()
    assert en != zh  # bilingual content really differs
