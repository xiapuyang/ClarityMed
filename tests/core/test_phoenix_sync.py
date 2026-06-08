"""YAML <-> Phoenix prompt sync.

The sync is a typed orchestration over ``phoenix.client.Client``; tests
inject a stub client so they don't depend on Phoenix being reachable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from claritymed.core.prompts.phoenix_sync import (
    LANGUAGES,
    PRODUCTION_TAG,
    SyncReport,
    phoenix_prompt_name,
    pull,
    push,
)


# --- stub client --------------------------------------------------------


class _StubPromptVersion:
    """Mimics phoenix.client.types.PromptVersion just enough for sync code."""

    def __init__(self, content: str, version_id: str = "v-1") -> None:
        self.id = version_id
        self._content = content

    def format(self):
        # Mirrors PromptVersion.format() which returns an OpenAIPrompt with
        # a ``messages`` attribute (list of dicts).
        class _OAI:
            def __init__(self, messages):
                self.messages = messages

        return _OAI([{"role": "system", "content": self._content}])


class _StubTagsApi:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create(self, *, prompt_version_id, name, description=""):
        self.created.append(
            {
                "prompt_version_id": prompt_version_id,
                "name": name,
                "description": description,
            }
        )


class _StubPromptsApi:
    def __init__(self) -> None:
        self.store: dict[str, _StubPromptVersion] = {}
        self.created: list[dict[str, Any]] = []
        self.tags = _StubTagsApi()
        self._next_id = 1

    def get(self, *, prompt_identifier=None, tag=None, prompt_version_id=None):
        key = f"{prompt_identifier}:{tag}"
        v = self.store.get(key)
        if v is None:
            raise RuntimeError(f"prompt not found: {key}")
        return v

    def create(self, *, version, name, prompt_description=None, prompt_metadata=None):
        vid = f"v-{self._next_id}"
        self._next_id += 1
        content = version.format().messages[0]["content"]
        created = _StubPromptVersion(content, version_id=vid)
        self.store[f"{name}:{PRODUCTION_TAG}"] = created  # will be re-tagged below
        self.created.append(
            {
                "name": name,
                "content": content,
                "prompt_description": prompt_description,
            }
        )
        return created


class _StubClient:
    def __init__(self) -> None:
        self.prompts = _StubPromptsApi()


# --- fixtures -----------------------------------------------------------


@pytest.fixture
def stub_client() -> _StubClient:
    return _StubClient()


@pytest.fixture
def store_dir(tmp_path: Path) -> Path:
    d = tmp_path / "prompts_store"
    d.mkdir()
    return d


def _write_prompt_yaml(
    store_dir: Path, name: str, en: str, zh: str, version: str = "v1"
) -> Path:
    path = store_dir / f"{name}.yaml"
    payload = {
        "name": name,
        "description": f"{name} description",
        "versions": [
            {
                "version": version,
                "created_at": "2026-06-07",
                "notes": "initial",
                "languages": {"en": en, "zh": zh},
            }
        ],
    }
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)
    return path


# --- push ---------------------------------------------------------------


def test_push_creates_one_phoenix_prompt_per_language(stub_client, store_dir):
    _write_prompt_yaml(store_dir, "ask", en="EN prompt", zh="ZH prompt")
    report = push(client=stub_client, store_dir=store_dir)

    assert report.direction == "push"
    actions = sorted(e.action for e in report.entries)
    assert actions == ["pushed", "pushed"]
    created_names = sorted(c["name"] for c in stub_client.prompts.created)
    assert created_names == [
        phoenix_prompt_name("ask", "en"),
        phoenix_prompt_name("ask", "zh"),
    ]
    # Each created version was tagged production.
    tags = sorted(t["name"] for t in stub_client.prompts.tags.created)
    assert tags == [PRODUCTION_TAG, PRODUCTION_TAG]


def test_push_skips_when_phoenix_already_matches(stub_client, store_dir):
    _write_prompt_yaml(store_dir, "ask", en="EN prompt", zh="ZH prompt")
    # Seed Phoenix with matching content
    for lang in LANGUAGES:
        text = "EN prompt" if lang == "en" else "ZH prompt"
        stub_client.prompts.store[
            f"{phoenix_prompt_name('ask', lang)}:{PRODUCTION_TAG}"
        ] = _StubPromptVersion(text)

    report = push(client=stub_client, store_dir=store_dir)
    assert all(e.action == "skipped" for e in report.entries)
    assert stub_client.prompts.created == []


def test_push_dry_run_does_not_call_create(stub_client, store_dir):
    _write_prompt_yaml(store_dir, "ask", en="EN", zh="ZH")
    report = push(client=stub_client, store_dir=store_dir, dry_run=True)
    assert all(e.action == "pushed" for e in report.entries)
    assert stub_client.prompts.created == []


def test_push_filters_by_name(stub_client, store_dir):
    _write_prompt_yaml(store_dir, "ask", en="A_en", zh="A_zh")
    _write_prompt_yaml(store_dir, "rag", en="R_en", zh="R_zh")
    report = push(name="ask", client=stub_client, store_dir=store_dir)
    assert {e.prompt_name for e in report.entries} == {"ask"}
    assert {c["name"] for c in stub_client.prompts.created} == {
        phoenix_prompt_name("ask", "en"),
        phoenix_prompt_name("ask", "zh"),
    }


def test_push_unknown_name_raises(stub_client, store_dir):
    _write_prompt_yaml(store_dir, "ask", en="A", zh="A")
    with pytest.raises(ValueError):
        push(name="bogus", client=stub_client, store_dir=store_dir)


def test_push_records_phoenix_errors_per_entry(store_dir):
    class _BrokenPrompts:
        tags = _StubTagsApi()

        def get(self, **kw):
            raise RuntimeError("network down")

        def create(self, **kw):
            raise RuntimeError("network down")

    class _BrokenClient:
        prompts = _BrokenPrompts()

    _write_prompt_yaml(store_dir, "ask", en="EN", zh="ZH")
    report = push(client=_BrokenClient(), store_dir=store_dir)
    assert all(e.action == "error" for e in report.entries)
    assert len(report.errors) == 2


# --- pull ---------------------------------------------------------------


def test_pull_default_overwrites_latest_version_in_place(stub_client, store_dir):
    """Default mode: same version string, languages replaced, ``created_at``
    bumped to today. Git diff is just the prompt text — no extra block."""
    yaml_path = _write_prompt_yaml(store_dir, "ask", en="OLD_en", zh="OLD_zh")
    stub_client.prompts.store[
        f"{phoenix_prompt_name('ask', 'en')}:{PRODUCTION_TAG}"
    ] = _StubPromptVersion("NEW_en")
    stub_client.prompts.store[
        f"{phoenix_prompt_name('ask', 'zh')}:{PRODUCTION_TAG}"
    ] = _StubPromptVersion("NEW_zh")

    report = pull(client=stub_client, store_dir=store_dir)
    actions = sorted(e.action for e in report.entries)
    assert actions == ["pulled", "pulled"]

    on_disk = yaml.safe_load(yaml_path.read_text("utf-8"))
    assert len(on_disk["versions"]) == 1, "default pull must not add a version"
    v0 = on_disk["versions"][0]
    assert v0["version"] == "v1"
    assert v0["languages"]["en"] == "NEW_en"
    assert v0["languages"]["zh"] == "NEW_zh"


def test_pull_into_new_version_appends_block_with_auto_vN_plus_1(
    stub_client, store_dir
):
    yaml_path = _write_prompt_yaml(store_dir, "ask", en="OLD_en", zh="OLD_zh")
    stub_client.prompts.store[
        f"{phoenix_prompt_name('ask', 'en')}:{PRODUCTION_TAG}"
    ] = _StubPromptVersion("NEW_en")
    stub_client.prompts.store[
        f"{phoenix_prompt_name('ask', 'zh')}:{PRODUCTION_TAG}"
    ] = _StubPromptVersion("NEW_zh")

    pull(client=stub_client, store_dir=store_dir, into_new_version=True)

    on_disk = yaml.safe_load(yaml_path.read_text("utf-8"))
    assert [v["version"] for v in on_disk["versions"]] == ["v1", "v2"]
    assert on_disk["versions"][0]["languages"]["en"] == "OLD_en"  # preserved
    assert on_disk["versions"][-1]["languages"]["en"] == "NEW_en"


def test_pull_with_explicit_version_name_implies_new_version(stub_client, store_dir):
    """Passing ``new_version_name`` alone implies append mode — callers
    don't need to set both flags."""
    yaml_path = _write_prompt_yaml(store_dir, "ask", en="OLD_en", zh="OLD_zh")
    stub_client.prompts.store[
        f"{phoenix_prompt_name('ask', 'en')}:{PRODUCTION_TAG}"
    ] = _StubPromptVersion("NEW_en")
    stub_client.prompts.store[
        f"{phoenix_prompt_name('ask', 'zh')}:{PRODUCTION_TAG}"
    ] = _StubPromptVersion("NEW_zh")

    pull(client=stub_client, store_dir=store_dir, new_version_name="v1.1")

    on_disk = yaml.safe_load(yaml_path.read_text("utf-8"))
    assert [v["version"] for v in on_disk["versions"]] == ["v1", "v1.1"]
    assert on_disk["versions"][-1]["languages"]["en"] == "NEW_en"


def test_pull_skips_when_yaml_already_matches(stub_client, store_dir):
    _write_prompt_yaml(store_dir, "ask", en="SAME_en", zh="SAME_zh")
    stub_client.prompts.store[
        f"{phoenix_prompt_name('ask', 'en')}:{PRODUCTION_TAG}"
    ] = _StubPromptVersion("SAME_en")
    stub_client.prompts.store[
        f"{phoenix_prompt_name('ask', 'zh')}:{PRODUCTION_TAG}"
    ] = _StubPromptVersion("SAME_zh")

    report = pull(client=stub_client, store_dir=store_dir)
    assert all(e.action == "skipped" for e in report.entries)


def test_pull_dry_run_does_not_touch_yaml(stub_client, store_dir):
    yaml_path = _write_prompt_yaml(store_dir, "ask", en="OLD_en", zh="OLD_zh")
    original = yaml_path.read_text("utf-8")
    stub_client.prompts.store[
        f"{phoenix_prompt_name('ask', 'en')}:{PRODUCTION_TAG}"
    ] = _StubPromptVersion("NEW_en")
    stub_client.prompts.store[
        f"{phoenix_prompt_name('ask', 'zh')}:{PRODUCTION_TAG}"
    ] = _StubPromptVersion("NEW_zh")

    report = pull(client=stub_client, store_dir=store_dir, dry_run=True)
    assert all(e.action == "pulled" for e in report.entries)
    assert yaml_path.read_text("utf-8") == original  # untouched


def test_pull_missing_production_tag_reports_missing(stub_client, store_dir):
    _write_prompt_yaml(store_dir, "ask", en="EN", zh="ZH")

    # "not found" sentinel — _try_get_production interprets this as missing
    class _Missing:
        def __init__(self) -> None:
            self.tags = _StubTagsApi()

        def get(self, **kw):
            raise RuntimeError("prompt not found: deadbeef")

        def create(self, **kw):
            raise AssertionError("should not be called on pull")

    class _Client:
        prompts = _Missing()

    report = pull(client=_Client(), store_dir=store_dir)
    assert all(e.action == "missing" for e in report.entries)


def test_pull_preserves_unchanged_language_when_only_one_diverged(
    stub_client, store_dir
):
    """If EN changed on Phoenix but ZH did not, the new YAML version
    inherits the ZH text from the existing latest YAML version — both
    languages stay valid per the registry's bilingual invariant."""
    yaml_path = _write_prompt_yaml(store_dir, "ask", en="OLD_en", zh="SAME_zh")
    stub_client.prompts.store[
        f"{phoenix_prompt_name('ask', 'en')}:{PRODUCTION_TAG}"
    ] = _StubPromptVersion("NEW_en")
    stub_client.prompts.store[
        f"{phoenix_prompt_name('ask', 'zh')}:{PRODUCTION_TAG}"
    ] = _StubPromptVersion("SAME_zh")

    pull(client=stub_client, store_dir=store_dir)
    on_disk = yaml.safe_load(yaml_path.read_text("utf-8"))
    new_version = on_disk["versions"][-1]
    assert new_version["languages"]["en"] == "NEW_en"
    assert new_version["languages"]["zh"] == "SAME_zh"


def test_sync_report_changed_vs_errors():
    from claritymed.core.prompts.phoenix_sync import SyncEntry

    report = SyncReport(
        direction="push",
        dry_run=False,
        entries=[
            SyncEntry(
                prompt_name="a", language="en", phoenix_name="x", action="pushed"
            ),
            SyncEntry(
                prompt_name="b", language="zh", phoenix_name="y", action="skipped"
            ),
            SyncEntry(prompt_name="c", language="en", phoenix_name="z", action="error"),
        ],
    )
    assert {e.prompt_name for e in report.changed} == {"a"}
    assert {e.prompt_name for e in report.errors} == {"c"}
