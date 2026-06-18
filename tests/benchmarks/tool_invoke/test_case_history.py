"""Phoenix-backed case-history lookup.

Pure unit tests with a stubbed Phoenix client — no server contact.
Exercises:

* Walking versions newest-first.
* Returning the first match.
* Failing loud with a useful message when ``case_id`` is unknown across
  every version (the typo / never-uploaded case).
* Returning the latest version's example when the same ``case_id``
  appears in multiple versions (deterministic, matches what the CLI
  prints).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.benchmarks.tool_invoke import case_history


# --- stubs ----------------------------------------------------------


class _StubExample:
    """One dataset example with the shape Phoenix's client returns."""

    def __init__(
        self,
        *,
        id: str,
        metadata: dict[str, Any],
        input_: dict[str, Any],
        output: dict[str, Any],
    ) -> None:
        self.id = id
        self.metadata = metadata
        self.input = input_
        self.output = output


class _StubDataset:
    def __init__(self, examples: list[_StubExample]) -> None:
        self.examples = examples


class _StubDatasetsApi:
    """Mimic ``phoenix.client.Client.datasets`` for lookup-side calls."""

    def __init__(
        self,
        *,
        versions: list[dict[str, Any]],
        examples_by_version: dict[str, list[_StubExample]],
    ) -> None:
        self._versions = versions
        self._examples_by_version = examples_by_version
        self.fetches: list[tuple[str, str | None]] = []

    def get_dataset_versions(self, *, dataset: str) -> list[dict[str, Any]]:
        return list(self._versions)

    def get_dataset(
        self, *, dataset: str, version_id: str | None = None
    ) -> _StubDataset:
        self.fetches.append((dataset, version_id))
        examples = self._examples_by_version.get(str(version_id), [])
        return _StubDataset(examples=examples)


class _StubClient:
    def __init__(self, datasets: _StubDatasetsApi) -> None:
        self.datasets = datasets


# --- tests ----------------------------------------------------------


def test_lookup_returns_newest_match():
    """When the same case_id appears in multiple versions, return the
    newest version's body."""
    versions = [
        {"version_id": "v-3", "created_at": "2026-06-17T12:00:00Z"},
        {"version_id": "v-2", "created_at": "2026-06-10T12:00:00Z"},
        {"version_id": "v-1", "created_at": "2026-06-01T12:00:00Z"},
    ]
    examples = {
        "v-3": [
            _StubExample(
                id="ex-new",
                metadata={"case_id": "save_allergy@v2"},
                input_={"prompts": {"en": "newer"}},
                output={},
            ),
        ],
        "v-2": [
            _StubExample(
                id="ex-old",
                metadata={"case_id": "save_allergy@v2"},
                input_={"prompts": {"en": "older"}},
                output={},
            ),
        ],
        "v-1": [],
    }
    client = _StubClient(
        _StubDatasetsApi(versions=versions, examples_by_version=examples)
    )
    result = case_history.lookup_case(
        runner="ingest", case_id="save_allergy@v2", client=client
    )
    assert result.dataset_version_id == "v-3"
    assert result.example_id == "ex-new"
    assert result.input["prompts"]["en"] == "newer"


def test_lookup_walks_to_older_version_when_latest_lacks_case():
    """Latest version doesn't have v2 (case has since been bumped to v3);
    we walk back to the version that does."""
    versions = [
        {"version_id": "v-newer", "created_at": "2026-06-17T12:00:00Z"},
        {"version_id": "v-older", "created_at": "2026-06-10T12:00:00Z"},
    ]
    examples = {
        "v-newer": [
            _StubExample(
                id="ex-v3",
                metadata={"case_id": "save_allergy@v3"},
                input_={"prompts": {"en": "v3 body"}},
                output={},
            ),
        ],
        "v-older": [
            _StubExample(
                id="ex-v2",
                metadata={"case_id": "save_allergy@v2"},
                input_={"prompts": {"en": "v2 body"}},
                output={},
            ),
        ],
    }
    client = _StubClient(
        _StubDatasetsApi(versions=versions, examples_by_version=examples)
    )
    result = case_history.lookup_case(
        runner="ingest", case_id="save_allergy@v2", client=client
    )
    assert result.dataset_version_id == "v-older"
    assert result.input["prompts"]["en"] == "v2 body"


def test_lookup_raises_when_no_version_has_case():
    """Case_id not found anywhere → CaseLookupError with the standard
    "either typo or never uploaded" remediation phrasing."""
    versions = [{"version_id": "v-1", "created_at": "2026-06-01T00:00:00Z"}]
    examples = {"v-1": []}
    client = _StubClient(
        _StubDatasetsApi(versions=versions, examples_by_version=examples)
    )
    with pytest.raises(case_history.CaseLookupError, match="not found in any version"):
        case_history.lookup_case(
            runner="ingest", case_id="ghost_case@v9", client=client
        )


def test_lookup_raises_when_dataset_has_no_versions():
    """Empty version list → distinct error so the user knows to check
    upload history, not the case_id spelling."""
    client = _StubClient(_StubDatasetsApi(versions=[], examples_by_version={}))
    with pytest.raises(case_history.CaseLookupError, match="no versions in Phoenix"):
        case_history.lookup_case(
            runner="ingest", case_id="save_allergy@v1", client=client
        )


def test_dataset_name_matches_uploader():
    """Lookup and uploader MUST agree on dataset name or the lookup
    will silently fail (Phoenix has the data under a different name)."""
    from tests.benchmarks.tool_invoke import phoenix_upload

    assert case_history.dataset_name_for_runner(
        "ingest"
    ) == phoenix_upload._dataset_name("ingest")
    assert case_history.dataset_name_for_runner(
        "symptoms"
    ) == phoenix_upload._dataset_name("symptoms")
    assert case_history.dataset_name_for_runner(
        "vision"
    ) == phoenix_upload._dataset_name("vision")
