"""Phoenix-backed lookup for historical case content.

Given a ``case_id`` like ``"save_allergy@v2"``, walk the dataset's
version history newest-first until we find the example whose
``metadata.case_id`` matches. Returns the example's input + output +
metadata as-is so the caller can pretty-print or pipe to jq.

Why walk versions: Phoenix datasets are content-versioned — when a
case revision bumps, a NEW dataset version is created and the prior
version (with the old case content) is retained. The latest version
of ``tool_invoke.ingest`` no longer has ``save_allergy@v2`` once v3
exists; we have to traverse older versions to find it.

Limitations
-----------

A case revision that was *never uploaded* to Phoenix is unrecoverable
via this path — there is no row anywhere with that content. This is
the explicit trade-off of relying on Phoenix as the archive: if the
bench was run with ``--no-phoenix-upload`` or Phoenix was down, that
specific revision is gone. Re-running the same bench against current
``cases.py`` is the only recovery, and only if the same revision is
still in cases.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CaseLookupResult:
    """One historical case content match."""

    case_id: str
    dataset_name: str
    dataset_version_id: str | None
    example_id: str | None
    input: dict[str, Any]
    output: dict[str, Any]
    metadata: dict[str, Any]


class CaseLookupError(RuntimeError):
    """Raised when no match is found across the dataset's version history."""


def dataset_name_for_runner(runner: str) -> str:
    """Same name the uploader uses — keep them in sync explicitly."""
    return f"tool_invoke.{runner}"


def lookup_case(
    *,
    runner: str,
    case_id: str,
    client: Any = None,
) -> CaseLookupResult:
    """Find the most-recent dataset version whose examples include ``case_id``.

    ``client`` is injectable for tests. In production it's
    ``phoenix.client.Client`` against the configured Phoenix endpoint
    (same endpoint resolution as ``phoenix_upload.resolve_endpoint``).

    Walks ``get_dataset_versions`` newest-first because Phoenix
    persists each revision's content in its own version — the latest
    version reflects the *current* cases.py, older versions hold the
    historical bodies we're after.
    """
    if client is None:
        client = _default_client()
    dataset_name = dataset_name_for_runner(runner)
    versions = _list_versions(client, dataset_name)
    if not versions:
        raise CaseLookupError(
            f"dataset {dataset_name!r} has no versions in Phoenix "
            "(was the runner ever uploaded?)"
        )
    # Walk newest-first so a case_id that exists in multiple versions
    # returns the most-recent body. Identical bodies across versions
    # have identical content; choosing the newest just minimizes
    # surprise (it's the form most recently confirmed by an upload).
    for version in versions:
        version_id = _attr(version, "version_id") or _attr(version, "id")
        if not version_id:
            continue
        match = _find_in_version(client, dataset_name, str(version_id), case_id)
        if match is not None:
            return CaseLookupResult(
                case_id=case_id,
                dataset_name=dataset_name,
                dataset_version_id=str(version_id),
                example_id=match.get("example_id"),
                input=match.get("input", {}),
                output=match.get("output", {}),
                metadata=match.get("metadata", {}),
            )
    raise CaseLookupError(
        f"case_id {case_id!r} not found in any version of dataset "
        f"{dataset_name!r}. Either the revision was never uploaded "
        "(--no-phoenix-upload at run time, or Phoenix was down), or "
        "the case_id is mistyped."
    )


# --- internals ------------------------------------------------------


def _default_client() -> Any:
    from phoenix.client import Client

    from tests.benchmarks.tool_invoke.phoenix_upload import (
        PhoenixUnreachable,
        _assert_reachable,
        resolve_endpoint,
    )

    endpoint = resolve_endpoint()
    if endpoint is None:
        raise CaseLookupError(
            "no Phoenix endpoint configured (set tracing.endpoint in "
            "configs/app.yaml or PHOENIX_COLLECTOR_ENDPOINT env var)"
        )
    # Probe before any Phoenix call so the user gets a clear "Phoenix
    # is down" message in seconds instead of a 30s httpx timeout
    # buried deep in the lookup loop.
    try:
        _assert_reachable(endpoint)
    except PhoenixUnreachable as exc:
        raise CaseLookupError(str(exc)) from exc
    import os

    api_key = os.environ.get("PHOENIX_API_KEY", "").strip() or None
    return Client(base_url=endpoint, api_key=api_key)


def _list_versions(client: Any, dataset_name: str) -> list[Any]:
    """Return dataset versions newest-first.

    Phoenix's API returns versions as a list of objects/dicts with at
    least ``version_id`` and ``created_at``. We sort defensively by
    ``created_at`` so a backend that returns oldest-first still gives
    us newest-first iteration.
    """
    try:
        versions = client.datasets.get_dataset_versions(dataset=dataset_name)
    except Exception as exc:  # noqa: BLE001
        raise CaseLookupError(
            f"failed to list versions of dataset {dataset_name!r}: {exc}"
        ) from exc
    items = list(versions or [])

    def _ts(v: Any) -> str:
        ts = _attr(v, "created_at") or ""
        return str(ts)

    items.sort(key=_ts, reverse=True)
    return items


def _find_in_version(
    client: Any, dataset_name: str, version_id: str, case_id: str
) -> Optional[dict[str, Any]]:
    """Return one example dict from ``version_id`` whose
    ``metadata.case_id`` matches, or ``None`` if no match."""
    try:
        dataset = client.datasets.get_dataset(
            dataset=dataset_name, version_id=version_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "case-lookup: failed to fetch %s @ version %s: %s",
            dataset_name,
            version_id,
            exc,
        )
        return None
    examples = _attr(dataset, "examples") or []
    for ex in examples:
        meta = _attr(ex, "metadata") or {}
        if isinstance(meta, dict) and meta.get("case_id") == case_id:
            return {
                "example_id": _attr(ex, "id"),
                "input": _attr(ex, "input") or {},
                "output": _attr(ex, "output") or {},
                "metadata": meta,
            }
    return None


def _attr(obj: Any, name: str) -> Any:
    """Lookup ``name`` on dict / object / pydantic shapes."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


__all__ = [
    "CaseLookupError",
    "CaseLookupResult",
    "dataset_name_for_runner",
    "lookup_case",
]
