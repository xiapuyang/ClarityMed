"""Dataset adapter registry — process-wide id → adapter dispatch.

Adapters self-register on import. The server's loader doesn't import
adapters directly; it imports
:mod:`claritymed.ingest.symptoms` (the package init triggers each
sub-package's adapter registration) and then calls :func:`build_dataset`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from claritymed.core.symptoms.datasets.adapter import DatasetAdapter
from claritymed.core.symptoms.datasets.canonical import LoadedDataset
from claritymed.core.symptoms.schemas import DatasetSpec, ModelSpec
from claritymed.errors import UnknownDatasetError

if TYPE_CHECKING:
    from claritymed.core.symptoms.init_matcher import InitMatcherEmbedder

_ADAPTERS: dict[str, type[DatasetAdapter]] = {}


def register_adapter(adapter: type[DatasetAdapter]) -> None:
    """Register an adapter class. Idempotent: re-registration is a no-op."""
    dataset_id = adapter.dataset_id
    if not dataset_id:
        raise ValueError(
            f"adapter {adapter.__name__} has no dataset_id; set the class attribute"
        )
    existing = _ADAPTERS.get(dataset_id)
    if existing is adapter:
        return
    if existing is not None:
        raise ValueError(
            f"dataset {dataset_id!r} already registered to {existing.__name__}; "
            f"refusing to clobber with {adapter.__name__}"
        )
    _ADAPTERS[dataset_id] = adapter


def unregister_adapter(dataset_id: str) -> None:
    """Test helper — drop a registration so a fresh registration can happen."""
    _ADAPTERS.pop(dataset_id, None)


def available_adapters() -> list[str]:
    return sorted(_ADAPTERS.keys())


def build_dataset(
    spec: DatasetSpec,
    model_specs: Iterable[ModelSpec],
    *,
    device: str,
    init_matcher: "InitMatcherEmbedder | None" = None,
) -> LoadedDataset:
    """Dispatch to the adapter registered for ``spec.id`` and load.

    Filters ``model_specs`` down to the ids the dataset references —
    adapters never see specs they don't need. ``init_matcher`` is the
    shared singleton embedder for chief-complaint matching; adapters
    use it to build the per-dataset init-symptom catalog at load time
    (see :class:`~claritymed.core.symptoms.init_matcher.InitMatcherEmbedder`).
    ``None`` disables the catalog build and the runtime falls back to
    zero-init state.

    Raises:
        UnknownDatasetError: No adapter is registered for ``spec.id``.
            Operator-facing — points at the adapter import path so a
            missing-adapter case is easy to diagnose.
    """
    adapter_key = spec.resolved_adapter_id()
    adapter = _ADAPTERS.get(adapter_key)
    if adapter is None:
        raise UnknownDatasetError(
            f"no adapter registered for adapter_id {adapter_key!r} "
            f"(dataset {spec.id!r}). Registered: {available_adapters()!r}. "
            f"Add a register_adapter(...) call in the dataset's adapter "
            f"module, or set adapter_id on the spec to an existing one."
        )
    by_id = {m.id: m for m in model_specs}
    needed = {mid: by_id[mid] for mid in spec.model_ids if mid in by_id}
    missing = [mid for mid in spec.model_ids if mid not in by_id]
    if missing:
        raise UnknownDatasetError(
            f"dataset {spec.id!r} references model ids not in models[]: {missing!r}"
        )
    return adapter.load(spec, needed, device=device, init_matcher=init_matcher)
