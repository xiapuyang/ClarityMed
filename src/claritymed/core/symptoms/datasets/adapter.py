"""Dataset adapter Protocol — the seam each dataset plugs into.

Implementations live next to their native data (e.g.
:mod:`claritymed.ingest.symptoms.ddxplus.adapter`) and self-register on
import via :func:`claritymed.core.symptoms.datasets.registry.register_adapter`.

Adapters are lightweight classes (typically just a ``dataset_id``
attribute + a ``load`` classmethod) so we can reference them in tests
without instantiating heavy resources.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from claritymed.core.symptoms.datasets.canonical import LoadedDataset
from claritymed.core.symptoms.schemas import DatasetSpec, ModelSpec


@runtime_checkable
class DatasetAdapter(Protocol):
    """Pluggable adapter that materializes one dataset into the canonical form.

    Implementations must expose ``dataset_id`` as a class attribute so
    the registry can dispatch by config-supplied ``DatasetSpec.id``.
    They must also implement :meth:`load` as a classmethod / staticmethod —
    no instance state is expected.
    """

    dataset_id: ClassVar[str]

    @classmethod
    def load(
        cls,
        spec: DatasetSpec,
        model_specs: dict[str, ModelSpec],
        *,
        device: str,
    ) -> LoadedDataset:
        """Build a :class:`LoadedDataset` from the dataset's native files.

        Args:
            spec: The dataset's config entry. ``spec.model_ids`` lists the
                model checkpoints to load; one :class:`LoadedModel` per id
                is materialized.
            model_specs: ``{model_id: ModelSpec}`` lookup for every model
                id the dataset references. The adapter does not load
                models that aren't in ``spec.model_ids``.
            device: Resolved torch device string (``"cpu"`` / ``"mps"`` /
                ``"cuda"``) — passed through to :func:`build_basd`.

        Raises:
            FileNotFoundError: A required native file or weights file is
                missing under :data:`CLARITYMED_HOME`.
            RuntimeError: The two-level manifest sha256 chain (KTD-6)
                failed at either link.
        """
        ...
