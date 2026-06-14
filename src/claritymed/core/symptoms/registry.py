"""Client-side dataset selection used by the symptoms plugin.

Distinct from :mod:`claritymed.core.symptoms.datasets.registry`, which
dispatches *server-side adapter classes* to construct
:class:`~claritymed.core.symptoms.datasets.canonical.LoadedDataset`. This
module sits on the plugin side and picks *which dataset id* to invoke
when the LLM calls ``predict_disease_from_symptoms``.

Resolution order (KTD-9 in the disease-prediction plan):

1. ``hint`` argument when it matches a registered dataset id.
2. ``eligibility_scores`` highest-confidence pick when supplied.
3. Registry-list order — the operator's preferred default.
4. ``min(dataset_id)`` as a deterministic tiebreaker.

``hint`` is **soft**: an LLM hallucination that doesn't match any
registered dataset falls through to (2)/(3) rather than raising —
:class:`~claritymed.errors.UnknownDatasetError` is reserved for the
config-load path. The plugin emits an audit event on hint fall-through
so we can quantify how often it happens.
"""

from __future__ import annotations

from claritymed.core.symptoms.schemas import DatasetSpec


class DatasetRegistry:
    """Holds the enabled subset of :class:`DatasetSpec` for plugin dispatch.

    Constructed once at plugin startup from
    :func:`claritymed.config.load_symptoms_config`'s ``datasets`` list.
    Immutable — request handlers don't mutate it.
    """

    def __init__(self, datasets: list[DatasetSpec]) -> None:
        self._enabled: list[DatasetSpec] = [d for d in datasets if d.enabled]
        # Preserve enable order from the config — operator's intent.
        self._by_id: dict[str, DatasetSpec] = {d.id: d for d in self._enabled}

    def list_enabled(self) -> list[DatasetSpec]:
        """Return the enabled :class:`DatasetSpec`s in registry order."""
        return list(self._enabled)

    def has(self, dataset_id: str) -> bool:
        return dataset_id in self._by_id

    def get(self, dataset_id: str) -> DatasetSpec | None:
        return self._by_id.get(dataset_id)

    def resolve(
        self,
        hint: str | None = None,
        *,
        eligibility_scores: dict[str, float] | None = None,
    ) -> DatasetSpec | None:
        """Pick a dataset per KTD-9 precedence.

        Returns ``None`` when no datasets are enabled — the plugin
        treats this as "feature off" and returns a soft ineligible
        result to the LLM.
        """
        if not self._enabled:
            return None

        # 1. Explicit hint, when it matches.
        if hint:
            picked = self._by_id.get(hint)
            if picked is not None:
                return picked
            # Soft-fail; fall through to eligibility / registry order.

        # 2. Eligibility-driven pick.
        if eligibility_scores:
            scored = [(d, eligibility_scores.get(d.id, 0.0)) for d in self._enabled]
            # Highest score first; ties broken by dataset id (deterministic).
            scored.sort(key=lambda kv: (-kv[1], kv[0].id))
            top, top_score = scored[0]
            # Only accept eligibility-driven pick when it has positive signal —
            # all-zeros means eligibility couldn't decide, fall through.
            if top_score > 0:
                return top

        # 3. Registry order — operator's preferred default.
        # 4. Implicit tiebreaker: list order is already deterministic.
        return self._enabled[0]

    def resolve_or_raise_on_hint_mismatch(self, hint: str) -> DatasetSpec:
        """Strict variant for code paths where hint MUST match (operator config).

        Distinct from :meth:`resolve` (soft) — used only in startup /
        config-load paths, never at LLM tool-call time.
        """
        from claritymed.errors import UnknownDatasetError

        picked = self._by_id.get(hint)
        if picked is None:
            raise UnknownDatasetError(
                f"dataset {hint!r} not in registry; enabled: "
                f"{[d.id for d in self._enabled]!r}"
            )
        return picked
