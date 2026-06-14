"""Canonical (dataset-agnostic) symptoms primitives.

The symptoms feature ships with one dataset (``ddxplus``) today and is
shaped for more. Every dataset has a different native format —
DDXPlus ships JSON with French + English question text, MedDialog-CN
would ship Chinese-only — so the rest of the codebase consumes
:class:`CanonicalDataset` instead of any particular file format.

Three layers:

* :mod:`.canonical` — pure dataclasses describing the normalized form.
* :mod:`.adapter` — the Protocol each dataset-specific adapter implements.
* :mod:`.registry` — id → adapter dispatch + ``build_dataset(spec)`` entry.

Adapters live under :mod:`claritymed.ingest.symptoms.<dataset>.adapter`
and self-register on import. The server's loader calls
:func:`registry.build_dataset` and never sees the native format.
"""

from claritymed.core.symptoms.datasets.adapter import DatasetAdapter
from claritymed.core.symptoms.datasets.canonical import (
    CanonicalCondition,
    CanonicalDataset,
    CanonicalEvidence,
    CanonicalValue,
    LoadedDataset,
    LoadedModel,
    slugify_condition,
)
from claritymed.core.symptoms.datasets.registry import (
    available_adapters,
    build_dataset,
    register_adapter,
    unregister_adapter,
)

__all__ = [
    "CanonicalCondition",
    "CanonicalDataset",
    "CanonicalEvidence",
    "CanonicalValue",
    "DatasetAdapter",
    "LoadedDataset",
    "LoadedModel",
    "available_adapters",
    "build_dataset",
    "register_adapter",
    "slugify_condition",
    "unregister_adapter",
]
