"""Symptoms-feature data + model surface.

The typed-BASD algorithm body lives in :mod:`claritymed.ingest.symptoms.typed_basd`
and is dataset-agnostic. Per-dataset adapters (DDXPlus, future MedDialog-CN)
live under sibling packages and translate raw corpora into the
:class:`~claritymed.ingest.symptoms.typed_basd.TypedEnv`-compatible shape.

Importing this package transitively imports every registered dataset
adapter, so the server's ``build_dataset(spec)`` dispatch finds them all
without an explicit hook. Add a new dataset by creating a sibling
package whose ``__init__`` calls ``register_adapter(<YourAdapter>)``
and adding the import below.
"""

from claritymed.ingest.symptoms import ddxplus  # noqa: F401 — registers adapter
