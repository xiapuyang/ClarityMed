"""DDXPlus dataset adapter: loaders, training, and data preparation.

Importing this package registers the :class:`DDXPlusAdapter` with
:mod:`claritymed.core.symptoms.datasets.registry`, so the server's
``build_dataset(spec)`` dispatch finds it without an explicit hook.
Adding a second dataset means a sibling package that mirrors this shape.
"""

from claritymed.core.symptoms.datasets.registry import register_adapter
from claritymed.ingest.symptoms.ddxplus.adapter import DDXPlusAdapter

register_adapter(DDXPlusAdapter)

__all__ = ["DDXPlusAdapter"]
