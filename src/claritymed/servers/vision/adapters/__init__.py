"""Framework adapters for the vision server.

Each module under this package registers its factory with
``servers/vision/loader.py::_ADAPTERS`` at import time, so importing
the package alone is enough to make every shipped framework available
to the loader.

Adding a new adapter = drop a module here + call ``register_adapter()``
at import — no edit to ``loader.py::load_model_for_spec`` needed.
"""

from __future__ import annotations

# Import the concrete adapter modules for side effects (registration).
# The order doesn't matter; the loader picks via the framework key.
from claritymed.servers.vision.adapters import (  # noqa: F401
    onnx_adapter,
    torch_adapter,
)
