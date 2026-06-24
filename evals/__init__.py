"""Off-package eval tooling.

The ``evals`` tree sits outside ``src/claritymed`` so it is not measured
against the package coverage threshold and is not shipped with the
wheel. It is added to pytest's ``pythonpath`` so the unit tests under
``tests/evals/`` can import the runner + metrics without the heavy
``src.*`` package boundary.
"""
