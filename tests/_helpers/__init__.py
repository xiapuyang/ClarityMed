"""Shared test helpers reusable across unit + e2e suites.

Helpers exposed here must be **stub transports** or **test data builders**,
not production-logic shims — anything that simulates real behaviour belongs
in ``src/`` so the tests exercise it.
"""
