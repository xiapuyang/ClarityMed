"""Standalone A/B benchmark for eligibility strategies.

Not part of the ``tool_invoke`` family — eligibility is a sub-component
that runs *before* any LLM tool call, so it deserves a dedicated runner
that measures the matching layer in isolation (precision / recall /
latency per strategy) rather than entangling it with model
tool-selection behavior.

See ``run.py`` for the harness and ``cases.py`` for the curated
complaint set.
"""
