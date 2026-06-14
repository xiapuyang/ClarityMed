"""Symptoms feature primitives (dataset/model registries, client, eligibility).

``core/`` modules here are dataset-agnostic and orchestrator-free per the
``core/`` → ``orchestrator/`` direction rule in CLAUDE.md. The plugin that
wires them into the agent loop lives at ``orchestrator/features/symptoms_plugin.py``.
"""
