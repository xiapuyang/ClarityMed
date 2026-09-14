"""MCQA evaluation subsystem — see ``docs/plans/2026-06-09-002-feat-evals-medqa-plan.md``.

This package wires lm-evaluation-harness to the project's provider catalog
so any model declared in ``configs/models.yaml`` can be scored on MedQA
(and follow-on CMB/MedMCQA/PubMedQA YAML tasks) with one CLI call.

Layout:

* ``lm/``       — ``lm_eval.api.model.LM`` subclasses bridging to pydantic-ai.
* ``runners/``  — drives ``lm_eval.simple_evaluate`` and writes JSONL results.
* ``tasks/``    — lm-eval-harness task YAMLs (one per benchmark).
* ``reporting/``— Phase 2 baseline-vs-RAG delta tables.

Import side-effect free: ``import claritymed.evals`` does not import lm-eval.
"""
