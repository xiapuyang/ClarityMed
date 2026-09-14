"""Emergency triage gate eval harness — Phase 5 Stage 1.

Loads bilingual cases from ``sources/*.yaml``, runs them through
:class:`~claritymed.core.emergency.EmergencyTriage` across the four
sensitivity profiles, and emits per-profile confusion matrices +
F-beta (β=2) / per-rule recall / adversarial-FPR metrics.

The Stage 1 runner targets the **deterministic** half of the gate:
cases ship with pre-extracted symptoms so the rule engine + profile
overrides are exercised without standing up a local LLM extractor.
Text-only cases (``turns`` instead of ``symptoms``) are skipped when no
extractor is wired — a real local provider will pick them up in the
Stage 2 e2e run.
"""
