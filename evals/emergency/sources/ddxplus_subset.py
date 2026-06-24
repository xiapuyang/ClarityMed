"""DDXPlus subset generator — **Stage 2 placeholder, not wired in Stage 1.**

Why this file ships empty:

The plan budgets ~80 DDXPlus-derived cases. Producing them requires
translating ``E_xxx`` evidence codes (DDXPlus's internal vocabulary)
into the canonical qualifier strings the emergency rule pack consumes
(``radiation_left_arm``, ``diaphoresis``, ``throat_tightness``, …).
There is no automatic mapping — the translation table has to be
hand-authored against ``release_evidences.json``, then each rule's
qualifier set audited for clinical fidelity. A ~half-day of careful
work, separable from the Stage 1 pipeline smoke test.

Stage 1 ships the harness and the hand-curated public vignettes
(``public_vignettes.yaml``, ~40 cases) — enough volume to exercise
every rule × every profile and produce the first baseline numbers.
Stage 2 wires DDXPlus in for tighter confidence intervals on per-rule
recall.

When you are ready to wire DDXPlus:

1. Read ``demo/ddxplus_demo/ddxplus/release_evidences.json`` and build
   a ``dict[str, str]`` from ``E_xxx`` → canonical qualifier. Only
   cover evidences appearing in the conditions that map to v1 rule
   pack: ``Possible NSTEMI / STEMI``, ``Anaphylaxis``, and any others
   that ship after the Phase 5 rule-pack expansion.
2. Read ``release_conditions.json`` and filter to ``severity == 1``
   conditions whose pathology aligns with a rule id.
3. (Optionally) Read patient files from
   ``release_test_patients.zip`` for case diversity beyond one case
   per condition.
4. For each case, emit a :class:`evals.emergency.schemas.Case` with
   ``source: ddxplus_subset``, ``citation`` pointing at the original
   pathology + severity entry, and ``symptoms`` populated from the
   translated evidences.
5. Write the result to
   ``evals/emergency/sources/ddxplus_subset.yaml`` so the runner
   picks it up automatically via ``discover_case_files()``.

Until then, ``discover_case_files()`` skips this module entirely
(it only globs ``*.yaml``), so the Stage 1 baseline run is
unaffected.
"""

from __future__ import annotations


def generate() -> None:
    """Not implemented — see module docstring for the Stage 2 plan."""
    raise NotImplementedError(
        "ddxplus_subset is a Stage 2 task; Stage 1 ships only the "
        "hand-curated public_vignettes.yaml. See the module docstring "
        "for the translation work this function needs to perform."
    )


if __name__ == "__main__":
    generate()
