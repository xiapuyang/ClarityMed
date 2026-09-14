"""Tests for ``src/claritymed/evals/tasks/medmcqa.yaml``.

Mirrors ``test_medqa_task.py``. MedMCQA shares the 4-option A–D shape
with MedQA, so the YAML is structurally a near-twin (only ``dataset_path``,
``test_split``, and the field names in the prompt template change). These
tests assert the differences are exactly those — schema, prompt, and
``cop``→letter mapping — and re-exercise the letter-extraction filter
end-to-end so a regex regression here is caught locally instead of
showing up as a quiet accuracy drop in a real eval run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.instance import Instance
from lm_eval.filters import build_filter_ensemble
from lm_eval.tasks import TaskManager

_TASKS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "claritymed"
    / "evals"
    / "tasks"
)
_TASK_NAME = "medmcqa"


@pytest.fixture(scope="module")
def medmcqa_task():
    tm = TaskManager(include_path=str(_TASKS_DIR))
    loaded = tm.load_task_or_group(_TASK_NAME)
    return loaded[_TASK_NAME]


@pytest.fixture(scope="module")
def medmcqa_yaml() -> dict:
    with (_TASKS_DIR / f"{_TASK_NAME}.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _sample_doc() -> dict:
    """A flattened MedMCQA row. ``cop`` is the int index of the correct
    option (0=A..3=D)."""
    return {
        "question": "Which structure does the trochlear nerve innervate?",
        "opa": "Superior oblique",
        "opb": "Lateral rectus",
        "opc": "Medial rectus",
        "opd": "Levator palpebrae",
        "cop": 0,
        "choice_type": "single",
    }


# ---------------------------------------------------------------------------
# Schema and TaskManager load
# ---------------------------------------------------------------------------


def test_task_manager_loads_medmcqa(medmcqa_task):
    assert medmcqa_task is not None
    assert medmcqa_task.config.task == _TASK_NAME
    assert medmcqa_task.config.output_type == "generate_until"


def test_dataset_path_uses_openlifescienceai(medmcqa_yaml):
    """The bundled lm-eval YAML uses bare ``medmcqa`` (resolves via the
    HF datasets script loader, which is removed in datasets>=3). We pin
    the canonical parquet repo explicitly to stay loader-compatible —
    same deviation rationale as the MedQA YAML's GBaker pin."""
    assert medmcqa_yaml["dataset_path"] == "openlifescienceai/medmcqa"
    # ``test`` has no labels in this HF release, so we eval on validation.
    assert medmcqa_yaml["test_split"] == "validation"


def test_generation_kwargs_pinned_for_letter_answer(medmcqa_yaml):
    gen = medmcqa_yaml["generation_kwargs"]
    assert gen["max_gen_toks"] >= 256
    assert gen["temperature"] == 0
    assert gen["do_sample"] is False
    assert "until" in gen and gen["until"] == [], (
        "must set `until: []` so lm-eval doesn't inject `[fewshot_delimiter]`"
    )


def test_filter_list_extracts_letter(medmcqa_yaml):
    filters = medmcqa_yaml["filter_list"]
    assert any(f["name"] == "extract_letter" for f in filters)
    extract = next(f for f in filters if f["name"] == "extract_letter")
    function_names = [step["function"] for step in extract["filter"]]
    assert function_names == ["regex", "take_first"]


def test_metric_list_uses_exact_match(medmcqa_yaml):
    metrics = medmcqa_yaml["metric_list"]
    assert any(m["metric"] == "exact_match" for m in metrics)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


def test_doc_to_text_contains_question_and_all_four_options(medmcqa_task):
    text = medmcqa_task.doc_to_text(_sample_doc())
    assert "trochlear nerve" in text
    assert "(A) Superior oblique" in text
    assert "(B) Lateral rectus" in text
    assert "(C) Medial rectus" in text
    assert "(D) Levator palpebrae" in text
    assert text.rstrip().endswith("Answer:")


def test_doc_to_target_maps_cop_to_letter(medmcqa_task):
    for cop, expected_letter in enumerate(("A", "B", "C", "D")):
        doc = {**_sample_doc(), "cop": cop}
        target = medmcqa_task.doc_to_target(doc).strip()
        assert target == expected_letter, (
            f"cop={cop} should map to {expected_letter}, got {target!r}"
        )


# ---------------------------------------------------------------------------
# Letter-extraction filter (regex + take_first)
# ---------------------------------------------------------------------------


def _build_extract_letter_filter(medmcqa_yaml: dict) -> FilterEnsemble:
    extract = next(
        f for f in medmcqa_yaml["filter_list"] if f["name"] == "extract_letter"
    )
    components = [
        (step["function"], {k: v for k, v in step.items() if k != "function"})
        for step in extract["filter"]
    ]
    return build_filter_ensemble("extract_letter", components)


def _apply(filt: FilterEnsemble, completion: str) -> str:
    inst = Instance(
        request_type="generate_until",
        doc={},
        arguments=("", {}),
        idx=0,
        resps=[completion],
    )
    filt.apply([inst])
    return inst.filtered_resps[filt.name]


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("A", "A"),
        ("d", "d"),
        ("Answer: B", "B"),
        ("The answer is C.", "C"),
        # Reasoning preamble — last "Answer:" wins (group_select: -1).
        ("I considered A and then C, but actually Answer: D", "D"),
        ("Therefore the answer is (B)", "B"),
    ],
)
def test_filter_prefers_trailing_answer(medmcqa_yaml, completion, expected):
    filt = _build_extract_letter_filter(medmcqa_yaml)
    assert _apply(filt, completion) == expected


def test_filter_fallback_on_no_match(medmcqa_yaml):
    filt = _build_extract_letter_filter(medmcqa_yaml)
    assert _apply(filt, "I have no idea") == "[invalid]"


def test_filter_on_empty_completion(medmcqa_yaml):
    filt = _build_extract_letter_filter(medmcqa_yaml)
    assert _apply(filt, "") == "[invalid]"
