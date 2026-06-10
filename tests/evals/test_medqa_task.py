"""Tests for ``src/claritymed/evals/tasks/medqa.yaml``.

These tests validate three things without ever hitting the network:
1. The task YAML parses through lm-eval-harness's ``TaskManager`` with
   our project-local ``include_path``.
2. The Jinja prompt template renders the expected MCQA shape on a
   hand-built sample.
3. The ``filter_list`` regex extracts the right letter from realistic
   completions, including edge cases the plan called out (chatty replies,
   bare letters, missing matches).

The dataset itself is cached by HF on first use of the test session; we
don't assert against real rows because that would couple test runtime
to dataset downloads.
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
_TASK_NAME = "medqa"


@pytest.fixture(scope="module")
def medqa_task():
    tm = TaskManager(include_path=str(_TASKS_DIR))
    loaded = tm.load_task_or_group(_TASK_NAME)
    return loaded[_TASK_NAME]


@pytest.fixture(scope="module")
def medqa_yaml() -> dict:
    with (_TASKS_DIR / f"{_TASK_NAME}.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _sample_doc() -> dict:
    return {
        "sent1": "A 65-year-old smoker presents with hemoptysis.",
        "ending0": "Order a chest X-ray",
        "ending1": "Reassure and discharge",
        "ending2": "Prescribe antibiotics",
        "ending3": "Refer to dermatology",
        "label": 0,
    }


# ---------------------------------------------------------------------------
# Schema and TaskManager load
# ---------------------------------------------------------------------------


def test_task_manager_loads_medqa(medqa_task):
    assert medqa_task is not None
    assert medqa_task.config.task == _TASK_NAME
    assert medqa_task.config.output_type == "generate_until"


def test_generation_kwargs_pinned_for_letter_answer(medqa_yaml):
    gen = medqa_yaml["generation_kwargs"]
    assert gen["max_gen_toks"] >= 256, (
        "headroom needed for reasoning models that think before answering"
    )
    assert gen["temperature"] == 0
    assert gen["do_sample"] is False
    # ``until`` must be present-but-empty: lm-eval-harness silently injects
    # ``[fewshot_delimiter]`` (= "\n\n") when the key is missing, which
    # would cut reasoning models off at the first paragraph break.
    assert "until" in gen, (
        "must set `until: []` so lm-eval doesn't inject `[fewshot_delimiter]`"
    )
    assert gen["until"] == [], "empty list disables stop-sequence injection"


def test_filter_list_extracts_letter(medqa_yaml):
    filters = medqa_yaml["filter_list"]
    assert any(f["name"] == "extract_letter" for f in filters)
    extract = next(f for f in filters if f["name"] == "extract_letter")
    function_names = [step["function"] for step in extract["filter"]]
    assert function_names == ["regex", "take_first"]


def test_metric_list_uses_exact_match(medqa_yaml):
    metrics = medqa_yaml["metric_list"]
    assert any(m["metric"] == "exact_match" for m in metrics)


def test_dataset_path_is_parquet_friendly(medqa_yaml):
    """Plan-deviation guard: the bigbio script loader doesn't work on
    datasets>=3, so we use the parquet-only GBaker dataset."""
    assert medqa_yaml["dataset_path"] == "GBaker/MedQA-USMLE-4-options-hf"


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


def test_doc_to_text_contains_question_and_all_four_options(medqa_task):
    text = medqa_task.doc_to_text(_sample_doc())
    assert "A 65-year-old smoker" in text
    assert "(A) Order a chest X-ray" in text
    assert "(B) Reassure and discharge" in text
    assert "(C) Prescribe antibiotics" in text
    assert "(D) Refer to dermatology" in text
    assert text.rstrip().endswith("Answer:")


def test_doc_to_target_maps_label_to_letter(medqa_task):
    for label, expected_letter in enumerate(("A", "B", "C", "D")):
        doc = {**_sample_doc(), "label": label}
        # ``doc_to_target`` returns a stripped string for our YAML.
        target = medqa_task.doc_to_target(doc).strip()
        assert target == expected_letter, (
            f"label={label} should map to {expected_letter}, got {target!r}"
        )


# ---------------------------------------------------------------------------
# Letter-extraction filter (regex + take_first)
# ---------------------------------------------------------------------------


def _build_extract_letter_filter(medqa_yaml: dict) -> FilterEnsemble:
    """Reuse our YAML's regex + take_first chain in isolation."""
    extract = next(
        f for f in medqa_yaml["filter_list"] if f["name"] == "extract_letter"
    )
    components = [
        (step["function"], {k: v for k, v in step.items() if k != "function"})
        for step in extract["filter"]
    ]
    return build_filter_ensemble("extract_letter", components)


def _apply(filt: FilterEnsemble, completion: str) -> str:
    """Run ``filt`` against ``completion`` and return the filtered string."""
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
        # Compliant single-letter replies — the system instruction's
        # happy path.
        ("B", "B"),
        ("D", "D"),
        ("a", "a"),
        # Explicit "Answer: X" suffixes — anchored regex prefers these.
        ("Answer: B", "B"),
        ("The answer is C.", "C"),
        # Reasoning models that ramble before concluding — last "Answer:"
        # wins. This is the case that motivated the rewrite.
        (
            "Option A is wrong because... Option B explores... "
            "After analysis, Answer: C",
            "C",
        ),
        # Even when reasoning mentions a different letter in passing,
        # the final "Answer: X" wins.
        (
            "I considered A and then C, but actually Answer: D",
            "D",
        ),
        # Parenthesized final answer ("the answer is (B)") — covered.
        ("Therefore the answer is (B)", "B"),
    ],
)
def test_filter_prefers_trailing_answer(medqa_yaml, completion, expected):
    filt = _build_extract_letter_filter(medqa_yaml)
    assert _apply(filt, completion) == expected


def test_filter_fallback_on_no_match(medqa_yaml):
    """When the model emits no letter, the filter yields the fallback
    sentinel ``"[invalid]"`` — exact_match scores it incorrect, which is
    the right behavior."""
    filt = _build_extract_letter_filter(medqa_yaml)
    assert _apply(filt, "I have no idea") == "[invalid]"


def test_filter_on_empty_completion(medqa_yaml):
    filt = _build_extract_letter_filter(medqa_yaml)
    assert _apply(filt, "") == "[invalid]"
