"""Tests for ``src/claritymed/evals/tasks/pubmedqa.yaml``.

PubMedQA is the framework-generalization proof case: same lm-eval-harness
scaffolding as MedQA / MedMCQA / CMB-Exam, but the gold answer is a word
(``yes`` / ``no`` / ``maybe``) instead of a single letter. These tests
cover three concerns the other task tests don't:

* Nested ``context.contexts`` rendering — abstracts are a list of
  paragraphs nested under a dict; the Jinja template must flatten them.
* Word-extraction regex against realistic chatty completions, including
  the ``last-match wins`` policy that motivated the letter-task rewrite.
* Dataset-path deviation guard (qiaojin parquet mirror, not the bundled
  bigbio script-loader path that breaks under datasets>=3).
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
_TASK_NAME = "pubmedqa"


@pytest.fixture(scope="module")
def pubmedqa_task():
    tm = TaskManager(include_path=str(_TASKS_DIR))
    loaded = tm.load_task_or_group(_TASK_NAME)
    return loaded[_TASK_NAME]


@pytest.fixture(scope="module")
def pubmedqa_yaml() -> dict:
    with (_TASKS_DIR / f"{_TASK_NAME}.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _sample_doc() -> dict:
    """A doc shaped like the qiaojin/PubMedQA pqa_labeled rows."""
    return {
        "pubid": 12345,
        "question": "Do mitochondria play a role in remodelling lace plant leaves?",
        "context": {
            "contexts": [
                "Programmed cell death (PCD) is the regulated death of cells.",
                "The lace plant produces perforations in its leaves through PCD.",
            ],
        },
        "long_answer": "Results depicted mitochondrial dynamics in vivo as PCD progresses.",
        "final_decision": "yes",
    }


# ---------------------------------------------------------------------------
# Schema and TaskManager load
# ---------------------------------------------------------------------------


def test_task_manager_loads_pubmedqa(pubmedqa_task):
    assert pubmedqa_task is not None
    assert pubmedqa_task.config.task == _TASK_NAME
    assert pubmedqa_task.config.output_type == "generate_until"


def test_dataset_path_uses_qiaojin_parquet_mirror(pubmedqa_yaml):
    """Deviation guard: the bundled lm-eval YAML targets ``bigbio/pubmed_qa``
    which is a script-loader dataset removed in datasets>=3. We pin
    ``qiaojin/PubMedQA`` — same content, parquet-backed, loader-compatible."""
    assert pubmedqa_yaml["dataset_path"] == "qiaojin/PubMedQA"
    assert pubmedqa_yaml["dataset_name"] == "pqa_labeled"
    # pqa_labeled only ships a single ``train`` split (the 1,000
    # expert-labelled questions); we score the whole thing.
    assert pubmedqa_yaml["test_split"] == "train"


def test_generation_kwargs_pinned_for_word_answer(pubmedqa_yaml):
    gen = pubmedqa_yaml["generation_kwargs"]
    assert gen["max_gen_toks"] >= 256
    assert gen["temperature"] == 0
    assert gen["do_sample"] is False
    assert "until" in gen and gen["until"] == [], (
        "must set `until: []` so lm-eval doesn't inject `[fewshot_delimiter]`"
    )


def test_filter_list_extracts_word(pubmedqa_yaml):
    filters = pubmedqa_yaml["filter_list"]
    assert any(f["name"] == "extract_word" for f in filters)
    extract = next(f for f in filters if f["name"] == "extract_word")
    function_names = [step["function"] for step in extract["filter"]]
    assert function_names == ["regex", "take_first"]


def test_metric_list_uses_exact_match(pubmedqa_yaml):
    metrics = pubmedqa_yaml["metric_list"]
    assert any(m["metric"] == "exact_match" for m in metrics)


# ---------------------------------------------------------------------------
# Prompt template — joins nested context.contexts, exposes question
# ---------------------------------------------------------------------------


def test_doc_to_text_joins_abstract_paragraphs(pubmedqa_task):
    text = pubmedqa_task.doc_to_text(_sample_doc())
    assert "Programmed cell death" in text
    assert "lace plant produces perforations" in text
    # The two abstract paragraphs are joined with a newline, not " ".
    assert "cells.\nThe lace plant" in text


def test_doc_to_text_includes_question_and_answer_anchor(pubmedqa_task):
    text = pubmedqa_task.doc_to_text(_sample_doc())
    assert "Do mitochondria" in text
    assert text.rstrip().endswith("Answer:")


def test_doc_to_target_passes_final_decision(pubmedqa_task):
    for decision in ("yes", "no", "maybe"):
        doc = {**_sample_doc(), "final_decision": decision}
        assert pubmedqa_task.doc_to_target(doc).strip() == decision


# ---------------------------------------------------------------------------
# Word-extraction filter (regex + take_first)
# ---------------------------------------------------------------------------


def _build_extract_word_filter(pubmedqa_yaml: dict) -> FilterEnsemble:
    extract = next(
        f for f in pubmedqa_yaml["filter_list"] if f["name"] == "extract_word"
    )
    components = [
        (step["function"], {k: v for k, v in step.items() if k != "function"})
        for step in extract["filter"]
    ]
    return build_filter_ensemble("extract_word", components)


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
        # Compliant single-word replies.
        ("yes", "yes"),
        ("no", "no"),
        ("maybe", "maybe"),
        # Capitalization variants — ``ignore_case: true`` on exact_match
        # would still score these correctly, but extracting the original
        # form keeps the JSONL faithful to the model's output.
        ("Yes", "Yes"),
        ("MAYBE", "MAYBE"),
        # Explicit "Answer: X" anchors.
        ("Answer: yes", "yes"),
        ("The answer is no.", "no"),
        # Reasoning preamble that mentions an alternative — last-match
        # wins (``group_select: -1``), same policy as the letter regex.
        (
            "The abstract supports yes for some cohorts but ultimately, Answer: no",
            "no",
        ),
        # Both yes and no appear; the trailing bare word wins.
        (
            "There is evidence for yes and arguments for no. Conclusion: maybe.",
            "maybe",
        ),
    ],
)
def test_filter_prefers_trailing_answer(pubmedqa_yaml, completion, expected):
    filt = _build_extract_word_filter(pubmedqa_yaml)
    assert _apply(filt, completion) == expected


def test_filter_fallback_on_no_match(pubmedqa_yaml):
    filt = _build_extract_word_filter(pubmedqa_yaml)
    # No yes/no/maybe word → fallback sentinel; exact_match scores wrong.
    assert _apply(filt, "Unclear from the abstract.") == "[invalid]"


def test_filter_on_empty_completion(pubmedqa_yaml):
    filt = _build_extract_word_filter(pubmedqa_yaml)
    assert _apply(filt, "") == "[invalid]"
