"""Tests for ``src/claritymed/evals/tasks/cmb_exam.yaml`` and its
sibling ``utils_cmb_exam.process_docs`` preprocessor.

Mirrors ``test_medqa_task.py`` and adds two CMB-specific concerns:

* The dataset's ``option`` field is a JSON string, so we exercise the
  ``process_docs`` helper directly on synthetic rows (no HF download).
* CMB has 5-option questions, so the letter-extraction regex must cover
  ``[A-E]`` and the Chinese ``答案：`` anchor — both verified below.
"""

from __future__ import annotations

import json
from pathlib import Path

import datasets
import pytest
import yaml
from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.instance import Instance
from lm_eval.filters import build_filter_ensemble
from lm_eval.tasks import TaskManager

from claritymed.evals.tasks import utils_cmb_exam

_TASKS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "claritymed"
    / "evals"
    / "tasks"
)
_TASK_NAME = "cmb_exam"


@pytest.fixture(scope="module")
def cmb_task():
    tm = TaskManager(include_path=str(_TASKS_DIR))
    loaded = tm.load_task_or_group(_TASK_NAME)
    return loaded[_TASK_NAME]


class _IgnoreFunctionTagLoader(yaml.SafeLoader):
    """SafeLoader that tolerates lm-eval's ``!function`` tag.

    The TaskManager fixture above resolves ``!function utils_cmb_exam...``
    into the real callable; the raw-YAML fixture below only inspects
    string-valued fields (dataset_path, filter_list, ...) so we don't
    need the resolution — just don't crash on the tag.
    """


_IgnoreFunctionTagLoader.add_constructor(
    "!function", lambda loader, node: f"<function:{node.value}>"
)


@pytest.fixture(scope="module")
def cmb_yaml() -> dict:
    with (_TASKS_DIR / f"{_TASK_NAME}.yaml").open("r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_IgnoreFunctionTagLoader)


def _sample_single_choice_doc() -> dict:
    """A flattened doc shaped the way ``process_docs`` leaves it."""
    return {
        "question": "下列哪种感染最常引起 HIV 患者的肺炎？",
        "answer": "D",
        "question_type": "单项选择题",
        "option_a": "大叶性肺炎",
        "option_b": "小叶性肺炎",
        "option_c": "非典型肺炎",
        "option_d": "卡氏囊虫性肺炎",
        "option_e": "",
    }


# ---------------------------------------------------------------------------
# Schema and TaskManager load
# ---------------------------------------------------------------------------


def test_task_manager_loads_cmb_exam(cmb_task):
    assert cmb_task is not None
    assert cmb_task.config.task == _TASK_NAME
    assert cmb_task.config.output_type == "generate_until"


def test_dataset_path_targets_freedomintelligence_subset(cmb_yaml):
    assert cmb_yaml["dataset_path"] == "FreedomIntelligence/CMB"
    # ``val`` is the only split with answers that's small enough to score;
    # ``test`` is the hidden-label leaderboard split, and its missing
    # ``answer`` field breaks datasets schema inference when loading the
    # full ``CMB-Exam`` config — see ``dataset_kwargs`` deviation guard.
    assert cmb_yaml["test_split"] == "val"


def test_dataset_kwargs_pins_val_to_avoid_broken_test_split(cmb_yaml):
    """Deviation guard: must pin ``data_files`` to val JSON only.

    Loading the bundled ``CMB-Exam`` config pulls all three splits and
    fails schema inference because the ``test`` split JSON is
    answer-less. The fix lives in the YAML's ``dataset_kwargs.data_files``
    block; this test makes accidental reversion loud."""
    files = cmb_yaml["dataset_kwargs"]["data_files"]
    assert "val" in files
    assert files["val"].endswith("CMB-val-merge.json")
    assert "test" not in files


def test_generation_kwargs_pinned_for_letter_answer(cmb_yaml):
    gen = cmb_yaml["generation_kwargs"]
    assert gen["max_gen_toks"] >= 256
    assert gen["temperature"] == 0
    assert gen["do_sample"] is False
    assert "until" in gen and gen["until"] == [], (
        "must set `until: []` so lm-eval doesn't inject `[fewshot_delimiter]`"
    )


def test_filter_list_extracts_letter(cmb_yaml):
    filters = cmb_yaml["filter_list"]
    assert any(f["name"] == "extract_letter" for f in filters)
    extract = next(f for f in filters if f["name"] == "extract_letter")
    function_names = [step["function"] for step in extract["filter"]]
    assert function_names == ["regex", "take_first"]


def test_metric_list_uses_exact_match(cmb_yaml):
    metrics = cmb_yaml["metric_list"]
    assert any(m["metric"] == "exact_match" for m in metrics)


# ---------------------------------------------------------------------------
# process_docs: filter to single-choice + flatten JSON options
# ---------------------------------------------------------------------------


def _fake_dataset(rows: list[dict]) -> datasets.Dataset:
    return datasets.Dataset.from_list(rows)


def test_process_docs_drops_multi_choice_rows():
    """Multi-choice rows have answers like ``"BCDE"`` that the
    single-letter exact_match scorer would silently mark wrong; the
    cleaner contract is to drop them."""
    ds = _fake_dataset(
        [
            {
                "question": "single?",
                "answer": "A",
                "question_type": "单项选择题",
                "option": json.dumps({"A": "x", "B": "y", "C": "z", "D": "w"}),
            },
            {
                "question": "multi?",
                "answer": "ABCD",
                "question_type": "多项选择题",
                "option": json.dumps({"A": "x", "B": "y", "C": "z", "D": "w"}),
            },
        ]
    )
    out = utils_cmb_exam.process_docs(ds)
    assert len(out) == 1
    assert out[0]["question"] == "single?"


def test_process_docs_flattens_json_options_into_letters():
    ds = _fake_dataset(
        [
            {
                "question": "?",
                "answer": "B",
                "question_type": "单项选择题",
                "option": json.dumps(
                    {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}
                ),
            }
        ]
    )
    row = utils_cmb_exam.process_docs(ds)[0]
    assert row["option_a"] == "alpha"
    assert row["option_b"] == "beta"
    assert row["option_c"] == "gamma"
    assert row["option_d"] == "delta"
    # Missing E → empty string so the Jinja template skips the (E) line.
    assert row["option_e"] == ""


def test_process_docs_carries_fifth_option_when_present():
    ds = _fake_dataset(
        [
            {
                "question": "?",
                "answer": "E",
                "question_type": "单项选择题",
                "option": json.dumps(
                    {"A": "a", "B": "b", "C": "c", "D": "d", "E": "epsilon"}
                ),
            }
        ]
    )
    assert utils_cmb_exam.process_docs(ds)[0]["option_e"] == "epsilon"


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


def test_doc_to_text_renders_chinese_prompt(cmb_task):
    text = cmb_task.doc_to_text(_sample_single_choice_doc())
    assert "HIV 患者" in text
    assert "(A) 大叶性肺炎" in text
    assert "(D) 卡氏囊虫性肺炎" in text
    # No empty fifth option line.
    assert "(E)" not in text
    assert text.rstrip().endswith("答案：")


def test_doc_to_text_renders_fifth_option_when_present(cmb_task):
    doc = {**_sample_single_choice_doc(), "answer": "E", "option_e": "病毒性肺炎"}
    text = cmb_task.doc_to_text(doc)
    assert "(E) 病毒性肺炎" in text


def test_doc_to_target_passes_through_answer_letter(cmb_task):
    for letter in ("A", "B", "C", "D", "E"):
        doc = {**_sample_single_choice_doc(), "answer": letter, "option_e": "x"}
        assert cmb_task.doc_to_target(doc).strip() == letter


# ---------------------------------------------------------------------------
# Letter-extraction filter (regex + take_first), with A–E and 答案 anchor
# ---------------------------------------------------------------------------


def _build_extract_letter_filter(cmb_yaml: dict) -> FilterEnsemble:
    extract = next(f for f in cmb_yaml["filter_list"] if f["name"] == "extract_letter")
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
        ("E", "E"),
        ("答案：B", "B"),
        ("答案: C", "C"),
        ("The answer is D.", "D"),
        ("Therefore the answer is (E)", "E"),
        # Chinese reasoning preamble — last 答案 wins.
        ("分析选项 A 不对，B 也不行...经过分析，答案：C", "C"),
        # English anchor in a Chinese-prompted run (some models reply in en).
        ("I think... Actually Answer: E", "E"),
    ],
)
def test_filter_prefers_trailing_answer(cmb_yaml, completion, expected):
    filt = _build_extract_letter_filter(cmb_yaml)
    assert _apply(filt, completion) == expected


def test_filter_recognizes_e_option(cmb_yaml):
    """CMB-specific: 5-option questions require the regex to cover E."""
    filt = _build_extract_letter_filter(cmb_yaml)
    assert _apply(filt, "答案：E") == "E"


def test_filter_fallback_on_no_match(cmb_yaml):
    filt = _build_extract_letter_filter(cmb_yaml)
    assert _apply(filt, "不知道") == "[invalid]"


def test_filter_on_empty_completion(cmb_yaml):
    filt = _build_extract_letter_filter(cmb_yaml)
    assert _apply(filt, "") == "[invalid]"
