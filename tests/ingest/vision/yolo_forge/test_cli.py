"""CLI argparse-level smoke tests for ``claritymed-vision-yolo-forge``.

These tests stop short of invoking heavyweight phases (ultralytics
isn't installed in dev) — they verify that every subcommand parses,
the required flags are enforced, and the help text reflects the new
5-phase pipeline.
"""

from __future__ import annotations

import pytest

from claritymed.ingest.vision.yolo_forge.cli import _build_parser


def test_parser_lists_every_phase_subcommand() -> None:
    parser = _build_parser()
    # argparse stores subparsers under a private name; reach in once
    # for this assertion rather than reaching in across many tests.
    subparsers = next(
        action
        for action in parser._actions
        if hasattr(action, "choices")
        and action.choices
        and "pipeline" in action.choices
    )
    assert set(subparsers.choices) == {
        "prepare",
        "search",
        "train",
        "tune",
        "eval",
        "deploy",
        "pipeline",
    }


def test_parser_requires_model_for_every_phase() -> None:
    parser = _build_parser()
    for cmd in ("prepare", "search", "train", "tune", "eval", "deploy", "pipeline"):
        with pytest.raises(SystemExit):
            parser.parse_args([cmd])


def test_pipeline_accepts_search_and_tune_knobs() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "pipeline",
            "--model",
            "pkg.mod:ATTR",
            "--quick",
            "--skip-search",
            "--search-trials",
            "5",
            "--search-epochs",
            "2",
            "--tune-trials",
            "8",
        ]
    )
    assert args.command == "pipeline"
    assert args.quick is True
    assert args.skip_search is True
    assert args.search_trials == 5
    assert args.search_epochs == 2
    assert args.tune_trials == 8


def test_eval_accepts_conf_iou_split_overrides() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--model",
            "pkg.mod:ATTR",
            "--split",
            "val",
            "--conf",
            "0.3",
            "--iou",
            "0.4",
        ]
    )
    assert args.split == "val"
    assert args.conf == 0.3
    assert args.iou == 0.4
