"""Deploy a tuned BUSI checkpoint to the versioned stable directory.

Reads a tune-finalised staging dir, gates promotion on:

1. **Floors** — the held-out test breakdown must clear medically-real
   thresholds (malignant recall ≥ 0.85, accuracy ≥ 0.85, dice ≥ 0.70).
   These are pass/fail; failing any one aborts the deploy.
2. **Regression gate** — the tuned **test composite** must beat the
   previously deployed model's test composite (recorded in
   ``LATEST.jsonl``). First-ever deploy skips this gate.

On pass:

* Copy ``staging/`` → ``<disease_root>/<MODEL_ID>__<version_tag>/``.
  ``version_tag`` is a UTC timestamp; the **destination directory must
  not exist** (refuse-to-overwrite is enforced at the OS level so a
  double-deploy can't silently clobber a prior promotion).
* Re-hash the (now stable-path) ``manifest.json`` and surgically edit
  ``configs/vision.yaml::models[i]`` for the matching ``model_id``,
  updating ``weights_subpath`` + ``manifest_sha256`` in place. The
  file's hand-written comments + ordering are preserved because the
  edit is line-based; the result is re-parsed via PyYAML to confirm it
  still validates against the schema.
* Append one JSONL row to ``<disease_root>/LATEST.jsonl`` carrying:
  version tag, deployed_at, weights_subpath, manifest path + sha,
  MLflow tracking URI + experiment + run ids, Optuna storage + study
  names, the best HP, the best tuned inference params, the metric
  breakdown, the previous-deploy delta. One file = the full audit
  trail; ``tail -n1 | jq`` shows the active deploy at a glance.

On fail: a clear ``SystemExit`` with the offending floor / regression
delta. No filesystem changes, no YAML edit, no LATEST.jsonl row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from claritymed import config as _cfg
from claritymed.core.vision.schemas import Manifest, VisionConfig
from claritymed.ingest.vision.busi.train import DATASET_ID, MODEL_ID
from claritymed.ingest.vision.busi.tune import _latest_staging_dir

logger = logging.getLogger(__name__)

# Medically-real floors. Failing any one of these means the model is
# not safe to ship — the deploy step refuses regardless of how much
# the candidate beats the active model on composite.
FLOOR_MALIGNANT_RECALL = 0.85
FLOOR_ACCURACY = 0.85
FLOOR_DICE = 0.70


def run_deploy(*, staging_dir: Path, smoke: bool = False) -> Path:
    """Promote ``staging_dir`` to a versioned stable path.

    Returns the new stable path on success. Raises ``SystemExit`` on
    floor or regression failure (no filesystem changes made).
    """
    manifest_path = staging_dir / "manifest.json"
    eval_path = staging_dir / "eval_metrics.json"
    provenance_path = staging_dir / "provenance.json"
    for path in (manifest_path, eval_path, provenance_path):
        if not path.exists():
            raise SystemExit(f"{path} missing — staging dir is incomplete")

    manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = Manifest.model_validate(manifest_dict)
    if manifest.tuned_inference is None:
        raise SystemExit(
            f"{manifest_path} has tuned_inference=None — run tune.py first."
        )

    eval_metrics = json.loads(eval_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    tuned_test = eval_metrics.get("tuned_test_breakdown")
    if not tuned_test:
        raise SystemExit(
            f"{eval_path} missing tuned_test_breakdown — run tune.py first."
        )

    # Step 1: floors.
    _check_floors(tuned_test)

    # Step 2: regression gate vs active deploy (if any).
    latest_path = _latest_jsonl_path()
    previous = _read_latest_entry(latest_path)
    if previous is not None:
        prev_score = float(previous["metrics"]["tuned_test_composite"])
        candidate_score = float(tuned_test["composite"])
        if candidate_score <= prev_score:
            raise SystemExit(
                f"regression gate failed: candidate tuned_test_composite="
                f"{candidate_score:.4f} ≤ active {prev_score:.4f} "
                f"(version {previous['version_tag']}). Re-tune or retrain "
                f"before retrying."
            )
        logger.info(
            "regression gate ok: %.4f > active %.4f (Δ=%+.4f)",
            candidate_score,
            prev_score,
            candidate_score - prev_score,
        )
    else:
        logger.info("first deploy — regression gate skipped")

    # Step 3: versioned sibling promotion. Refuse to overwrite.
    version_tag = _version_tag()
    stable_dirname = f"{MODEL_ID}__{version_tag}"
    disease_root = _cfg.CLARITYMED_HOME / "models" / "vision" / DATASET_ID
    stable_path = disease_root / stable_dirname
    if stable_path.exists():
        raise SystemExit(
            f"refusing to overwrite {stable_path} — pick a different "
            f"version_tag or remove the existing dir manually."
        )

    logger.info("promoting %s → %s", staging_dir, stable_path)
    shutil.copytree(staging_dir, stable_path)

    # Step 4: re-hash the promoted manifest and edit configs/vision.yaml.
    new_manifest_sha = _sha256_file(stable_path / "manifest.json")
    new_weights_subpath = f"vision/{DATASET_ID}/{stable_dirname}"

    if not smoke:
        _patch_vision_yaml(
            model_id=MODEL_ID,
            new_weights_subpath=new_weights_subpath,
            new_manifest_sha=new_manifest_sha,
        )

    # Step 5: LATEST.jsonl append with full provenance.
    entry = _build_latest_entry(
        version_tag=version_tag,
        weights_subpath=new_weights_subpath,
        manifest_path=stable_path / "manifest.json",
        manifest_sha=new_manifest_sha,
        provenance=provenance,
        eval_metrics=eval_metrics,
        tuned_test=tuned_test,
        previous_score=(
            previous["metrics"]["tuned_test_composite"] if previous else None
        ),
    )
    _append_latest_entry(latest_path, entry)
    logger.info("appended LATEST.jsonl entry for version %s", version_tag)
    return stable_path


# --- floor + regression helpers -----------------------------------------


def _check_floors(tuned_test: dict[str, float]) -> None:
    """Raise SystemExit when any floor fails."""
    fails: list[str] = []
    if tuned_test.get("malignant_recall", 0.0) < FLOOR_MALIGNANT_RECALL:
        fails.append(
            f"malignant_recall={tuned_test.get('malignant_recall'):.3f} "
            f"< floor {FLOOR_MALIGNANT_RECALL}"
        )
    if tuned_test.get("accuracy", 0.0) < FLOOR_ACCURACY:
        fails.append(
            f"accuracy={tuned_test.get('accuracy'):.3f} < floor {FLOOR_ACCURACY}"
        )
    if tuned_test.get("dice", 0.0) < FLOOR_DICE:
        fails.append(f"dice={tuned_test.get('dice'):.3f} < floor {FLOOR_DICE}")
    if fails:
        raise SystemExit("floor gate failed: " + "; ".join(fails))


# --- LATEST.jsonl helpers ------------------------------------------------


def _latest_jsonl_path() -> Path:
    """Return the LATEST.jsonl path for the BUSI disease root."""
    return _cfg.CLARITYMED_HOME / "models" / "vision" / DATASET_ID / "LATEST.jsonl"


def _read_latest_entry(path: Path) -> dict[str, Any] | None:
    """Return the last non-empty JSON object in ``LATEST.jsonl``, or None."""
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    return last


def _append_latest_entry(path: Path, entry: dict[str, Any]) -> None:
    """Append one JSON line to ``path``, creating the file if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _build_latest_entry(
    *,
    version_tag: str,
    weights_subpath: str,
    manifest_path: Path,
    manifest_sha: str,
    provenance: dict[str, Any],
    eval_metrics: dict[str, Any],
    tuned_test: dict[str, float],
    previous_score: float | None,
) -> dict[str, Any]:
    """Compose the JSONL row carrying the full deploy audit trail.

    The lineage ``task_id`` lives at the top level so a viewer can grep
    one row to find the active deploy's lineage and then pivot into
    MLflow (filter by `tags.claritymed.task_id`) or Optuna (filter by
    `user_attrs.task_id`). ``search_trial_number`` +
    ``search_trial_task_id`` close the gap "which Optuna trial fed
    train this HP?" — search-time lineage is preserved even when train
    pulls HPs from a different pipeline run's trials.
    """
    candidate_score = float(tuned_test["composite"])
    delta = (
        candidate_score - float(previous_score) if previous_score is not None else None
    )
    return {
        "version_tag": version_tag,
        "deployed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task_id": provenance.get("task_id"),
        "model_id": MODEL_ID,
        "disease_id": DATASET_ID,
        "weights_subpath": weights_subpath,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "mlflow": {
            "tracking_uri": provenance.get("mlflow", {}).get("tracking_uri"),
            "experiment_name": provenance.get("mlflow", {}).get("experiment_name"),
            "train_run_id": provenance.get("mlflow", {}).get("train_run_id"),
            "tune_run_id": provenance.get("tune", {})
            .get("mlflow", {})
            .get("tune_run_id"),
        },
        "optuna": {
            "storage_uri": provenance.get("optuna", {}).get("storage_uri"),
            "search_study_name": provenance.get("optuna", {}).get("search_study_name"),
            "search_trial_number": provenance.get("optuna", {}).get(
                "search_trial_number"
            ),
            "search_trial_task_id": provenance.get("optuna", {}).get(
                "search_trial_task_id"
            ),
            "tune_study_name": provenance.get("tune", {})
            .get("optuna", {})
            .get("tune_study_name"),
        },
        "best_hp": provenance.get("params"),
        "best_inference_params": provenance.get("tune", {}).get("params"),
        "metrics": {
            "best_val_score": eval_metrics.get("best_val_score"),
            "test_score": eval_metrics.get("test_score"),
            "tuned_val_breakdown": eval_metrics.get("tuned_val_breakdown"),
            "tuned_test_breakdown": tuned_test,
            "tuned_test_composite": candidate_score,
            "best_epoch": eval_metrics.get("best_epoch"),
            "early_stopped": eval_metrics.get("early_stopped"),
            "epochs_trained": eval_metrics.get("epochs_trained"),
        },
        "previous_tuned_test_composite": previous_score,
        "delta": delta,
        "floors_passed": {
            "malignant_recall": tuned_test["malignant_recall"]
            >= FLOOR_MALIGNANT_RECALL,
            "accuracy": tuned_test["accuracy"] >= FLOOR_ACCURACY,
            "dice": tuned_test["dice"] >= FLOOR_DICE,
        },
    }


# --- configs/vision.yaml surgical patch ----------------------------------


def _patch_vision_yaml(
    *, model_id: str, new_weights_subpath: str, new_manifest_sha: str
) -> None:
    """Edit ``configs/vision.yaml::models[i]`` for ``model_id`` in place.

    Line-based replacement preserves comments and ordering — PyYAML's
    roundtrip would otherwise strip the heavy commentary that vision.yaml
    leans on for operator-facing context.

    Safety:

    1. Parse the original file with PyYAML and confirm the targeted
       ``model_id`` exists.
    2. Apply the line-level edit (only the ``weights_subpath`` +
       ``manifest_sha256`` lines under the matched ``- id:`` entry).
    3. Re-parse the modified text via :class:`VisionConfig` so a malformed
       result aborts before the file is committed.
    4. Atomic write through a sibling tmp + rename.
    """
    yaml_path = _vision_yaml_path()
    original = yaml_path.read_text(encoding="utf-8")

    parsed = yaml.safe_load(original) or {}
    models = parsed.get("models", [])
    if not any(m.get("id") == model_id for m in models):
        raise SystemExit(
            f"configs/vision.yaml has no model entry with id={model_id!r}; "
            f"deploy aborted."
        )

    edited = _replace_model_fields(
        original,
        model_id=model_id,
        new_weights_subpath=new_weights_subpath,
        new_manifest_sha=new_manifest_sha,
    )

    # Re-parse to confirm the file still validates against the schema.
    re_parsed = yaml.safe_load(edited)
    VisionConfig.model_validate(re_parsed)

    tmp_path = yaml_path.with_suffix(yaml_path.suffix + ".tmp")
    tmp_path.write_text(edited, encoding="utf-8")
    tmp_path.replace(yaml_path)


def _vision_yaml_path() -> Path:
    """Return the path to ``configs/vision.yaml`` next to the package source."""
    return _cfg.PROJECT_ROOT / "configs" / "vision.yaml"


def _replace_model_fields(
    text: str,
    *,
    model_id: str,
    new_weights_subpath: str,
    new_manifest_sha: str,
) -> str:
    """Surgical line-based replace of two fields inside the matching model entry."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []

    # Scan to the matching `- id: <model_id>` entry. The block ends at
    # the next sibling `- id:` line or at a top-level key.
    in_target_block = False
    block_indent = 0
    replaced_weights = False
    replaced_sha = False

    id_pattern = re.compile(r"^(\s*-\s*id:\s*)([A-Za-z0-9_\-]+)\s*$")

    for line in lines:
        match = id_pattern.match(line)
        if match:
            entry_id = match.group(2)
            if entry_id == model_id:
                in_target_block = True
                block_indent = len(match.group(1)) - len("- id: ")
            elif in_target_block:
                # Hit the next sibling entry — we're done.
                in_target_block = False
            out.append(line)
            continue
        if in_target_block:
            stripped = line.lstrip(" ")
            indent = len(line) - len(stripped)
            # Top-level key (zero indent) ends the models[] block.
            if stripped.startswith(
                ("servers:", "models:", "tool:", "ocr_report:", "diseases:")
            ):
                in_target_block = False
            elif indent > block_indent:
                if stripped.startswith("weights_subpath:"):
                    line = " " * indent + f"weights_subpath: {new_weights_subpath}\n"
                    replaced_weights = True
                elif stripped.startswith("manifest_sha256:"):
                    line = " " * indent + f'manifest_sha256: "{new_manifest_sha}"\n'
                    replaced_sha = True
        out.append(line)

    if not replaced_weights:
        raise SystemExit(
            f"configs/vision.yaml: could not find weights_subpath line under "
            f"model id={model_id!r}"
        )
    if not replaced_sha:
        raise SystemExit(
            f"configs/vision.yaml: could not find manifest_sha256 line under "
            f"model id={model_id!r}"
        )
    return "".join(out)


# --- misc helpers --------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _version_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# --- CLI -----------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Path to the tune-finalised staging dir. Defaults to the most recent.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Skip the configs/vision.yaml patch (useful for offline smoke runs).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    staging = args.staging_dir or _latest_staging_dir()
    stable = run_deploy(staging_dir=staging, smoke=args.smoke)
    print(f"deployed {stable}")
    return 0


def cli() -> None:  # pragma: no cover
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    cli()
