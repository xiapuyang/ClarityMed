"""RSNA Pneumonia Detection Challenge — Kaggle competitions download.

Unlike the regular Kaggle datasets we ingest elsewhere
(``paultimothymooney/chest-xray-pneumonia`` etc.), RSNA Pneumonia is a
**competition** archive, which requires:

* The Kaggle CLI installed and authenticated.
* The operator having accepted the competition rules at
  https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules
  before the download will succeed.

The competition CLI command is ``kaggle competitions download -c <slug>``
(not ``kaggle datasets download -d <slug>``). This module wraps it
with the same idempotent + force flags as the dataset-style modules.

Layout after extraction (the archive unpacks to a flat directory)::

    rsna_pneumonia/
    ├── stage_2_train_labels.csv          # patientId,x,y,width,height,Target
    ├── stage_2_detailed_class_info.csv   # patientId,class
    ├── stage_2_train_images/             # DICOMs, ~28k files
    │   ├── 0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm
    │   └── ...
    └── stage_2_test_images/              # held-out DICOMs (no labels)
        └── ...

We use ``stage_2_train_labels.csv`` + ``stage_2_train_images/`` for the
bench eval split — the official test split has no labels.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from claritymed import config as _cfg

logger = logging.getLogger(__name__)

KAGGLE_SLUG = "rsna-pneumonia-detection-challenge"

# Subdir name expected after extraction. The CLI unzips into the target
# root directly (no top-level wrapper dir like the regular Kaggle
# archives have), so we pin the train-images dir as the presence
# tripwire.
DATASET_SUBDIR = "."
_TRAIN_IMAGES_DIR = "stage_2_train_images"
_TRAIN_LABELS_CSV = "stage_2_train_labels.csv"


def rsna_pneumonia_data_root() -> Path:
    """Where the RSNA dataset lives on disk. ``CLARITYMED_HOME`` aware."""
    return _cfg.CLARITYMED_HOME / "data" / "vision" / "rsna_pneumonia"


def is_present(root: Path | None = None) -> bool:
    """Return True iff the dataset has been extracted with the expected layout."""
    target = root or rsna_pneumonia_data_root()
    return (target / _TRAIN_IMAGES_DIR).is_dir() and (
        target / _TRAIN_LABELS_CSV
    ).is_file()


def download(root: Path | None = None, *, force: bool = False) -> Path:
    """Download + extract the RSNA competition archive via the Kaggle CLI.

    Returns the path to the extraction root.
    """
    target_root = root or rsna_pneumonia_data_root()
    target_root.mkdir(parents=True, exist_ok=True)
    if is_present(target_root) and not force:
        logger.info(
            "rsna_pneumonia already extracted at %s; skipping download",
            target_root,
        )
        return target_root

    if shutil.which("kaggle") is None:
        raise SystemExit(
            "kaggle CLI not found. Install via `uv pip install kaggle` and "
            "configure credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME + "
            "KAGGLE_KEY env vars)."
        )

    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        KAGGLE_SLUG,
        "-p",
        str(target_root),
    ]
    if force:
        cmd.append("--force")
    logger.info("running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"kaggle competitions download failed (exit {exc.returncode}). "
            f"Verify credentials and that you've accepted the rules at "
            f"https://www.kaggle.com/competitions/{KAGGLE_SLUG}/rules"
        ) from exc

    # The competition CLI does not auto-unzip; archive lands as
    # <target_root>/<slug>.zip. Unpack it in-place.
    archive = target_root / f"{KAGGLE_SLUG}.zip"
    if archive.is_file():
        logger.info("unpacking %s", archive)
        shutil.unpack_archive(str(archive), str(target_root))
        archive.unlink()

    if not is_present(target_root):
        raise SystemExit(
            f"download completed but expected files not found at {target_root}. "
            f"Inspect manually — the upstream archive shape may have changed."
        )
    return target_root


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Override target root")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = download(args.root, force=args.force)
    print(f"rsna_pneumonia extracted at: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main(sys.argv[1:]))
