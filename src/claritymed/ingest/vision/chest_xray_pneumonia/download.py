"""Chest X-Ray Pneumonia dataset download — Kaggle CLI wrapper.

Pulls ``paultimothymooney/chest-xray-pneumonia`` into
``~/.claritymed/data/vision/chest_xray_pneumonia/``. Mirrors
``ingest/vision/lung_colon_histopath/download.py`` — the only
differences are the Kaggle slug and the on-disk subdir name.

Idempotent: re-runs skip when the target dir already holds the expected
top-level directory. Force a refetch with ``--force``.

The upstream archive (Mooney's repackage of Kermany et al. 2018) bundles
a top-level ``chest_xray/`` dir and a duplicate nested ``chest_xray/
chest_xray/`` copy — the unzip step keeps only the canonical top-level
tree so the bench / e2e see each image exactly once. Layout after
extraction::

    chest_xray_pneumonia/
    └── chest_xray/
        ├── train/{NORMAL,PNEUMONIA}/
        ├── test/{NORMAL,PNEUMONIA}/
        └── val/{NORMAL,PNEUMONIA}/
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

KAGGLE_SLUG = "paultimothymooney/chest-xray-pneumonia"
# Upstream zip unpacks to a top-level ``chest_xray/`` dir. Pinned here
# so the dataset loader can resolve the root without re-scanning the
# archive shape, matching the convention in lung_colon_histopath.
DATASET_SUBDIR = "chest_xray"
# Mooney's archive ships a redundant nested ``chest_xray/chest_xray/``
# tree mirroring the top-level one. Removing it after unzip keeps each
# image addressable by a single canonical path and halves the bench's
# sample-pool size.
_NESTED_DUPLICATE = "chest_xray/chest_xray"


def chest_xray_pneumonia_data_root() -> Path:
    """Where the chest-xray pneumonia dataset lives on disk. ``CLARITYMED_HOME`` aware."""
    return _cfg.CLARITYMED_HOME / "data" / "vision" / "chest_xray_pneumonia"


def is_present(root: Path | None = None) -> bool:
    """Return True iff the dataset has already been extracted."""
    target = (root or chest_xray_pneumonia_data_root()) / DATASET_SUBDIR
    return target.is_dir() and any(target.iterdir())


def download(root: Path | None = None, *, force: bool = False) -> Path:
    """Download + extract the dataset via the Kaggle CLI.

    Returns the path to the top-level archive directory.
    """
    target_root = root or chest_xray_pneumonia_data_root()
    target_root.mkdir(parents=True, exist_ok=True)
    if is_present(target_root) and not force:
        logger.info(
            "chest_xray_pneumonia already extracted at %s; skipping download",
            target_root,
        )
        return target_root / DATASET_SUBDIR

    if shutil.which("kaggle") is None:
        raise SystemExit(
            "kaggle CLI not found. Install via `uv pip install kaggle` and "
            "configure credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME + "
            "KAGGLE_KEY env vars)."
        )

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_SLUG,
        "-p",
        str(target_root),
        "--unzip",
    ]
    if force:
        cmd.append("--force")
    logger.info("running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"kaggle download failed (exit {exc.returncode}). "
            "Verify credentials and the dataset slug."
        ) from exc

    out = target_root / DATASET_SUBDIR
    if not out.is_dir():
        raise SystemExit(
            f"download completed but {out} not found; the upstream archive "
            f"shape may have changed. Inspect {target_root} manually."
        )

    nested = target_root / _NESTED_DUPLICATE
    if nested.is_dir():
        logger.info("removing redundant nested copy at %s", nested)
        shutil.rmtree(nested)
    return out


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Override target root")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = download(args.root, force=args.force)
    print(f"chest_xray_pneumonia extracted at: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main(sys.argv[1:]))
