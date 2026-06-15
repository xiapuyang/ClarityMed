"""BUSI dataset download — Kaggle CLI wrapper.

Pulls ``aryashah2k/breast-ultrasound-images-dataset`` into
``~/.claritymed/data/vision/busi/``. The Kaggle CLI handles credentials
either via ``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME`` /
``KAGGLE_KEY`` env vars; document both in ``.env.example``.

Idempotent: re-runs skip the download when the target directory already
holds ``Dataset_BUSI_with_GT/``. Force a refetch with ``--force``.
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

KAGGLE_SLUG = "aryashah2k/breast-ultrasound-images-dataset"
DATASET_SUBDIR = "Dataset_BUSI_with_GT"


def busi_data_root() -> Path:
    """Where BUSI lives on disk. ``CLARITYMED_HOME`` aware."""
    return _cfg.CLARITYMED_HOME / "data" / "vision" / "busi"


def is_present(root: Path | None = None) -> bool:
    """Return True iff the BUSI dataset has already been extracted."""
    target = (root or busi_data_root()) / DATASET_SUBDIR
    return target.is_dir() and any(target.iterdir())


def download(root: Path | None = None, *, force: bool = False) -> Path:
    """Download + extract BUSI via the Kaggle CLI.

    Returns the path to ``Dataset_BUSI_with_GT/``.
    """
    target_root = root or busi_data_root()
    target_root.mkdir(parents=True, exist_ok=True)
    if is_present(target_root) and not force:
        logger.info("BUSI already extracted at %s; skipping download", target_root)
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
    return out


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Override target root")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = download(args.root, force=args.force)
    print(f"BUSI extracted at: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main(sys.argv[1:]))
