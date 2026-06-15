"""Print the sha256 of a promoted manifest + the weights file it pins.

Operator helper for Step 5 of ``docs/vision-model-workflow.md`` — the
output is the value you paste into
``configs/vision.yaml::models[i].manifest_sha256``.

Usage::

    uv run python scripts/print_manifest_sha.py vision/breast_cancer_ultrasound/breast_busi_unet_v1

The path is relative to ``CLARITYMED_HOME/models/``. Prints two
digests (manifest + weights) so the operator can verify the
two-level chain (KTD-V7) ahead of restarting the server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from claritymed import config as _cfg


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "subpath", help="Path under CLARITYMED_HOME/models/ to the promoted dir"
    )
    args = parser.parse_args(argv)
    root = _cfg.CLARITYMED_HOME / "models" / args.subpath
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    manifest_sha = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = root / "weights.pt"
    if weights_path.exists():
        weights_sha = _sha256(weights_path)
    else:
        weights_sha = "<missing>"
    print(f"manifest.json sha256: {manifest_sha}")
    print(f"weights.pt sha256:    {weights_sha}")
    print(f"manifest declares:    {manifest.get('sha256_weights', '<missing>')}")
    if weights_sha != "<missing>" and weights_sha != manifest.get("sha256_weights", ""):
        print(
            "WARNING: weights sha mismatch — manifest will be rejected at startup",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
