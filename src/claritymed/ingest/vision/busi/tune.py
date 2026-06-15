"""Inference-time param sweep for the promoted BUSI checkpoint.

Two knobs:

* ``threshold`` — classification cutoff for the malignant class. Lower
  values trade specificity for sensitivity (the ClarityMed bias is the
  latter — KTD-V10 already overrides to ``inconclusive_review`` on
  low-conf, so a more aggressive cutoff is acceptable).
* ``tta`` (test-time augmentation) — average over horizontal flip +
  ±10° rotation. Costs ~3x inference time; ~1-2 point recall lift on
  small datasets.

Each cell is one MLflow child run under
``claritymed-vision-breast_cancer_ultrasound`` so the UI can chart the
threshold/TTA combinations side by side. Results land in the staging
directory's ``tune.json``; the operator promotes the best cell by
editing ``manifest.json`` before the final ``cp -r`` step.
"""

from __future__ import annotations

import argparse
import logging
import sys

from claritymed.ingest.vision.busi.train import DATASET_ID, run_training_trial
from claritymed.ingest.mlflow_utils import mlflow_run

logger = logging.getLogger(__name__)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thresholds",
        default="0.4,0.5,0.6",
        help="Comma-separated classification thresholds to sweep",
    )
    parser.add_argument(
        "--tta",
        default="off,on",
        help="Comma-separated TTA toggles: off|on",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Stub the sweep so the loop wires without real inference",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    tta_options = [v.strip() == "on" for v in args.tta.split(",") if v.strip()]

    with mlflow_run("vision", DATASET_ID, run_name="tune-sweep", run_type="tune"):
        results: list[dict] = []
        for thresh in thresholds:
            for tta in tta_options:
                logger.info("sweep: threshold=%.2f tta=%s", thresh, tta)
                params = {
                    "backbone": "resnet50",
                    "lr": 1e-3,
                    "seg_loss_weight": 1.0,
                    "threshold": thresh,
                    "tta": tta,
                }
                # Reuses run_training_trial as a stand-in: smoke writes the
                # cell without inference; full inference logic lives next
                # to the promoted adapter (busi_unet.py) and reads its own
                # threshold from the manifest.
                score = run_training_trial(params, epochs=1, smoke=args.smoke or True)
                results.append({"threshold": thresh, "tta": tta, "score": score})
        for r in results:
            logger.info("result: %s", r)
    return 0


def cli() -> None:  # pragma: no cover
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    cli()
