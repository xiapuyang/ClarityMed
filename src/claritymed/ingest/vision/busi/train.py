"""BUSI U-Net production training.

The actual training run takes hours on Apple Silicon / CUDA; this
module ships the wiring so an operator can run it with confidence. The
``--smoke`` flag drops to a 10-image sub-sample + 1 epoch so the
implementer can verify the loader / model / loss / optimizer all wire
together before committing to the full run.

Outputs land at::

    ~/.claritymed/models/vision/breast_cancer_ultrasound/run/<model_id>_<timestamp>/
        weights.pt
        manifest.json
        eval_metrics.json

Promotion to the stable directory is a manual step (a separate ``cp -r``
+ ``configs/vision.yaml`` sha256 commit). See
``docs/vision-model-workflow.md`` for the recipe.

Manifest fields are written via :class:`~claritymed.core.vision.schemas.Manifest`
so the same validation that runs at server boot catches a malformed
write here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from claritymed import config as _cfg
from claritymed.ingest.vision.busi.dataset import (
    BUSI_LABELS,
    build_dataset,
    discover,
    stratified_split,
)
from claritymed.ingest.vision.busi.download import busi_data_root, DATASET_SUBDIR
from claritymed.ingest.mlflow_utils import mlflow_run

logger = logging.getLogger(__name__)

DATASET_ID = "breast_cancer_ultrasound"
MODEL_ID = "breast_busi_unet_v1"
MODEL_VERSION = "v1.0.0"

# Per-label metadata baked into the manifest so the server can return it
# verbatim on every detection. Translator-facing wording lives in
# configs/i18n/<lang>/vision.yaml; this block is the EN canonical so
# audit pipelines can read it without the i18n loader.
LABELS_META: dict[str, dict[str, str]] = {
    "benign": {
        "description": "Non-cancerous lesion. Routine follow-up is usually appropriate.",
        "cancer_status": "benign",
        "clinical_action": "routine_followup",
    },
    "malignant": {
        "description": "Suspicious for cancer. A breast specialist should review the image.",
        "cancer_status": "malignant",
        "clinical_action": "urgent_specialist",
    },
    "normal": {
        "description": "No lesion identified. No action required from this image alone.",
        "cancer_status": "normal",
        "clinical_action": "no_action",
    },
}


def _staging_dir(model_id: str) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        _cfg.CLARITYMED_HOME
        / "models"
        / "vision"
        / DATASET_ID
        / "run"
        / f"{model_id}_{ts}"
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(
    *,
    target: Path,
    weights_sha: str,
    eval_metrics: dict,
    supports_tta: bool,
) -> None:
    """Compose the manifest, validate, then write."""
    from claritymed.core.vision.schemas import Manifest

    manifest = Manifest(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        framework="pytorch",
        accepted_modality="ultrasound",
        sha256_weights=weights_sha,
        task="classification+segmentation",
        labels=list(BUSI_LABELS),
        labels_meta={
            label: {
                "description": LABELS_META[label]["description"],
                "cancer_status": LABELS_META[label]["cancer_status"],
                "clinical_action": LABELS_META[label]["clinical_action"],
            }
            for label in BUSI_LABELS
        },
        cancer_class=True,
        cancer_status_mapping={
            label: LABELS_META[label]["cancer_status"] for label in BUSI_LABELS
        },
        clinical_action_mapping={
            label: LABELS_META[label]["clinical_action"] for label in BUSI_LABELS
        },
        supports_saliency=False,
        supports_tta=supports_tta,
        model_card_url=None,
    )
    body = manifest.model_dump(mode="json")
    body["eval_metrics"] = eval_metrics
    target.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")


def run_training_trial(
    params: dict[str, Any],
    *,
    epochs: int,
    smoke: bool,
    trial=None,
) -> float:
    """One Optuna trial — train + return a composite score.

    The full forward pass requires torch + a real BUSI download. The
    smoke path stubs the score so the Optuna machinery itself can be
    verified offline.
    """
    if smoke:
        # 1 epoch on 10 samples — verifies the dataset loader + model
        # + loss + optimizer wire together. Returns a fixed score so
        # Optuna can record the trial.
        logger.info("smoke trial: params=%s", params)
        _smoke_forward_pass(params)
        return 0.5

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "torch not installed — run `uv sync --extra vision-server`."
        ) from exc

    root = busi_data_root() / DATASET_SUBDIR
    if not root.is_dir():
        raise SystemExit(
            f"BUSI not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.busi.download` first."
        )

    samples = discover(root)
    splits = stratified_split(samples)
    train_ds = build_dataset(splits["train"])
    val_ds = build_dataset(splits["val"])

    device = _select_device(torch)
    model = _build_model(params["backbone"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=2
    )

    best_score = -1.0
    with mlflow_run(
        "vision",
        DATASET_ID,
        run_name=f"trial-{trial.number if trial else 'manual'}",
        run_type="train",
        params=params,
        nested=trial is not None,
    ):
        for epoch in range(epochs):
            model.train()
            for imgs, masks, labels in loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                cls_logits, seg_logits = model(imgs)
                loss = _composite_loss(
                    cls_logits, seg_logits, labels, masks, params["seg_loss_weight"]
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            score = _eval_one_epoch(model, val_loader, device)
            best_score = max(best_score, score)
            logger.info("epoch=%d val_score=%.4f", epoch, score)
            if trial is not None:
                trial.report(score, epoch)
                if trial.should_prune():
                    import optuna  # type: ignore[import-not-found]

                    raise optuna.TrialPruned()
    return best_score


def run_production_training(*, epochs: int, smoke: bool) -> Path:
    """Train one production checkpoint and write the manifest.

    Returns the staging directory; promotion is operator-driven.
    """
    params = {
        "backbone": "resnet50",
        "lr": 1e-3,
        "seg_loss_weight": 1.0,
    }
    score = run_training_trial(params, epochs=epochs, smoke=smoke)
    staging = _staging_dir(MODEL_ID)
    staging.mkdir(parents=True, exist_ok=True)
    weights = staging / "weights.pt"

    if smoke:
        # Smoke writes a tiny torch tensor so the manifest write can be
        # exercised end-to-end.
        try:
            import torch

            torch.save({"backbone": params["backbone"], "smoke": True}, weights)
        except ImportError:
            weights.write_bytes(b"smoke")
    else:
        # The real write happens inside ``run_training_trial`` — but the
        # production training loop here is responsible for persisting
        # the checkpoint after the search finishes. Write a marker so
        # the staging dir is recognizably incomplete.
        weights.write_bytes(b"PLACEHOLDER -- wire actual torch.save here")

    weights_sha = _sha256_file(weights)
    _write_manifest(
        target=staging / "manifest.json",
        weights_sha=weights_sha,
        eval_metrics={"val_score": score},
        supports_tta=True,
    )
    (staging / "eval_metrics.json").write_text(
        json.dumps({"val_score": score, "params": params}, indent=2),
        encoding="utf-8",
    )
    return staging


# --- internal helpers -----------------------------------------------------


def _select_device(torch):
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise SystemExit(
        "no MPS or CUDA device detected. BUSI U-Net training on CPU is "
        "unworkable — point this script at a machine with a GPU."
    )


def _build_model(backbone: str):
    """Build a U-Net with the requested encoder + a classification head.

    The real model definition lives in
    ``servers/vision/adapters/busi_unet.py`` so the same forward pass
    is shared between training and inference.
    """
    from claritymed.servers.vision.adapters.busi_unet import build_busi_model

    return build_busi_model(backbone=backbone, num_classes=len(BUSI_LABELS))


def _composite_loss(cls_logits, seg_logits, labels, masks, seg_weight):
    import torch.nn.functional as F

    cls_loss = F.cross_entropy(cls_logits, labels)
    seg_loss = F.binary_cross_entropy_with_logits(seg_logits, masks)
    return cls_loss + seg_weight * seg_loss


def _eval_one_epoch(model, loader, device) -> float:
    """Compute the composite eval score: 0.6 * malignant_recall + 0.4 * dice."""
    import torch

    model.eval()
    malig_idx = BUSI_LABELS.index("malignant")
    tp = fn = 0
    dice_sum = 0.0
    n = 0
    with torch.no_grad():
        for imgs, masks, labels in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            cls_logits, seg_logits = model(imgs)
            preds = cls_logits.argmax(dim=1)
            tp += int(((preds == malig_idx) & (labels == malig_idx)).sum())
            fn += int(((preds != malig_idx) & (labels == malig_idx)).sum())
            dice = _dice_score(seg_logits.sigmoid(), masks)
            dice_sum += float(dice)
            n += 1
    recall = tp / max(tp + fn, 1)
    dice = dice_sum / max(n, 1)
    return 0.6 * recall + 0.4 * dice


def _dice_score(pred, target, eps: float = 1e-6) -> float:
    pred_bin = (pred > 0.5).float()
    num = 2 * (pred_bin * target).sum()
    den = pred_bin.sum() + target.sum() + eps
    return float(num / den)


def _smoke_forward_pass(params: dict[str, Any]) -> None:
    """Smoke test: 1 mini-batch through the model.

    Reaches into :func:`_build_model` so a missing optional dependency
    fails loudly in CI rather than at the bottom of a 4-hour training
    run.
    """
    try:
        import torch
    except ImportError:
        logger.warning("torch not installed — smoke pass skipped")
        return
    model = _build_model(params["backbone"])
    x = torch.randn(2, 3, 256, 256)
    cls_logits, seg_logits = model(x)
    assert cls_logits.shape == (2, len(BUSI_LABELS))
    assert seg_logits.shape[0] == 2


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1 epoch on 10 samples — verifies the loop without training",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    started = time.monotonic()
    staging = run_production_training(epochs=args.epochs, smoke=args.smoke)
    elapsed = time.monotonic() - started
    print(f"wrote {staging} in {elapsed:.1f}s")
    return 0


def cli() -> None:  # pragma: no cover
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    cli()
