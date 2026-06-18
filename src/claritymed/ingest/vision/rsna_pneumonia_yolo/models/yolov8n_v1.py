"""YOLOv8-nano on RSNA Pneumonia — first detection ModelSpec.

YOLOv8n is the smallest Ultralytics variant (~3.2M params); chosen as
the entry point because:

* Pretrained ``yolov8n.pt`` fits in <10MB so Ultralytics' auto-download
  is fast on a fresh box.
* Single Apple-silicon GPU can train it in <1 hour on the full RSNA
  train split at imgsz=640.
* If mAP plateaus too low we step up to ``yolov8s``/``yolov8m`` as
  separate ``_v2.py`` ModelSpecs rather than mutating this one.

Eval thresholds picked conservatively for a first cut. ``image_recall``
floor is intentionally aggressive (0.80): a missed positive is the
clinical-harm failure mode, and any deploy that regresses recall
below the floor should fail the gate rather than ship.

Bump ``model_version`` (and reflect that in the filename) when any
hparam changes — the gate's regression check compares apples to apples
against the last entry for the same ``model_id`` in ``LATEST.jsonl``.
"""

from __future__ import annotations

from claritymed.ingest.vision.rsna_pneumonia_yolo.dataset_spec import (
    RSNA_PNEUMONIA_YOLO_DATASET,
)
from claritymed.ingest.vision.yolo_forge.spec import (
    LogUniform,
    Uniform,
    YoloModelSpec,
    YoloTrainHparams,
)

RSNA_YOLOV8N_V1 = YoloModelSpec(
    dataset=RSNA_PNEUMONIA_YOLO_DATASET,
    model_id="rsna_pneumonia_yolov8n_v1",
    model_version="v1",
    base_weights="yolov8n.pt",
    train_hparams=YoloTrainHparams(
        epochs=30,
        # ``imgsz=512`` (down from yolov8 default 640) — chest
        # pneumonia regions are large relative to image, the mAP
        # cost is <1 point, and the ~35% augmentation/forward
        # speedup matters on a macOS dev box where ``workers=0``
        # is forced and dataloader can't overlap GPU.
        imgsz=512,
        batch=16,
        # ``lr0=0.01`` (yolov8 default) — paired with
        # ``warmup_epochs=3`` below. Earlier revision used
        # ``0.005 + warmup_epochs=1`` after a prior run with
        # ``0.01 + warmup=3`` blew up the val head in epoch 2; the
        # conservative schedule capped the curve at mAP50 ~0.33
        # with overfit by epoch 7, so we're walking back to the
        # default schedule to see if the current spec state (disk
        # cache, mosaic=1, mixup=0, single-class) tolerates it.
        # **Fail-fast signal**: if val/box_loss exceeds 5 in epoch
        # 2-3, revert to ``lr0=0.005`` + ``warmup_epochs=1``.
        lr0=0.01,
        # Slightly more patience than the default — single-class
        # medical detection benefits from longer training plateaus.
        patience=10,
        # Three-epoch warmup at default ``warmup_bias_lr=0.1``.
        # Previous revision shrank this to 1 epoch to compensate
        # for the conservative ``lr0=0.005``; now that ``lr0`` is
        # back at the YOLOv8 default we restore the default warmup
        # to give the pretrained head enough steps to align BN
        # stats and bias before full LR kicks in.
        warmup_epochs=3.0,
        # Keep mosaic on but disable mixup; mixup makes per-pixel
        # bbox truth ambiguous on near-uniform x-ray backgrounds.
        mosaic=1.0,
        mixup=0.0,
        # ``"disk"`` — pre-decode the train set into .npy files
        # next to the JPEGs so every epoch skips JPEG decode and
        # the macOS ``workers=0`` dataloader stall vanishes.
        # Bonus: deterministic, unlike ``"ram"``. Requires ~82GB
        # free disk (1.5× the decoded ~55GB at original RSNA
        # resolution); ensure headroom before launching.
        cache="disk",
    ),
    # Search-phase HPO ranges. Five continuous knobs covering the SGD
    # optimizer triple (``lr0``/``lrf``/``momentum``), regularisation
    # (``weight_decay``), and the detection-side loss balance (``box``).
    # ``box`` matters because the fitness formula is 0.1·mAP50 +
    # 0.9·mAP50-95 — mAP50-95 rewards tight localisation, which the box
    # weight directly controls. ``cls`` is excluded (single-class makes
    # cls-vs-objectness balance low-signal); ``mosaic`` is excluded as a
    # binary toggle — better run as a 2-row ablation than as one Optuna
    # dimension.
    hparam_space={
        "lr0": LogUniform(1e-4, 5e-2),
        # Lower bound raised from 0.001 → 0.01. At ``lrf=0.001`` the
        # end-of-training LR is ``lr0 × 0.001 = 1e-5`` (or 5e-6 at the
        # previous ``lr0=0.005``) — too small to escape a noisy plateau
        # in the back half of training. 0.01–0.1 keeps the cosine
        # schedule's tail with real teeth.
        "lrf": Uniform(0.01, 0.1),
        "momentum": Uniform(0.85, 0.99),
        "weight_decay": LogUniform(1e-6, 1e-2),
        "box": Uniform(5.0, 10.0),
    },
    # Tune-phase inference knobs. ``conf`` dominates the recall/precision
    # tradeoff at the binary level; ``iou`` only matters when multiple
    # bboxes per image overlap, which is rare on RSNA single-region cases.
    # ``conf`` lower bound raised to 0.25 — below that the per-image
    # positive-region count balloons and image-level precision tanks
    # without a matching recall gain on this dataset.
    inference_space={
        "conf": Uniform(0.25, 0.5),
        "iou": Uniform(0.3, 0.7),
    },
    eval_thresholds={
        "image_recall": 0.80,
        "mAP50": 0.30,
    },
)


__all__ = ["RSNA_YOLOV8N_V1"]
