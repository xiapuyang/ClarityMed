"""Sanity tests for ``forge_torch``'s two model factories.

Both ``build_classifier_model`` (cls-only) and ``build_cls_seg_model``
(U-Net) are called by BOTH the forge Task subclass and the runtime
adapter — a silent shape drift across backbones would break checkpoint
load. These tests pin the forward contract per backbone.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# ruff: noqa: E402 — module-level import lands after importorskip on
# purpose so the suite is skippable on boxes without the vision extra.
from claritymed.servers.vision.adapters.forge_torch import (
    build_classifier_model,
    build_cls_seg_model,
)


@pytest.mark.parametrize("backbone", ["custom_unet", "resnet50", "efficientnet_b0"])
def test_cls_seg_forward_pass_shapes_match_training_contract(backbone: str) -> None:
    """Every supported backbone returns ``(cls_logits, seg_logits)``."""
    model = build_cls_seg_model(backbone=backbone, num_classes=3, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        cls_logits, seg_logits = model(x)
    assert cls_logits.shape == (2, 3)
    assert seg_logits.shape == (2, 1, 256, 256)


def test_cls_seg_unknown_backbone_raises_with_known_names() -> None:
    with pytest.raises(ValueError) as exc:
        build_cls_seg_model(backbone="resnext101", num_classes=3, pretrained=False)
    msg = str(exc.value)
    assert "resnext101" in msg
    assert "custom_unet" in msg
    assert "resnet50" in msg


@pytest.mark.parametrize("backbone", ["resnet50", "efficientnet_b0", "efficientnet_b3"])
def test_classifier_forward_pass_shape_matches_training_contract(backbone: str) -> None:
    """Every supported classifier backbone returns ``cls_logits``."""
    model = build_classifier_model(backbone=backbone, num_classes=4, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        cls_logits = model(x)
    assert cls_logits.shape == (2, 4)


def test_classifier_unknown_backbone_raises_with_known_names() -> None:
    with pytest.raises(ValueError) as exc:
        build_classifier_model(backbone="vit_huge", num_classes=4, pretrained=False)
    msg = str(exc.value)
    assert "vit_huge" in msg
    assert "resnet50" in msg
    assert "efficientnet_b3" in msg


def test_built_models_expose_backbone_attribute() -> None:
    """The runtime adapter reads ``.backbone`` off the wrapper to roundtrip."""
    cls_model = build_classifier_model(
        backbone="resnet50", num_classes=4, pretrained=False
    )
    cls_seg_model = build_cls_seg_model(
        backbone="custom_unet", num_classes=3, pretrained=False
    )
    assert cls_model.backbone == "resnet50"
    assert cls_seg_model.backbone == "custom_unet"
