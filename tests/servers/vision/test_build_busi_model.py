"""Sanity tests for ``build_busi_model``'s backbone branches.

The model is wired into three places (train.py, tune.py, the
BUSIUnetAdapter) and silently swapping the architecture would break
checkpoint shape compat — so the forward contract is asserted here
explicitly per backbone.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# ruff: noqa: E402 — module-level import lands after importorskip on
# purpose so the suite is skippable on boxes without the vision extra.
from claritymed.servers.vision.adapters.busi_unet import build_busi_model


@pytest.mark.parametrize("backbone", ["custom_unet", "resnet50", "efficientnet_b0"])
def test_forward_pass_shapes_match_training_contract(backbone: str) -> None:
    """Every supported backbone must return ``(cls_logits, seg_logits)`` at
    the shapes train.py + the adapter assume."""
    model = build_busi_model(backbone=backbone, num_classes=3, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        cls_logits, seg_logits = model(x)
    assert cls_logits.shape == (2, 3)
    assert seg_logits.shape == (2, 1, 256, 256)


def test_unknown_backbone_raises_with_known_names() -> None:
    """Misspelled backbones must fail loud — silent fallback to
    custom_unet would mask Optuna or manifest drift."""
    with pytest.raises(ValueError) as exc:
        build_busi_model(backbone="resnext101", num_classes=3, pretrained=False)
    msg = str(exc.value)
    assert "resnext101" in msg
    assert "custom_unet" in msg
    assert "resnet50" in msg
