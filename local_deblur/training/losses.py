"""Lightweight restoration losses for smoke and trainable paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from local_deblur.data.transforms import image_to_array, mask_to_array


def masked_l1(prediction, target, mask) -> float:
    pred = image_to_array(prediction)
    tgt = image_to_array(target)
    m = mask_to_array(mask)[..., None]
    denom = float(m.sum() * pred.shape[-1]) + 1e-8
    return float((np.abs(pred - tgt) * m).sum() / denom)


def charbonnier_loss(prediction, target, mask, epsilon: float = 1e-3) -> float:
    pred = image_to_array(prediction)
    tgt = image_to_array(target)
    m = mask_to_array(mask)[..., None]
    value = np.sqrt((pred - tgt) ** 2 + epsilon**2) * m
    return float(value.sum() / (m.sum() * pred.shape[-1] + 1e-8))


def ssim_loss_placeholder(prediction, target, mask=None) -> float:
    """Optional placeholder: returns 1 - SSIM if skimage is available."""
    try:
        from skimage.metrics import structural_similarity

        pred = image_to_array(prediction)
        tgt = image_to_array(target)
        return float(1.0 - structural_similarity(tgt, pred, channel_axis=-1, data_range=1.0))
    except Exception:
        return 0.0


def binary_cross_entropy_mask(predicted_mask, target_mask, epsilon: float = 1e-6) -> float:
    pred = np.clip(mask_to_array(predicted_mask), epsilon, 1.0 - epsilon)
    target = mask_to_array(target_mask)
    loss = -(target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred))
    return float(loss.mean())


def torch_masked_l1(prediction: Any, target: Any, mask: Any | None = None, epsilon: float = 1e-8):
    """Masked L1 for BCHW tensors, normalized over supervised pixels."""
    diff = (prediction - target).abs()
    if mask is None:
        return diff.mean()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    denom = mask.sum() * prediction.shape[1] + epsilon
    return (diff * mask).sum() / denom


def torch_charbonnier_loss(prediction: Any, target: Any, mask: Any | None = None, epsilon: float = 1e-3):
    value = ((prediction - target) ** 2 + epsilon**2).sqrt()
    if mask is None:
        return value.mean()
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    denom = mask.sum() * prediction.shape[1] + 1e-8
    return (value * mask).sum() / denom


def restoration_loss(
    prediction: Any,
    target: Any,
    mask: Any | None = None,
    *,
    l1_weight: float = 0.8,
    charbonnier_weight: float = 0.2,
) -> dict[str, Any]:
    """Image restoration loss for the trainable PyTorch baseline."""
    l1 = torch_masked_l1(prediction, target, mask)
    charb = torch_charbonnier_loss(prediction, target, mask)
    total = l1_weight * l1 + charbonnier_weight * charb
    return {"loss": total, "l1": l1, "charbonnier": charb}


def dice_loss_from_logits(mask_logits: Any, target_mask: Any, epsilon: float = 1e-6):
    import torch

    if target_mask.ndim == 3:
        target_mask = target_mask.unsqueeze(1)
    probs = torch.sigmoid(mask_logits)
    intersection = (probs * target_mask).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return 1.0 - dice.mean()


def mask_prediction_loss(
    mask_logits: Any,
    target_mask: Any,
    *,
    bce_weight: float = 1.0,
    dice_weight: float = 0.5,
) -> dict[str, Any]:
    """BCE-with-logits plus Dice-compatible loss for blur mask supervision."""
    import torch.nn.functional as F

    if target_mask.ndim == 3:
        target_mask = target_mask.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
    dice = dice_loss_from_logits(mask_logits, target_mask)
    total = bce_weight * bce + dice_weight * dice
    return {"loss": total, "bce": bce, "dice": dice}


def combined_trainable_loss(
    restored: Any,
    target: Any,
    mask_logits: Any,
    target_mask: Any,
    *,
    image_weight: float = 1.0,
    mask_weight: float = 0.1,
    restoration_l1_weight: float = 0.8,
    restoration_charbonnier_weight: float = 0.2,
    mask_bce_weight: float = 1.0,
    mask_dice_weight: float = 0.5,
) -> dict[str, Any]:
    """Combined restoration and mask-head objective for ConditionalLocalDeblurNet."""
    image_terms = restoration_loss(
        restored,
        target,
        target_mask,
        l1_weight=restoration_l1_weight,
        charbonnier_weight=restoration_charbonnier_weight,
    )
    mask_terms = mask_prediction_loss(mask_logits, target_mask, bce_weight=mask_bce_weight, dice_weight=mask_dice_weight)
    total = image_weight * image_terms["loss"] + mask_weight * mask_terms["loss"]
    return {
        "loss": total,
        "restoration_loss": image_terms["loss"],
        "mask_loss": mask_terms["loss"],
        "masked_l1": image_terms["l1"],
        "charbonnier": image_terms["charbonnier"],
        "mask_bce": mask_terms["bce"],
        "mask_dice": mask_terms["dice"],
    }
