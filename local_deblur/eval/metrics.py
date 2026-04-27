"""PSNR/SSIM metrics for local deblurring evaluation."""

from __future__ import annotations

import math

import numpy as np

from local_deblur.data.transforms import image_to_array, mask_to_array


def psnr(prediction, target, mask=None) -> float:
    pred = image_to_array(prediction)
    tgt = image_to_array(target)
    if mask is not None:
        m = mask_to_array(mask)[..., None]
        mse = float((((pred - tgt) ** 2) * m).sum() / (m.sum() * pred.shape[-1] + 1e-8))
    else:
        mse = float(np.mean((pred - tgt) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def ssim(prediction, target, mask=None) -> float:
    pred = image_to_array(prediction)
    tgt = image_to_array(target)
    try:
        from skimage.metrics import structural_similarity

        if mask is None:
            return float(structural_similarity(tgt, pred, channel_axis=-1, data_range=1.0))
    except Exception:
        pass

    c1 = 0.01**2
    c2 = 0.03**2
    if mask is None:
        weights = np.ones(pred.shape[:2] + (1,), dtype=np.float32)
    else:
        weights = mask_to_array(mask)[..., None]
    denom = float(weights.sum() * pred.shape[-1]) + 1e-8
    mux = float((pred * weights).sum() / denom)
    muy = float((tgt * weights).sum() / denom)
    vx = float((((pred - mux) ** 2) * weights).sum() / denom)
    vy = float((((tgt - muy) ** 2) * weights).sum() / denom)
    cov = float((((pred - mux) * (tgt - muy)) * weights).sum() / denom)
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2)))


def metric_bundle(prediction, target, mask) -> dict[str, float]:
    return {
        "psnr": psnr(prediction, target),
        "ssim": ssim(prediction, target),
        "weighted_psnr": psnr(prediction, target, mask),
        "weighted_ssim": ssim(prediction, target, mask),
    }


def mask_iou(predicted_mask, target_mask, threshold: float = 0.5) -> float:
    pred = mask_to_array(predicted_mask) >= threshold
    target = mask_to_array(target_mask) >= threshold
    intersection = float(np.logical_and(pred, target).sum())
    union = float(np.logical_or(pred, target).sum())
    return intersection / (union + 1e-8)


def mask_dice(predicted_mask, target_mask, threshold: float = 0.5) -> float:
    pred = mask_to_array(predicted_mask) >= threshold
    target = mask_to_array(target_mask) >= threshold
    intersection = float(np.logical_and(pred, target).sum())
    total = float(pred.sum() + target.sum())
    return 2.0 * intersection / (total + 1e-8)


def mask_bce(predicted_mask, target_mask, epsilon: float = 1e-6) -> float:
    pred = np.clip(mask_to_array(predicted_mask), epsilon, 1.0 - epsilon)
    target = mask_to_array(target_mask)
    return float((-(target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred))).mean())
