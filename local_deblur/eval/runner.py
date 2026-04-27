"""Evaluation model invocation helpers."""

from __future__ import annotations

from local_deblur.models.pipeline import LocalDeblurPipeline

from .alignment import align_prediction
from .metrics import mask_bce, mask_dice, mask_iou, metric_bundle, psnr, ssim


def evaluate_sample(sample, pipeline: LocalDeblurPipeline) -> dict:
    output = pipeline(sample)
    target = sample.target or sample.blurred
    metrics = metric_bundle(output.image, target, sample.mask)
    aligned = align_prediction(output.image, target)
    metrics["aligned_psnr"] = psnr(aligned, target)
    metrics["aligned_ssim"] = ssim(aligned, target)
    metrics["aligned_weighted_psnr"] = psnr(aligned, target, sample.mask)
    metrics["aligned_weighted_ssim"] = ssim(aligned, target, sample.mask)
    if output.predicted_mask is not None:
        metrics["mask_iou"] = mask_iou(output.predicted_mask, sample.mask)
        metrics["mask_dice"] = mask_dice(output.predicted_mask, sample.mask)
        metrics["mask_bce"] = mask_bce(output.predicted_mask, sample.mask)
    return {
        "sample_id": sample.sample_id,
        "metrics": metrics,
        "metadata": output.metadata,
        "prediction": output.image,
        "predicted_mask": output.predicted_mask,
        "input": sample.blurred,
        "target": target,
        "mask": sample.mask,
        "sample_metadata": sample.metadata or {},
    }
