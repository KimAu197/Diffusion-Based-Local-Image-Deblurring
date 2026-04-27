"""Reusable inference helpers."""

from __future__ import annotations

from pathlib import Path

from local_deblur.data.synthetic_blur import make_dry_run_sample
from local_deblur.data.transforms import load_mask, load_rgb, save_image
from local_deblur.data.types import LocalDeblurSample
from local_deblur.models.pipeline import LocalDeblurPipeline


def load_inference_sample(
    image_path: str | Path | None,
    mask_path: str | Path | None,
    segmentation_path: str | Path | None = None,
    *,
    dry_run: bool = False,
) -> LocalDeblurSample:
    if dry_run and (image_path is None or mask_path is None):
        return make_dry_run_sample()
    if image_path is None or mask_path is None:
        raise ValueError("--image and --mask are required unless --dry-run is used")
    image = load_rgb(image_path)
    mask = load_mask(mask_path)
    segmentation = load_mask(segmentation_path) if segmentation_path else None
    sample = LocalDeblurSample(sample_id=Path(image_path).stem, blurred=image, mask=mask, segmentation=segmentation)
    sample.validate()
    return sample


def run_inference(
    *,
    image_path: str | Path | None,
    mask_path: str | Path | None,
    output_path: str | Path,
    segmentation_path: str | Path | None = None,
    checkpoint: str | Path | None = None,
    dry_run: bool = False,
    mask_output_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    sample = load_inference_sample(image_path, mask_path, segmentation_path, dry_run=dry_run)
    pipeline = LocalDeblurPipeline.load(checkpoint=checkpoint, dry_run=dry_run or checkpoint is None)
    output = pipeline(sample)
    saved_image = save_image(output.image, output_path)
    saved_mask = save_image(output.predicted_mask, mask_output_path) if output.predicted_mask and mask_output_path else None
    return saved_image, saved_mask
