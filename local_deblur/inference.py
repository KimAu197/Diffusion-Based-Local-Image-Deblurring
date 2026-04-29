"""Reusable inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from local_deblur.data.transforms import array_to_image
from local_deblur.data.synthetic_blur import make_dry_run_sample
from local_deblur.data.transforms import load_mask, load_rgb, save_image
from local_deblur.data.types import LocalDeblurSample
from local_deblur.models.pipeline import LocalDeblurPipeline
from local_deblur.models.postprocess import smooth_boundary
from local_deblur.models.sd_controlnet import SDControlNetConfig, StableDiffusionControlNetLocalDeblurPipeline


def load_inference_sample(
    image_path: str | Path | None,
    mask_path: str | Path | None,
    segmentation_path: str | Path | None = None,
    *,
    dry_run: bool = False,
) -> LocalDeblurSample:
    if dry_run and (image_path is None or mask_path is None):
        return make_dry_run_sample()
    if image_path is None:
        raise ValueError("--image is required unless --dry-run is used")
    image = load_rgb(image_path)
    mask = load_mask(mask_path) if mask_path else Image.new("L", image.size, color=0)
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


def _mask_output_to_image(mask_output: Any) -> Image.Image | None:
    if mask_output is None or getattr(mask_output, "mask_prob", None) is None:
        return None
    prob = mask_output.mask_prob.detach().float().cpu()
    if prob.ndim == 4:
        prob = prob[0, 0]
    return array_to_image(prob.numpy()).convert("L")


def run_sd_controlnet_inference(
    *,
    image_path: str | Path,
    mask_path: str | Path | None,
    output_path: str | Path,
    sd_controlnet_config: dict[str, Any],
    segmentation_path: str | Path | None = None,
    mask_output_path: str | Path | None = None,
    prompt: str = "local deblur restoration",
    negative_prompt: str | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    strength: float = 0.8,
    seed: int | None = None,
    preserve_background: bool = True,
) -> tuple[Path, Path | None]:
    """Run the trained SD + ControlNet local-deblur checkpoint path."""

    sample = load_inference_sample(image_path, mask_path, segmentation_path, dry_run=False)
    pipeline = StableDiffusionControlNetLocalDeblurPipeline.from_config(
        SDControlNetConfig.from_dict(sd_controlnet_config),
        load_checkpoints=True,
    )

    generator = None
    if seed is not None:
        import torch

        device = pipeline.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(int(seed))

    bootstrap_sample = LocalDeblurSample(
        sample_id=sample.sample_id,
        blurred=sample.blurred,
        mask=Image.new("L", sample.blurred.size, color=0),
        target=sample.target,
        segmentation=None,
        metadata=sample.metadata,
    )
    bootstrap_condition = pipeline.prepare_condition(bootstrap_sample)
    predicted_mask = _mask_output_to_image(pipeline.predict_mask_from_condition(bootstrap_condition))
    inference_mask = predicted_mask.resize(sample.blurred.size) if predicted_mask is not None else bootstrap_sample.mask
    inference_sample = LocalDeblurSample(
        sample_id=sample.sample_id,
        blurred=sample.blurred,
        mask=inference_mask,
        target=sample.target,
        segmentation=inference_mask,
        metadata=sample.metadata,
    )

    result = pipeline(
        inference_sample,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        strength=float(strength),
        generator=generator,
    )
    restored = result.images[0] if hasattr(result, "images") else result
    if preserve_background:
        input_image = sample.blurred.resize(restored.size, Image.Resampling.BICUBIC)
        mask = inference_mask.resize(restored.size, Image.Resampling.NEAREST)
        restored = smooth_boundary(input_image, restored, mask)

    saved_image = save_image(restored, output_path)
    saved_mask = None
    if mask_output_path and predicted_mask is not None:
        saved_mask = save_image(predicted_mask, mask_output_path)
    return saved_image, saved_mask
