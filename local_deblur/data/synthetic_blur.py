"""Synthetic local blur and mask utilities."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .transforms import array_to_image, image_to_array, mask_to_array, save_image
from .types import LocalDeblurSample


def make_arbitrary_mask(size: tuple[int, int], seed: int = 0, shapes: int = 4) -> Image.Image:
    rng = random.Random(seed)
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(shapes):
        cx = rng.randint(width // 5, width * 4 // 5)
        cy = rng.randint(height // 5, height * 4 // 5)
        rx = rng.randint(max(8, width // 12), max(12, width // 4))
        ry = rng.randint(max(8, height // 12), max(12, height // 4))
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(radius=max(1, min(size) // 80)))


def object_aware_mask(instance_mask: Image.Image | None, size: tuple[int, int], seed: int = 0) -> Image.Image:
    if instance_mask is None:
        return make_arbitrary_mask(size, seed=seed)
    mask = instance_mask.convert("L").resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(mask)
    if arr.max() == 0:
        return make_arbitrary_mask(size, seed=seed)
    return Image.fromarray(((arr > 0) * 255).astype(np.uint8), mode="L")


def apply_gaussian_local_blur(image: Image.Image, mask: Image.Image, radius: float = 5.0) -> Image.Image:
    blurred = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    return composite_masked(image.convert("RGB"), blurred, mask)


def apply_motion_local_blur(image: Image.Image, mask: Image.Image, radius: int = 9) -> Image.Image:
    radius = max(3, radius | 1)
    kernel = np.zeros((radius, radius), dtype=np.float32)
    kernel[radius // 2, :] = 1.0 / radius
    flat = kernel.flatten().tolist()
    blurred = image.convert("RGB").filter(ImageFilter.Kernel((radius, radius), flat, scale=1.0))
    return composite_masked(image.convert("RGB"), blurred, mask)


def apply_defocus_local_blur(image: Image.Image, mask: Image.Image, radius: int = 5) -> Image.Image:
    blurred = image.convert("RGB").filter(ImageFilter.BoxBlur(radius=max(1, radius)))
    return composite_masked(image.convert("RGB"), blurred, mask)


def composite_masked(background: Image.Image, foreground: Image.Image, mask: Image.Image) -> Image.Image:
    alpha = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=1.5))
    return Image.composite(foreground.convert("RGB"), background.convert("RGB"), alpha)


def make_gradient_image(size: tuple[int, int] = (512, 512), seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    width, height = size
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    arr = np.stack(
        [
            np.tile(x, (height, 1)),
            np.tile(y, (1, width)),
            0.35 + 0.25 * np.sin(2 * math.pi * (x[None, :] + y)),
        ],
        axis=-1,
    )
    arr += rng.normal(0, 0.015, arr.shape).astype(np.float32)
    return array_to_image(arr)


def make_dry_run_sample(sample_id: str = "dry_run_000", size: int = 512, seed: int = 0) -> LocalDeblurSample:
    target = make_gradient_image((size, size), seed=seed)
    mask = make_arbitrary_mask((size, size), seed=seed)
    blurred = apply_gaussian_local_blur(target, mask, radius=6.0)
    segmentation = Image.fromarray(((mask_to_array(mask) > 0.1) * 127).astype(np.uint8), mode="L")
    sample = LocalDeblurSample(sample_id=sample_id, blurred=blurred, mask=mask, target=target, segmentation=segmentation)
    sample.validate()
    return sample


def write_dry_run_artifacts(output_dir: str | Path, count: int = 1, size: int = 512) -> list[dict[str, str]]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for index in range(count):
        sample = make_dry_run_sample(sample_id=f"dry_run_{index:03d}", size=size, seed=index)
        sample_dir = root / sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        blurred = save_image(sample.blurred, sample_dir / "Ib.png")
        mask = save_image(sample.mask, sample_dir / "M.png")
        target = save_image(sample.target, sample_dir / "target.png") if sample.target else None
        segmentation = save_image(sample.segmentation, sample_dir / "S.png") if sample.segmentation else None
        records.append(
            {
                "sample_id": sample.sample_id,
                "Ib": str(blurred),
                "M": str(mask),
                "target": str(target) if target else "",
                "S": str(segmentation) if segmentation else "",
            }
        )
    return records
