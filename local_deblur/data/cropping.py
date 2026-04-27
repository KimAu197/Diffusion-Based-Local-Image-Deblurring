"""Mask-centered crop helpers."""

from __future__ import annotations

from PIL import Image
import numpy as np


def mask_centered_crop(
    image: Image.Image,
    mask: Image.Image,
    size: int = 512,
    segmentation: Image.Image | None = None,
    target: Image.Image | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image | None, Image.Image | None]:
    """Crop around the mask center, pad if needed, and resize to a square."""
    rgb = image.convert("RGB")
    mask_l = mask.convert("L").resize(rgb.size, Image.Resampling.NEAREST)
    arr = np.asarray(mask_l)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        cx, cy = rgb.size[0] // 2, rgb.size[1] // 2
    else:
        cx, cy = int(xs.mean()), int(ys.mean())

    side = min(max(size, int(max(rgb.size) * 0.75)), max(rgb.size))
    left = cx - side // 2
    top = cy - side // 2
    right = left + side
    bottom = top + side

    def crop_one(img: Image.Image | None, is_mask: bool = False) -> Image.Image | None:
        if img is None:
            return None
        mode = "L" if is_mask else "RGB"
        canvas = Image.new(mode, (side, side), 0)
        src = img.convert(mode).resize(rgb.size, Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC)
        src_box = (max(0, left), max(0, top), min(rgb.size[0], right), min(rgb.size[1], bottom))
        dst_box = (max(0, -left), max(0, -top))
        canvas.paste(src.crop(src_box), dst_box)
        return canvas.resize((size, size), Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC)

    return crop_one(rgb), crop_one(mask_l, True), crop_one(segmentation, True), crop_one(target)
