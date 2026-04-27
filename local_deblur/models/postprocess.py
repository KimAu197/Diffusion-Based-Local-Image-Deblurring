"""Post-processing helpers for local restoration outputs."""

from __future__ import annotations

from PIL import Image, ImageFilter


def preserve_background(input_image: Image.Image, restored_image: Image.Image, mask: Image.Image) -> Image.Image:
    alpha = mask.convert("L")
    return Image.composite(restored_image.convert("RGB"), input_image.convert("RGB"), alpha)


def smooth_boundary(input_image: Image.Image, restored_image: Image.Image, mask: Image.Image, radius: float = 2.0) -> Image.Image:
    alpha = mask.convert("L").filter(ImageFilter.GaussianBlur(radius=radius))
    return Image.composite(restored_image.convert("RGB"), input_image.convert("RGB"), alpha)
