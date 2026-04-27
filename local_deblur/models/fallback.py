"""Deterministic fallback deblurring used for smoke tests."""

from __future__ import annotations

from PIL import Image, ImageFilter

from .postprocess import smooth_boundary
from .mask_head import LightweightMaskHead


class FallbackDeblurModel:
    """A lightweight local sharpening model, not a diffusion checkpoint."""

    def __init__(self, sharpen_radius: float = 1.3):
        self.sharpen_radius = sharpen_radius
        self.mask_head = LightweightMaskHead()

    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        enhanced = image.convert("RGB").filter(ImageFilter.UnsharpMask(radius=self.sharpen_radius, percent=180, threshold=2))
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        return smooth_boundary(image, enhanced, mask)

    def predict_mask(self, image: Image.Image) -> Image.Image:
        return self.mask_head(image)
