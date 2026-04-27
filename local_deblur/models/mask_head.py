"""Lightweight blur-mask head interfaces."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from local_deblur.data.transforms import array_to_image, image_to_array


class LightweightMaskHead:
    """Predicts blur likelihood maps from intermediate features or image cues.

    Full training should attach this head to ControlNet encoder features and optimize
    it with BCE against the known blur mask. The fallback implementation uses local
    high-frequency residuals so smoke runs can exercise the same output contract.
    """

    def __init__(self, threshold: float = 0.12):
        self.threshold = threshold

    def __call__(self, image: Image.Image, features: np.ndarray | None = None) -> Image.Image:
        if features is not None:
            score = np.asarray(features, dtype=np.float32)
            if score.ndim == 3:
                score = score.mean(axis=-1)
            score = score - float(score.min())
            score = score / (float(score.max()) + 1e-8)
            return array_to_image(score).convert("L")

        rgb = image.convert("RGB")
        low_pass = rgb.filter(ImageFilter.GaussianBlur(radius=2.0))
        residual = np.abs(image_to_array(rgb) - image_to_array(low_pass)).mean(axis=-1)
        inverse_detail = 1.0 - (residual / (float(residual.max()) + 1e-8))
        soft = np.clip((inverse_detail - self.threshold) / max(1e-6, 1.0 - self.threshold), 0.0, 1.0)
        mask = array_to_image(soft).convert("L")
        return ImageOps.autocontrast(mask).filter(ImageFilter.GaussianBlur(radius=1.0))
