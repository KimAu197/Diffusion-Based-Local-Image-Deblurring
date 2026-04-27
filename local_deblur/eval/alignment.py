"""Tiny alignment helper for smoke evaluation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from local_deblur.data.transforms import image_to_array


def align_prediction(prediction: Image.Image, target: Image.Image, max_shift: int = 1) -> Image.Image:
    """Search small integer shifts and return the prediction with lowest MSE."""
    pred = image_to_array(prediction)
    tgt = image_to_array(target)
    best = pred
    best_mse = float(np.mean((pred - tgt) ** 2))
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = np.roll(np.roll(pred, dy, axis=0), dx, axis=1)
            mse = float(np.mean((shifted - tgt) ** 2))
            if mse < best_mse:
                best = shifted
                best_mse = mse
    return Image.fromarray((np.clip(best, 0, 1) * 255 + 0.5).astype(np.uint8))
