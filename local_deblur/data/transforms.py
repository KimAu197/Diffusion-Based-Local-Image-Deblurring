"""Image loading and conversion helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def save_image(image: Image.Image, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def mask_to_array(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask.convert("L"), dtype=np.float32) / 255.0


def array_to_image(array: np.ndarray) -> Image.Image:
    clipped = np.clip(array, 0.0, 1.0)
    return Image.fromarray((clipped * 255.0 + 0.5).astype(np.uint8))


def resize_to(image: Image.Image, size: int = 512, *, is_mask: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
    return image.resize((size, size), resample)
