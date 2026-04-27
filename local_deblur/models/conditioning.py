"""Conditioning assembly for local deblurring pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from local_deblur.data.transforms import image_to_array, mask_to_array
from local_deblur.data.types import LocalDeblurSample


@dataclass
class ConditioningBatch:
    image: np.ndarray
    mask: np.ndarray
    segmentation: np.ndarray | None
    conditioning: np.ndarray


def build_conditioning(sample: LocalDeblurSample) -> ConditioningBatch:
    image = image_to_array(sample.blurred)
    mask = mask_to_array(sample.mask)[..., None]
    segmentation = mask_to_array(sample.segmentation)[..., None] if sample.segmentation is not None else None
    parts = [image, mask]
    if segmentation is not None:
        parts.append(segmentation)
    conditioning = np.concatenate(parts, axis=-1)
    return ConditioningBatch(image=image, mask=mask, segmentation=segmentation, conditioning=conditioning)
