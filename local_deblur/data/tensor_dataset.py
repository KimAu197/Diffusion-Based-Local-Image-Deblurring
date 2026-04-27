"""Torch tensor adapters for manifest-backed local deblur samples."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .datasets import ManifestDeblurDataset
from .transforms import image_to_array, mask_to_array, resize_to
from .types import LocalDeblurSample


def sample_to_tensors(sample: LocalDeblurSample, *, image_size: int | None = None, include_segmentation: bool = True) -> dict[str, Any]:
    """Convert a PIL LocalDeblurSample into CHW float tensors in [0, 1]."""
    blurred = sample.blurred
    mask = sample.mask
    target = sample.target or sample.blurred
    segmentation = sample.segmentation

    if image_size is not None:
        blurred = resize_to(blurred, image_size, is_mask=False)
        mask = resize_to(mask, image_size, is_mask=True)
        target = resize_to(target, image_size, is_mask=False)
        if segmentation is not None:
            segmentation = resize_to(segmentation, image_size, is_mask=True)

    Ib = torch.from_numpy(image_to_array(blurred)).permute(2, 0, 1).contiguous()
    M = torch.from_numpy(mask_to_array(mask)).unsqueeze(0).contiguous()
    target_tensor = torch.from_numpy(image_to_array(target)).permute(2, 0, 1).contiguous()

    item: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "Ib": Ib,
        "M": M,
        "target": target_tensor,
    }
    if include_segmentation:
        if segmentation is None:
            item["S"] = torch.zeros_like(M)
        else:
            item["S"] = torch.from_numpy(mask_to_array(segmentation)).unsqueeze(0).contiguous()
    return item


def deterministic_split_indices(total: int, *, val_fraction: float = 0.1, seed: int = 42) -> tuple[list[int], list[int]]:
    """Return reproducible train/validation indices for a manifest-backed dataset."""
    if total <= 1:
        return list(range(total)), []
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(total * val_fraction)))
    val_count = min(val_count, total - 1)
    val_indices = sorted(indices[:val_count])
    train_indices = sorted(indices[val_count:])
    return train_indices, val_indices


class TensorManifestDeblurDataset(Dataset):
    """Torch Dataset wrapper around ManifestDeblurDataset."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_size: int | None = None,
        include_segmentation: bool = True,
        indices: list[int] | None = None,
    ):
        self.dataset = ManifestDeblurDataset(manifest_path)
        self.image_size = image_size
        self.include_segmentation = include_segmentation
        self.indices = list(indices) if indices is not None else list(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[self.indices[index]]
        return sample_to_tensors(sample, image_size=self.image_size, include_segmentation=self.include_segmentation)
