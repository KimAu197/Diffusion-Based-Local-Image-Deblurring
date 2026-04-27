"""Evaluation dataset helpers for deterministic manifest splits."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from local_deblur.data.datasets import DryRunDeblurDataset, ManifestDeblurDataset
from local_deblur.data.tensor_dataset import deterministic_split_indices
from local_deblur.data.transforms import resize_to
from local_deblur.data.types import LocalDeblurSample


class SplitManifestDeblurDataset:
    """Manifest-backed evaluation dataset using the same split policy as training."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str = "val",
        val_fraction: float = 0.1,
        split_seed: int = 42,
        count: int = 0,
        image_size: int | None = None,
    ):
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be one of: train, val, all")
        self.dataset = ManifestDeblurDataset(manifest_path)
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.image_size = image_size

        train_indices, val_indices = deterministic_split_indices(
            len(self.dataset),
            val_fraction=val_fraction,
            seed=split_seed,
        )
        if split == "train":
            selected = train_indices
        elif split == "val":
            selected = val_indices
        else:
            selected = list(range(len(self.dataset)))
        self.indices = selected[:count] if count and count > 0 else selected

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[LocalDeblurSample]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> LocalDeblurSample:
        manifest_index = self.indices[index]
        record = self.dataset.records[manifest_index]
        sample = self.dataset[manifest_index]
        if self.image_size is not None:
            sample = LocalDeblurSample(
                sample_id=sample.sample_id,
                blurred=resize_to(sample.blurred, self.image_size, is_mask=False),
                mask=resize_to(sample.mask, self.image_size, is_mask=True),
                target=resize_to(sample.target, self.image_size, is_mask=False) if sample.target else None,
                segmentation=resize_to(sample.segmentation, self.image_size, is_mask=True) if sample.segmentation else None,
                metadata=sample.metadata,
            )
        metadata = dict(sample.metadata or {})
        metadata.update(
            {
                "manifest": str(self.manifest_path),
                "manifest_index": manifest_index,
                "split": self.split,
                "split_seed": self.split_seed,
                "val_fraction": self.val_fraction,
                "image_size": self.image_size,
                "input_path": str(record.blurred_path) if record.blurred_path else "",
                "mask_path": str(record.mask_path) if record.mask_path else "",
                "target_path": str(record.target_path) if record.target_path else "",
                "segmentation_path": str(record.segmentation_path) if record.segmentation_path else "",
            }
        )
        sample.metadata = metadata
        sample.validate()
        return sample


def build_eval_dataset(
    dataset: str,
    count: int,
    *,
    dry_run: bool,
    manifest: str | None = None,
    split: str = "val",
    val_fraction: float = 0.1,
    split_seed: int = 42,
    image_size: int | None = None,
):
    if dry_run or manifest is None:
        effective_count = 1 if count == 0 else max(1, count)
        return DryRunDeblurDataset(count=effective_count, size=image_size or 512)
    return SplitManifestDeblurDataset(
        manifest,
        split=split,
        val_fraction=val_fraction,
        split_seed=split_seed,
        count=count,
        image_size=image_size,
    )
