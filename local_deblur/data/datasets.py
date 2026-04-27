"""Manifest and dry-run datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .synthetic_blur import make_dry_run_sample
from .transforms import load_mask, load_rgb
from .types import LocalDeblurRecord, LocalDeblurSample


def _resolve(path: str | None, base: Path) -> Path | None:
    if not path:
        return None
    value = Path(path)
    return value if value.is_absolute() else base / value


class ManifestDeblurDataset:
    """Dataset backed by a JSON manifest with Ib/M/target/S fields."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        samples = payload.get("samples", payload) if isinstance(payload, dict) else payload
        if not isinstance(samples, list):
            raise ValueError("Manifest must be a list or a dict containing samples")
        self.base = self.manifest_path.parent
        self.records = [
            LocalDeblurRecord(
                sample_id=str(item.get("sample_id", f"sample_{i:06d}")),
                blurred_path=_resolve(item.get("Ib") or item.get("blurred"), self.base),
                mask_path=_resolve(item.get("M") or item.get("mask"), self.base),
                target_path=_resolve(item.get("target") or item.get("sharp"), self.base),
                segmentation_path=_resolve(item.get("S") or item.get("segmentation"), self.base),
                metadata=item.get("metadata", {}),
            )
            for i, item in enumerate(samples)
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[LocalDeblurSample]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> LocalDeblurSample:
        record = self.records[index]
        if record.blurred_path is None or record.mask_path is None:
            raise ValueError(f"Record {record.sample_id} must include Ib and M")
        sample = LocalDeblurSample(
            sample_id=record.sample_id,
            blurred=load_rgb(record.blurred_path),
            mask=load_mask(record.mask_path),
            target=load_rgb(record.target_path) if record.target_path else None,
            segmentation=load_mask(record.segmentation_path) if record.segmentation_path else None,
            metadata=record.metadata,
        )
        sample.validate()
        return sample


class DryRunDeblurDataset:
    def __init__(self, count: int = 1, size: int = 512, seed: int = 0):
        self.count = max(1, count)
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[LocalDeblurSample]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> LocalDeblurSample:
        if index < 0 or index >= self.count:
            raise IndexError(index)
        return make_dry_run_sample(sample_id=f"dry_run_{index:03d}", size=self.size, seed=self.seed + index)


def load_dataset(manifest: str | Path | None = None, *, dry_run: bool = False, count: int = 1, size: int = 512):
    if dry_run or manifest is None:
        return DryRunDeblurDataset(count=count, size=size)
    return ManifestDeblurDataset(manifest)
