"""Typed sample contracts for local deblurring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class LocalDeblurRecord:
    sample_id: str
    blurred_path: Path | None
    mask_path: Path | None
    target_path: Path | None
    segmentation_path: Path | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class LocalDeblurSample:
    sample_id: str
    blurred: Image.Image
    mask: Image.Image
    target: Image.Image | None = None
    segmentation: Image.Image | None = None
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.blurred.size != self.mask.size:
            raise ValueError(f"Image and mask sizes differ for {self.sample_id}: {self.blurred.size} vs {self.mask.size}")
        if self.target is not None and self.target.size != self.blurred.size:
            raise ValueError(f"Target size differs for {self.sample_id}: {self.target.size} vs {self.blurred.size}")
        if self.segmentation is not None and self.segmentation.size != self.blurred.size:
            raise ValueError(f"Segmentation size differs for {self.sample_id}: {self.segmentation.size} vs {self.blurred.size}")
