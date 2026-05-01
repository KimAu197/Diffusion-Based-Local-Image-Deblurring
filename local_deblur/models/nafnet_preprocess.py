"""Frozen NAFNet preprocessing for Stage 3 ControlNet training/inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import os
from PIL import Image, ImageFilter

from local_deblur.data.transforms import load_rgb, save_image


class ModelScopeNAFNetMaskPreprocessor:
    """Apply ModelScope NAFNet and composite it directly inside a mask.

    The model is frozen and lazily loaded. The full image is deblurred first,
    then the NAFNet output is used inside the mask with only boundary feathering.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        mask_blur_radius: float = 4.0,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_dir = str(model_dir)
        self.mask_blur_radius = float(mask_blur_radius)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._pipeline = None

    @classmethod
    def from_config(cls, config: dict[str, Any] | None):
        cfg = dict(config or {})
        if not bool(cfg.get("enabled", False)):
            return None
        return cls(
            cfg.get("model_dir", "/root/autodl-tmp/models/modelscope_nafnet/damo/cv_nafnet_image-deblur_gopro"),
            mask_blur_radius=float(cfg.get("mask_blur_radius", 4.0)),
            cache_dir=cfg.get("cache_dir"),
        )

    def _load_pipeline(self):
        if self._pipeline is None:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            self._pipeline = pipeline(Tasks.image_deblurring, model=self.model_dir)
        return self._pipeline

    def _cache_path(self, sample_id: str, size: tuple[int, int], crop_box: Any = None) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_id = str(sample_id).replace("/", "_").replace(" ", "_")
        if crop_box is not None:
            suffix = "_".join(str(int(v)) for v in crop_box)
        else:
            suffix = f"{size[0]}x{size[1]}"
        return self.cache_dir / f"{safe_id}_{suffix}_nafnet_masked.png"

    def process(
        self,
        image: Image.Image,
        mask: Image.Image,
        *,
        sample_id: str = "sample",
        crop_box: Any = None,
    ) -> Image.Image:
        cache_path = self._cache_path(sample_id, image.size, crop_box)
        if cache_path is not None and cache_path.is_file():
            return load_rgb(cache_path)

        import cv2
        from modelscope.outputs import OutputKeys

        pipe = self._load_pipeline()
        rgb = image.convert("RGB")
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
        naf_bgr = pipe(bgr)[OutputKeys.OUTPUT_IMG]
        naf_rgb = cv2.cvtColor(naf_bgr, cv2.COLOR_BGR2RGB)
        naf_image = Image.fromarray(naf_rgb).resize(rgb.size, Image.Resampling.BICUBIC)
        alpha = mask.convert("L").resize(rgb.size, Image.Resampling.BICUBIC).filter(
            ImageFilter.GaussianBlur(radius=self.mask_blur_radius)
        )
        output = Image.composite(naf_image.convert("RGB"), rgb, alpha)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.png")
            save_image(output, tmp)
            tmp.replace(cache_path)
        return output
