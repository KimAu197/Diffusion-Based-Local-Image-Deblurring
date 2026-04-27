"""Pipeline wrapper for checkpoint-backed or fallback local deblurring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from local_deblur.data.types import LocalDeblurSample

from .conditioning import build_conditioning
from .fallback import FallbackDeblurModel
from .postprocess import smooth_boundary


@dataclass
class PipelineOutput:
    image: Image.Image
    predicted_mask: Image.Image | None
    metadata: dict[str, Any]


class LocalDeblurPipeline:
    """Stable Diffusion/ControlNet-compatible interface with a smoke fallback."""

    def __init__(self, checkpoint: str | Path | None = None, *, dry_run: bool = True, use_diffusers: bool = False):
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.dry_run = dry_run
        self.use_diffusers = use_diffusers and not dry_run
        self.fallback = FallbackDeblurModel()
        self.diffusion_pipeline = None

    @classmethod
    def load(cls, checkpoint: str | Path | None = None, *, dry_run: bool = True, use_diffusers: bool = False) -> "LocalDeblurPipeline":
        pipeline = cls(checkpoint=checkpoint, dry_run=dry_run, use_diffusers=use_diffusers)
        if pipeline.use_diffusers:
            if checkpoint is None:
                raise ValueError("A checkpoint path is required for diffusers-backed inference")
            try:
                from diffusers import DiffusionPipeline

                pipeline.diffusion_pipeline = DiffusionPipeline.from_pretrained(str(checkpoint), local_files_only=True)
            except ImportError as exc:
                raise RuntimeError("diffusers is not installed; use --dry-run or install optional dependencies") from exc
        return pipeline

    def prepare_inputs(self, sample: LocalDeblurSample) -> dict[str, Any]:
        sample.validate()
        conditioning = build_conditioning(sample)
        return {
            "image": sample.blurred,
            "mask": sample.mask,
            "segmentation": sample.segmentation,
            "conditioning": conditioning,
        }

    def generate(self, sample: LocalDeblurSample, **kwargs: Any) -> PipelineOutput:
        inputs = self.prepare_inputs(sample)
        if self.diffusion_pipeline is not None:
            # Placeholder: full diffusion integration should map image/mask/semantic features
            # into ControlNet, latent initialization, and MaskHead hooks without downloads.
            result = self.diffusion_pipeline(image=inputs["image"], mask_image=inputs["mask"], **kwargs)
            image = result.images[0] if hasattr(result, "images") else inputs["image"]
            predicted_mask = getattr(result, "predicted_mask", None)
        else:
            image = self.fallback(inputs["image"], inputs["mask"])
            predicted_mask = self.fallback.predict_mask(inputs["image"])
        image = smooth_boundary(sample.blurred, image, sample.mask)
        return PipelineOutput(
            image=image,
            predicted_mask=predicted_mask,
            metadata={
                "used_fallback": self.diffusion_pipeline is None,
                "checkpoint": str(self.checkpoint) if self.checkpoint else None,
                "conditioning_channels": int(inputs["conditioning"].conditioning.shape[-1]),
                "predicts_blur_mask": predicted_mask is not None,
                "mask_head": "lightweight-controlnet-feature-head",
            },
        )

    def __call__(self, sample: LocalDeblurSample, **kwargs: Any) -> PipelineOutput:
        return self.generate(sample, **kwargs)
