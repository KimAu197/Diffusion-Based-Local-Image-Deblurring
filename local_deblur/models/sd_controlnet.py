"""Stable Diffusion + ControlNet adapters for local deblurring.

This module is intentionally smoke-safe: config construction, conditioning
image assembly, and auxiliary mask-head forwards do not download checkpoints.
Diffusers checkpoint loading only happens when explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn

from local_deblur.data.tensor_dataset import sample_to_tensors
from local_deblur.data.transforms import array_to_image, image_to_array, mask_to_array, resize_to
from local_deblur.data.types import LocalDeblurSample


@dataclass
class SDControlNetConfig:
    """Configuration for SD + ControlNet local deblurring.

    Condition RGB channel mapping:
    - R: blurred image grayscale/context.
    - G: blur mask, where 1.0 means supervised/local deblur region.
    - B: segmentation map if provided; otherwise inverse mask background context.
    """

    base_sd_checkpoint: str | None = None
    controlnet_checkpoint: str | None = None
    mask_head_checkpoint: str | None = None
    cache_dir: str | None = None
    variant: str | None = None
    device: str | None = None
    local_files_only: bool = True
    allow_downloads: bool = False
    dtype: str = "float16"
    precision: str = "fp16"
    image_size: int = 512
    conditioning_channels: int = 3
    mask_head_enabled: bool = True
    mask_head_channels: int = 32

    def __post_init__(self) -> None:
        if self.conditioning_channels != 3:
            raise ValueError("ControlNet conditioning images must be RGB with 3 channels")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not self.allow_downloads:
            self.local_files_only = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SDControlNetConfig":
        data = dict(payload or {})
        if "base_checkpoint" in data and "base_sd_checkpoint" not in data:
            data["base_sd_checkpoint"] = data.pop("base_checkpoint")
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})

    @property
    def torch_dtype(self):
        import torch

        normalized = (self.dtype or self.precision).lower()
        if normalized in {"float16", "fp16", "half"}:
            return torch.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if normalized in {"float32", "fp32", "full"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype/precision: {self.dtype!r}")

    def diffusers_load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "local_files_only": self.local_files_only,
            "torch_dtype": self.torch_dtype,
        }
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        if self.variant:
            kwargs["variant"] = self.variant
        return kwargs


@dataclass
class SDControlNetMaskOutput:
    """Evaluation-compatible auxiliary blur-mask output contract."""

    mask_logits: Any
    mask_prob: Any


class ControlNetAuxMaskHead(nn.Module):
    """Lightweight torch mask head trained beside the diffusion objective.

    Diffusers does not expose a stable, version-independent feature hook for all
    ControlNet pipelines. This head consumes the RGB condition tensor for smoke
    and integration training, and can later be reattached to ControlNet features
    without changing the `mask_logits`/`mask_prob` evaluation contract.
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32):
        super().__init__()
        if in_channels <= 0 or hidden_channels <= 0:
            raise ValueError("in_channels and hidden_channels must be positive")
        groups = max(group for group in range(1, min(8, hidden_channels) + 1) if hidden_channels % group == 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> SDControlNetMaskOutput:
        logits = self.net(features)
        return SDControlNetMaskOutput(mask_logits=logits, mask_prob=torch.sigmoid(logits))


def _as_single_channel(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def build_controlnet_condition_image(
    blurred: Image.Image,
    mask: Image.Image,
    segmentation: Image.Image | None = None,
    *,
    image_size: int | None = None,
) -> Image.Image:
    """Build the RGB ControlNet condition image for local deblurring.

    RGB mapping is R=blurred grayscale context, G=blur mask, and B=segmentation
    if available. Without segmentation, B stores inverse-mask background context
    so the condition still distinguishes foreground edit regions from context.
    """

    if blurred.size != mask.size:
        raise ValueError(f"Blurred image and mask sizes differ: {blurred.size} vs {mask.size}")
    if segmentation is not None and segmentation.size != blurred.size:
        raise ValueError(f"Segmentation size differs: {segmentation.size} vs {blurred.size}")

    if image_size is not None:
        blurred = resize_to(blurred, image_size, is_mask=False)
        mask = resize_to(mask, image_size, is_mask=True)
        if segmentation is not None:
            segmentation = resize_to(segmentation, image_size, is_mask=True)

    blurred_rgb = image_to_array(blurred)
    blurred_gray = blurred_rgb.mean(axis=-1)
    mask_array = mask_to_array(mask)
    if segmentation is not None:
        blue = mask_to_array(segmentation)
    else:
        blue = blurred_gray * (1.0 - mask_array)
    condition = np.stack([blurred_gray, mask_array, blue], axis=-1)
    return array_to_image(condition).convert("RGB")


def build_controlnet_condition_from_sample(sample: LocalDeblurSample, *, image_size: int | None = None) -> Image.Image:
    sample.validate()
    return build_controlnet_condition_image(sample.blurred, sample.mask, sample.segmentation, image_size=image_size)


def controlnet_condition_to_tensor(condition_image: Image.Image, *, device: str | None = None, dtype=None):
    tensor = torch.from_numpy(image_to_array(condition_image)).permute(2, 0, 1).unsqueeze(0).contiguous()
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


class StableDiffusionControlNetLocalDeblurPipeline:
    """No-download wrapper around diffusers ControlNet img2img for deblurring."""

    def __init__(self, config: SDControlNetConfig):
        self.config = config
        self.controlnet = None
        self.pipeline = None
        self.mask_head = None
        if config.mask_head_enabled:
            self.mask_head = ControlNetAuxMaskHead(
                in_channels=config.conditioning_channels,
                hidden_channels=config.mask_head_channels,
            )

    @classmethod
    def from_config(
        cls,
        config: SDControlNetConfig | dict[str, Any] | None = None,
        *,
        load_checkpoints: bool = False,
    ) -> "StableDiffusionControlNetLocalDeblurPipeline":
        resolved = config if isinstance(config, SDControlNetConfig) else SDControlNetConfig.from_dict(config)
        instance = cls(resolved)
        if load_checkpoints:
            instance.load_diffusers()
        return instance

    def load_diffusers(self) -> None:
        """Load local diffusers components; may download only when config allows."""

        if not self.config.base_sd_checkpoint:
            raise ValueError("base_sd_checkpoint is required to load a diffusers pipeline")
        if not self.config.controlnet_checkpoint:
            raise ValueError("controlnet_checkpoint is required to load ControlNet weights")

        try:
            from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
        except ImportError as exc:
            raise RuntimeError("diffusers is required for SD + ControlNet checkpoint loading") from exc

        kwargs = self.config.diffusers_load_kwargs()
        controlnet_kwargs = dict(kwargs)
        try:
            self.controlnet = ControlNetModel.from_pretrained(self.config.controlnet_checkpoint, **controlnet_kwargs)
        except OSError:
            if "variant" not in controlnet_kwargs:
                raise
            controlnet_kwargs.pop("variant", None)
            self.controlnet = ControlNetModel.from_pretrained(self.config.controlnet_checkpoint, **controlnet_kwargs)

        pipeline_kwargs = dict(kwargs)
        pipeline_kwargs.update(
            {
                "controlnet": self.controlnet,
                "safety_checker": None,
                "requires_safety_checker": False,
            }
        )
        self.pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            self.config.base_sd_checkpoint,
            **pipeline_kwargs,
        )
        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(device)
        self.controlnet.to(device=device, dtype=self.config.torch_dtype)
        if self.mask_head is not None:
            self.mask_head.to(device=device, dtype=torch.float32)
            self.mask_head.eval()
            if self.config.mask_head_checkpoint:
                checkpoint = torch.load(Path(self.config.mask_head_checkpoint), map_location="cpu")
                state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                self.mask_head.load_state_dict(state_dict, strict=True)

    def prepare_condition(self, sample: LocalDeblurSample) -> Image.Image:
        return build_controlnet_condition_from_sample(sample, image_size=self.config.image_size)

    def predict_mask_from_condition(self, condition_image: Image.Image):
        if self.mask_head is None:
            return None

        device = next(self.mask_head.parameters()).device
        condition = controlnet_condition_to_tensor(condition_image, device=str(device), dtype=torch.float32)
        with torch.no_grad():
            return self.mask_head(condition)

    def prepare_training_inputs(self, sample: LocalDeblurSample) -> dict[str, Any]:
        item = sample_to_tensors(sample, image_size=self.config.image_size, include_segmentation=True)
        condition_image = self.prepare_condition(sample)
        condition_tensor = controlnet_condition_to_tensor(condition_image, dtype=None)
        mask_output = self.predict_mask_from_condition(condition_image)
        return {
            **item,
            "controlnet_condition_image": condition_image,
            "controlnet_condition": condition_tensor,
            "mask_output": mask_output,
        }

    def __call__(self, sample: LocalDeblurSample, **kwargs: Any):
        if self.pipeline is None:
            raise RuntimeError("Diffusers pipeline is not loaded; call load_diffusers() with local checkpoints first")
        condition_image = self.prepare_condition(sample)
        image = resize_to(sample.blurred, self.config.image_size, is_mask=False)
        return self.pipeline(image=image, control_image=condition_image, **kwargs)
