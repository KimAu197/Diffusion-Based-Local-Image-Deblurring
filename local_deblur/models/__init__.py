"""Model and pipeline interfaces."""

from .conditioning import ConditioningBatch, build_conditioning
from .mask_head import LightweightMaskHead
from .pipeline import LocalDeblurPipeline, PipelineOutput

try:
    from .conditional_unet import ConditionalDeblurOutput, ConditionalLocalDeblurNet
except ImportError as exc:  # Keep PIL fallback imports usable when torch is unavailable.
    if getattr(exc, "name", "") != "torch":
        raise
    ConditionalDeblurOutput = None
    ConditionalLocalDeblurNet = None

try:
    from .sd_controlnet import (
        ControlNetAuxMaskHead,
        SDControlNetConfig,
        SDControlNetMaskOutput,
        StableDiffusionControlNetLocalDeblurPipeline,
        build_controlnet_condition_from_sample,
        build_controlnet_condition_image,
        controlnet_condition_to_tensor,
    )
except ImportError as exc:  # Keep lightweight PIL fallback imports usable when torch/diffusers are unavailable.
    if getattr(exc, "name", "") not in {"torch", "diffusers"}:
        raise
    ControlNetAuxMaskHead = None
    SDControlNetConfig = None
    SDControlNetMaskOutput = None
    StableDiffusionControlNetLocalDeblurPipeline = None
    build_controlnet_condition_from_sample = None
    build_controlnet_condition_image = None
    controlnet_condition_to_tensor = None

__all__ = [
    "ConditionalDeblurOutput",
    "ConditionalLocalDeblurNet",
    "ConditioningBatch",
    "ControlNetAuxMaskHead",
    "LightweightMaskHead",
    "LocalDeblurPipeline",
    "PipelineOutput",
    "SDControlNetConfig",
    "SDControlNetMaskOutput",
    "StableDiffusionControlNetLocalDeblurPipeline",
    "build_conditioning",
    "build_controlnet_condition_from_sample",
    "build_controlnet_condition_image",
    "controlnet_condition_to_tensor",
]
