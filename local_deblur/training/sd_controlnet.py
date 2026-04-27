"""Training helpers for the SD + ControlNet local deblur path.

The helpers here prepare batches and auxiliary mask-head losses only. Full
diffusion training is intentionally not launched by task-015.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from local_deblur.data.types import LocalDeblurSample
from local_deblur.models.sd_controlnet import SDControlNetConfig, StableDiffusionControlNetLocalDeblurPipeline
from local_deblur.training.losses import mask_prediction_loss


@dataclass
class SDControlNetTrainingBatch:
    """Tensor contract for SD + ControlNet training integration."""

    sample_id: str
    Ib: Any
    M: Any
    S: Any
    target: Any
    controlnet_condition: Any
    mask_logits: Any | None
    mask_prob: Any | None


def prepare_sd_controlnet_training_batch(
    sample: LocalDeblurSample,
    config: SDControlNetConfig | dict[str, Any] | None = None,
) -> SDControlNetTrainingBatch:
    """Build tensors needed by a future SD + ControlNet training step."""

    pipeline = StableDiffusionControlNetLocalDeblurPipeline.from_config(config, load_checkpoints=False)
    prepared = pipeline.prepare_training_inputs(sample)
    mask_output = prepared["mask_output"]
    return SDControlNetTrainingBatch(
        sample_id=str(prepared["sample_id"]),
        Ib=prepared["Ib"].unsqueeze(0),
        M=prepared["M"].unsqueeze(0),
        S=prepared["S"].unsqueeze(0),
        target=prepared["target"].unsqueeze(0),
        controlnet_condition=prepared["controlnet_condition"],
        mask_logits=None if mask_output is None else mask_output.mask_logits,
        mask_prob=None if mask_output is None else mask_output.mask_prob,
    )


def auxiliary_mask_head_loss(mask_logits: Any, target_mask: Any, *, bce_weight: float = 1.0, dice_weight: float = 0.5):
    """Return the existing evaluation-compatible mask-head loss terms."""

    return mask_prediction_loss(mask_logits, target_mask, bce_weight=bce_weight, dice_weight=dice_weight)
