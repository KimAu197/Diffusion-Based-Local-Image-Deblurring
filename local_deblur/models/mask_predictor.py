"""Standalone blur-mask prediction models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from local_deblur.models.conditional_unet import ConvBlock


@dataclass
class MaskPredictorOutput:
    mask_logits: torch.Tensor
    mask_prob: torch.Tensor


class MaskPredictorUNet(nn.Module):
    """Compact U-Net for the Stage 1 task: blurred RGB image -> blur mask."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16):
        super().__init__()
        if in_channels != 3:
            raise ValueError("MaskPredictorUNet expects RGB Ib input with 3 channels")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.dec3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.mask_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, Ib: torch.Tensor) -> MaskPredictorOutput:
        if Ib.ndim != 4 or Ib.shape[1] != 3:
            raise ValueError("Ib must have shape [B, 3, H, W]")

        e1 = self.enc1(Ib)
        e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2))
        e3 = self.enc3(F.avg_pool2d(e2, kernel_size=2))
        b = self.bottleneck(F.avg_pool2d(e3, kernel_size=2))

        u3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        logits = self.mask_head(d1)
        return MaskPredictorOutput(mask_logits=logits, mask_prob=torch.sigmoid(logits))
