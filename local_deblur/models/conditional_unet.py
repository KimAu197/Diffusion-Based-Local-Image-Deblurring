"""Compact trainable PyTorch baseline for conditional local deblurring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ConditionalDeblurOutput:
    restored: torch.Tensor
    mask_logits: torch.Tensor
    mask_prob: torch.Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalLocalDeblurNet(nn.Module):
    """Small U-Net that consumes Ib, M, optional S and predicts image plus mask."""

    def __init__(self, base_channels: int = 32, *, use_segmentation: bool = True):
        super().__init__()
        self.use_segmentation = use_segmentation
        in_channels = 5 if use_segmentation else 4

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.dec2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.image_head = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        self.mask_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def _condition(self, Ib: torch.Tensor, M: torch.Tensor, S: torch.Tensor | None) -> torch.Tensor:
        if Ib.ndim != 4 or Ib.shape[1] != 3:
            raise ValueError("Ib must have shape [B, 3, H, W]")
        if M.ndim == 3:
            M = M.unsqueeze(1)
        if M.ndim != 4 or M.shape[1] != 1:
            raise ValueError("M must have shape [B, 1, H, W] or [B, H, W]")
        parts = [Ib, M]
        if self.use_segmentation:
            if S is None:
                S = torch.zeros_like(M)
            elif S.ndim == 3:
                S = S.unsqueeze(1)
            if S.ndim != 4 or S.shape[1] != 1:
                raise ValueError("S must have shape [B, 1, H, W] or [B, H, W]")
            parts.append(S)
        return torch.cat(parts, dim=1)

    def forward(self, Ib: torch.Tensor, M: torch.Tensor, S: torch.Tensor | None = None) -> ConditionalDeblurOutput:
        x = self._condition(Ib, M, S)
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2))
        b = self.bottleneck(F.avg_pool2d(e2, kernel_size=2))

        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        residual = torch.tanh(self.image_head(d1)) * 0.25
        restored = torch.clamp(Ib + residual * M, 0.0, 1.0)
        mask_logits = self.mask_head(d1)
        return ConditionalDeblurOutput(restored=restored, mask_logits=mask_logits, mask_prob=torch.sigmoid(mask_logits))
