"""Checkpoint-backed evaluation model helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_deblur.data.tensor_dataset import sample_to_tensors
from local_deblur.data.transforms import array_to_image
from local_deblur.data.types import LocalDeblurSample
from local_deblur.models.pipeline import LocalDeblurPipeline, PipelineOutput


class TrainableCheckpointEvaluator:
    """Run ConditionalLocalDeblurNet checkpoints through the evaluation interface."""

    def __init__(self, checkpoint: str | Path, *, device: str | None = None):
        import torch

        from local_deblur.models.conditional_unet import ConditionalLocalDeblurNet

        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        payload: dict[str, Any] = torch.load(self.checkpoint, map_location="cpu")
        self.base_channels = int(payload.get("base_channels", 32))
        self.step = payload.get("step")
        self.split = payload.get("split", {})
        self.model = ConditionalLocalDeblurNet(base_channels=self.base_channels, use_segmentation=True).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    def __call__(self, sample: LocalDeblurSample, **_: Any) -> PipelineOutput:
        import torch

        item = sample_to_tensors(sample, image_size=None, include_segmentation=True)
        Ib = item["Ib"].unsqueeze(0).to(self.device)
        M = item["M"].unsqueeze(0).to(self.device)
        S = item["S"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(Ib, M, S)
        restored = output.restored[0].detach().cpu().permute(1, 2, 0).numpy()
        mask_prob = output.mask_prob[0, 0].detach().cpu().numpy()
        return PipelineOutput(
            image=array_to_image(restored),
            predicted_mask=array_to_image(mask_prob),
            metadata={
                "used_fallback": False,
                "checkpoint": str(self.checkpoint),
                "checkpoint_step": self.step,
                "device": str(self.device),
                "base_channels": self.base_channels,
                "predicts_blur_mask": True,
                "mask_head": "ConditionalLocalDeblurNet.mask_head",
                "split": self.split,
            },
        )


def build_eval_model(*, checkpoint: str | None = None, dry_run: bool = False, device: str | None = None):
    if dry_run:
        return LocalDeblurPipeline.load(checkpoint=checkpoint, dry_run=True)
    if checkpoint is None:
        raise ValueError("A trained PyTorch checkpoint is required unless --dry-run is set")
    return TrainableCheckpointEvaluator(checkpoint, device=device)
