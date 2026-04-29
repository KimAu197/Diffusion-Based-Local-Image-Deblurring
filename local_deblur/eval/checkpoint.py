"""Checkpoint-backed evaluation model helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from local_deblur.data.tensor_dataset import sample_to_tensors
from local_deblur.data.transforms import array_to_image
from local_deblur.data.types import LocalDeblurSample
from local_deblur.models.pipeline import LocalDeblurPipeline, PipelineOutput
from local_deblur.models.postprocess import smooth_boundary
from local_deblur.models.sd_controlnet import SDControlNetConfig, StableDiffusionControlNetLocalDeblurPipeline


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


class SDControlNetCheckpointEvaluator:
    """Run trained SD + ControlNet checkpoints through the evaluation interface."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        checkpoint: str | None = None,
        device: str | None = None,
    ):
        sd_config = dict(config.get("sd_controlnet", {}))
        if checkpoint:
            sd_config["controlnet_checkpoint"] = checkpoint
        if device:
            sd_config["device"] = device
        self.prompt = str(config.get("inference", {}).get("prompt", "local deblur restoration"))
        self.negative_prompt = config.get("inference", {}).get("negative_prompt")
        self.num_inference_steps = int(config.get("inference", {}).get("num_inference_steps", 50))
        self.guidance_scale = float(config.get("inference", {}).get("guidance_scale", 7.5))
        self.strength = float(config.get("inference", {}).get("strength", 0.8))
        self.seed = config.get("inference", {}).get("seed")
        self.preserve_background = bool(config.get("runtime", {}).get("preserve_background", True))
        self.pipeline = StableDiffusionControlNetLocalDeblurPipeline.from_config(
            SDControlNetConfig.from_dict(sd_config),
            load_checkpoints=True,
        )

    def _generator(self):
        if self.seed is None:
            return None
        import torch

        device = self.pipeline.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return torch.Generator(device=device).manual_seed(int(self.seed))

    def __call__(self, sample: LocalDeblurSample, **_: Any) -> PipelineOutput:
        blank_mask = Image.new("L", sample.blurred.size, color=0)
        input_sample = LocalDeblurSample(
            sample_id=sample.sample_id,
            blurred=sample.blurred,
            mask=blank_mask,
            target=sample.target,
            segmentation=None,
            metadata=sample.metadata,
        )
        bootstrap_condition = self.pipeline.prepare_condition(input_sample)
        mask_output = self.pipeline.predict_mask_from_condition(bootstrap_condition)
        predicted_mask = None
        if mask_output is not None and getattr(mask_output, "mask_prob", None) is not None:
            mask_prob = mask_output.mask_prob.detach().float().cpu()
            if mask_prob.ndim == 4:
                mask_prob = mask_prob[0, 0]
            predicted_mask = array_to_image(mask_prob.numpy()).convert("L")

        inference_mask = predicted_mask.resize(sample.blurred.size) if predicted_mask is not None else blank_mask
        inference_sample = LocalDeblurSample(
            sample_id=sample.sample_id,
            blurred=sample.blurred,
            mask=inference_mask,
            target=sample.target,
            segmentation=inference_mask,
            metadata=sample.metadata,
        )
        result = self.pipeline(
            inference_sample,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            strength=self.strength,
            generator=self._generator(),
        )
        restored = result.images[0] if hasattr(result, "images") else result
        if self.preserve_background:
            input_image = sample.blurred.resize(restored.size)
            mask = inference_mask.resize(restored.size)
            restored = smooth_boundary(input_image, restored, mask)

        return PipelineOutput(
            image=restored,
            predicted_mask=predicted_mask,
            metadata={
                "used_fallback": False,
                "checkpoint": self.pipeline.config.controlnet_checkpoint,
                "base_sd_checkpoint": self.pipeline.config.base_sd_checkpoint,
                "mask_head_checkpoint": self.pipeline.config.mask_head_checkpoint,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "strength": self.strength,
                "seed": self.seed,
                "preserve_background": self.preserve_background,
                "predicts_blur_mask": predicted_mask is not None,
                "uses_gt_mask_as_input": False,
                "mask_head": "ControlNetAuxMaskHead",
            },
        )


def build_eval_model(
    *,
    checkpoint: str | None = None,
    dry_run: bool = False,
    device: str | None = None,
    config: dict[str, Any] | None = None,
    model_type: str | None = None,
):
    if dry_run:
        return LocalDeblurPipeline.load(checkpoint=checkpoint, dry_run=True)
    if model_type == "sd_controlnet":
        return SDControlNetCheckpointEvaluator(config or {}, checkpoint=checkpoint, device=device)
    if checkpoint is None:
        raise ValueError("A trained PyTorch checkpoint is required unless --dry-run is set")
    return TrainableCheckpointEvaluator(checkpoint, device=device)
