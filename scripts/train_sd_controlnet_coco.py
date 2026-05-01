#!/usr/bin/env python
"""Bounded SD + ControlNet pretraining smoke on the COCO synthetic local-blur set."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_deblur.config import load_yaml_config
from local_deblur.data.datasets import ManifestDeblurDataset
from local_deblur.data.tensor_dataset import deterministic_split_indices, sample_to_tensors
from local_deblur.data.types import LocalDeblurSample
from local_deblur.logging_utils import configure_logging
from local_deblur.models.nafnet_preprocess import ModelScopeNAFNetMaskPreprocessor
from local_deblur.models.sd_controlnet import ControlNetAuxMaskHead, build_controlnet_condition_from_sample
from local_deblur.paths import resolve_project_path
from local_deblur.training.sd_controlnet import auxiliary_mask_head_loss


@dataclass
class RunPaths:
    output_dir: Path
    log_dir: Path
    checkpoint_dir: Path
    loss_curve: Path
    summary: Path
    log_file: Path


class SDControlNetCocoDataset:
    """Manifest dataset that emits target, mask, and RGB ControlNet condition tensors."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        indices: list[int],
        image_size: int,
        training: bool = False,
        bapc_probability: float = 0.0,
        bapc_min_mask_ratio: float = 0.05,
        nafnet_preprocess_config: dict[str, Any] | None = None,
    ):
        self.dataset = ManifestDeblurDataset(manifest)
        self.indices = list(indices)
        self.image_size = image_size
        self.training = training
        self.bapc_probability = float(bapc_probability)
        self.bapc_min_mask_ratio = float(bapc_min_mask_ratio)
        self.nafnet_preprocessor = ModelScopeNAFNetMaskPreprocessor.from_config(nafnet_preprocess_config)

    def __len__(self) -> int:
        return len(self.indices)

    def _crop_sample(self, sample: LocalDeblurSample) -> LocalDeblurSample:
        """Apply ReLoBlur-style blur-aware patch cropping for training samples."""

        patch_size = int(self.image_size)
        width, height = sample.blurred.size
        if not self.training or patch_size <= 0 or width <= patch_size or height <= patch_size:
            return sample

        left = random.randint(0, width - patch_size)
        top = random.randint(0, height - patch_size)
        mask_array = np.asarray(sample.mask.convert("L"))
        mask_ratio = float((mask_array > 0).mean())
        force_blur_region = self.bapc_probability > 0.0 and random.random() < self.bapc_probability
        if force_blur_region and mask_ratio > self.bapc_min_mask_ratio:
            margin = patch_size // 2
            interior = mask_array[margin : height - margin, margin : width - margin] > 0
            blur_pixels = np.argwhere(interior)
            if len(blur_pixels) > 0:
                center_y, center_x = blur_pixels[random.randrange(len(blur_pixels))]
                top = int(center_y)
                left = int(center_x)

        box = (left, top, left + patch_size, top + patch_size)
        metadata = dict(sample.metadata or {})
        metadata.update(
            {
                "bapc_enabled": True,
                "bapc_probability": self.bapc_probability,
                "bapc_min_mask_ratio": self.bapc_min_mask_ratio,
                "bapc_mask_ratio": mask_ratio,
                "crop_box": box,
            }
        )
        return LocalDeblurSample(
            sample_id=sample.sample_id,
            blurred=sample.blurred.crop(box),
            mask=sample.mask.crop(box),
            target=sample.target.crop(box) if sample.target is not None else None,
            segmentation=sample.segmentation.crop(box) if sample.segmentation is not None else None,
            metadata=metadata,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[self.indices[index]]
        sample = self._crop_sample(sample)
        if self.nafnet_preprocessor is not None:
            metadata = dict(sample.metadata or {})
            processed = self.nafnet_preprocessor.process(
                sample.blurred,
                sample.mask,
                sample_id=sample.sample_id,
                crop_box=metadata.get("crop_box"),
            )
            metadata.update(
                {
                    "nafnet_preprocess": "modelscope_gopro_width64_direct_mask_soft_boundary",
                    "nafnet_model_dir": self.nafnet_preprocessor.model_dir,
                    "nafnet_mask_blur_radius": self.nafnet_preprocessor.mask_blur_radius,
                }
            )
            sample = LocalDeblurSample(
                sample_id=sample.sample_id,
                blurred=processed,
                mask=sample.mask,
                target=sample.target,
                segmentation=sample.segmentation,
                metadata=metadata,
            )
        item = sample_to_tensors(sample, image_size=self.image_size, include_segmentation=True)
        condition = build_controlnet_condition_from_sample(sample, image_size=self.image_size)
        import torch

        condition_tensor = torch.from_numpy(np.asarray(condition, dtype="float32") / 255.0)
        item["controlnet_condition"] = condition_tensor.permute(2, 0, 1).contiguous()
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_sd_controlnet_coco.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--subset-train-count", type=int, default=None)
    parser.add_argument("--validation-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a saved step_* checkpoint directory, or checkpoint/ containing latest.json, or a controlnet/ subfolder of a step_* dir.",
    )
    parser.add_argument(
        "--log-append",
        action="store_true",
        help="Append to logging.log and loss_curve.csv instead of overwriting (automatic when --resume is set).",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Optional W&B run id; if set, wandb.init uses resume=allow to continue the same run (also respects WANDB_RUN_ID env).",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def make_paths(output_dir: str | Path) -> RunPaths:
    root = resolve_project_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_dir = root / "log"
    checkpoint_dir = root / "checkpoint"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        output_dir=root,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        loss_curve=root / "loss_curve.csv",
        summary=root / "training_summary.json",
        log_file=root / "logging.log",
    )


def scalar(value: Any) -> float:
    return float(value.detach().cpu()) if hasattr(value, "detach") else float(value)


def write_blocked_report(paths: RunPaths, config: dict[str, Any], stage: str, exc: BaseException) -> None:
    payload = {
        "status": "blocked",
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "label": "blocked_no_diffusers_backed_checkpoint",
        "used_baseline_fallback": False,
        "config": config,
        "artifacts": {
            "summary": str(paths.summary),
            "log": str(paths.log_file),
            "loss_curve": None,
            "checkpoint": None,
        },
    }
    paths.summary.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_sd_controlnet_components(config: dict[str, Any], device: Any, logger: logging.Logger):
    import torch
    from diffusers import ControlNetModel, StableDiffusionPipeline

    sd_config = config["sd_controlnet"]
    checkpoint = sd_config["base_sd_checkpoint"]
    controlnet_checkpoint = sd_config.get("controlnet_checkpoint")
    controlnet_init = sd_config.get("controlnet_init", "from_unet")
    local_files_only = bool(sd_config.get("local_files_only", False))
    cache_dir = sd_config.get("cache_dir")
    variant = sd_config.get("variant")
    dtype_name = str(sd_config.get("dtype", "float32")).lower()
    dtype = torch.float16 if dtype_name in {"float16", "fp16", "half"} else torch.float32
    logger.info(
        "loading Stable Diffusion checkpoint=%s ControlNet=%s init=%s local_files_only=%s dtype=%s",
        checkpoint,
        controlnet_checkpoint,
        controlnet_init,
        local_files_only,
        dtype,
    )
    pipe_kwargs = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
        "cache_dir": cache_dir,
        "safety_checker": None,
        "requires_safety_checker": False,
    }
    if variant:
        pipe_kwargs["variant"] = variant
    try:
        pipe = StableDiffusionPipeline.from_pretrained(checkpoint, **pipe_kwargs)
    except OSError:
        if not variant:
            raise
        logger.warning("base checkpoint variant=%s unavailable; retrying without variant", variant)
        pipe_kwargs.pop("variant", None)
        pipe = StableDiffusionPipeline.from_pretrained(checkpoint, **pipe_kwargs)
    pipe.to(device)
    if controlnet_checkpoint:
        controlnet_kwargs = {
            "torch_dtype": dtype,
            "local_files_only": local_files_only,
            "cache_dir": cache_dir,
        }
        if variant:
            controlnet_kwargs["variant"] = variant
        try:
            controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint, **controlnet_kwargs).to(device=device, dtype=dtype)
        except OSError:
            if not variant:
                raise
            logger.warning("ControlNet variant=%s unavailable; retrying without variant", variant)
            controlnet_kwargs.pop("variant", None)
            controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint, **controlnet_kwargs).to(device=device, dtype=dtype)
        logger.info("loaded pretrained ControlNet checkpoint=%s", controlnet_checkpoint)
    elif controlnet_init == "from_unet":
        controlnet = ControlNetModel.from_unet(pipe.unet).to(device=device, dtype=dtype)
        logger.info("initialized ControlNet from Stable Diffusion UNet")
    else:
        raise ValueError("controlnet_checkpoint is required unless controlnet_init is 'from_unet'")
    return pipe, controlnet, dtype


def limit_indices(indices: list[int], count: int | None) -> list[int]:
    if count is None or count <= 0:
        return indices
    return indices[:count]


def prepare_datasets(config: dict[str, Any], args: argparse.Namespace) -> tuple[SDControlNetCocoDataset, SDControlNetCocoDataset, dict[str, Any]]:
    data_config = config["data"]
    manifest = resolve_project_path(data_config["manifest"])
    validation_manifest_value = args.validation_manifest or data_config.get("val_manifest") or data_config.get("validation_manifest")
    validation_manifest = resolve_project_path(validation_manifest_value) if validation_manifest_value else None
    full_dataset = ManifestDeblurDataset(manifest)
    if validation_manifest is None:
        train_indices, val_indices = deterministic_split_indices(
            len(full_dataset),
            val_fraction=float(data_config.get("val_fraction", 0.1)),
            seed=int(data_config.get("split_seed", 42)),
        )
        val_manifest = manifest
        val_total_samples = len(full_dataset)
    else:
        train_indices = list(range(len(full_dataset)))
        val_dataset_for_count = ManifestDeblurDataset(validation_manifest)
        val_indices = list(range(len(val_dataset_for_count)))
        val_manifest = validation_manifest
        val_total_samples = len(val_dataset_for_count)
    subset_train_value = args.subset_train_count if args.subset_train_count is not None else data_config.get("subset_train_count", 8)
    subset_val_value = data_config.get("subset_val_count", 2)
    subset_train_count = int(subset_train_value) if subset_train_value is not None else None
    subset_val_count = int(subset_val_value) if subset_val_value is not None else None
    train_indices = limit_indices(train_indices, subset_train_count)
    val_indices = limit_indices(val_indices, subset_val_count)
    image_size = int(data_config.get("image_size", 64))
    bapc_probability = float(data_config.get("bapc_probability", 0.0))
    bapc_min_mask_ratio = float(data_config.get("bapc_min_mask_ratio", 0.05))
    nafnet_preprocess_config = config.get("nafnet_preprocess")
    metadata = {
        "manifest": str(manifest),
        "validation_manifest": str(val_manifest),
        "total_samples": len(full_dataset),
        "validation_total_samples": val_total_samples,
        "train_samples_used": len(train_indices),
        "val_samples_used": len(val_indices),
        "image_size": image_size,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "bapc_probability": bapc_probability,
        "bapc_min_mask_ratio": bapc_min_mask_ratio,
    }
    return (
        SDControlNetCocoDataset(
            manifest,
            indices=train_indices,
            image_size=image_size,
            training=True,
            bapc_probability=bapc_probability,
            bapc_min_mask_ratio=bapc_min_mask_ratio,
            nafnet_preprocess_config=nafnet_preprocess_config,
        ),
        SDControlNetCocoDataset(
            val_manifest,
            indices=val_indices,
            image_size=image_size,
            training=False,
            nafnet_preprocess_config=nafnet_preprocess_config,
        ),
        metadata,
    )


def encode_prompt(pipe: Any, prompt: str, batch_size: int, device: Any):
    import torch

    tokens = pipe.tokenizer(
        [prompt] * batch_size,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    with torch.no_grad():
        return pipe.text_encoder(input_ids)[0]


def diffusion_step(
    pipe: Any,
    controlnet: Any,
    batch: dict[str, Any],
    prompt: str,
    device: Any,
    dtype: Any,
    *,
    loss_mode: str = "full",
    mask_weight_scale: float = 0.0,
    timestep_min: int = 0,
    timestep_max: int | None = None,
):
    import torch
    import torch.nn.functional as F

    target = batch["target"].to(device=device, dtype=dtype)
    condition = batch["controlnet_condition"].to(device=device, dtype=dtype)
    normalized_loss_mode = str(loss_mode or "full").lower()
    with torch.no_grad():
        latents = pipe.vae.encode(target * 2.0 - 1.0).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        scheduler_steps = int(pipe.scheduler.config.num_train_timesteps)
        low = max(0, int(timestep_min))
        high = scheduler_steps if timestep_max is None else min(scheduler_steps, int(timestep_max) + 1)
        if low >= high:
            raise ValueError(f"Invalid timestep range [{low}, {high - 1}] for scheduler steps={scheduler_steps}")
        timesteps = torch.randint(
            low,
            high,
            (latents.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = encode_prompt(pipe, prompt, latents.shape[0], device)
    down_res, mid_res = controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=condition,
        return_dict=False,
    )
    noise_pred = pipe.unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        down_block_additional_residuals=down_res,
        mid_block_additional_residual=mid_res,
    ).sample
    if normalized_loss_mode in {"mask_weighted", "mask_only"}:
        mask = batch["M"].to(device=device, dtype=torch.float32)
        latent_mask = F.interpolate(mask, size=noise_pred.shape[-2:], mode="nearest")
        weights = latent_mask.expand_as(noise_pred.float())
        denom = weights.sum().clamp_min(1.0)
        return ((noise_pred.float() - noise.float()).pow(2) * weights).sum() / denom
    if normalized_loss_mode in {"mask_weighted_full", "full_mask_weighted", "mask_amplified"}:
        mask = batch["M"].to(device=device, dtype=torch.float32)
        latent_mask = F.interpolate(mask, size=noise_pred.shape[-2:], mode="nearest")
        weights = (1.0 + float(mask_weight_scale) * latent_mask).expand_as(noise_pred.float())
        return ((noise_pred.float() - noise.float()).pow(2) * weights).sum() / weights.sum().clamp_min(1.0)
    if normalized_loss_mode != "full":
        raise ValueError(f"Unsupported diffusion_loss_mode: {loss_mode}")
    return F.mse_loss(noise_pred.float(), noise.float())


def evaluate_validation_loss(
    pipe: Any,
    controlnet: Any,
    mask_head: Any | None,
    val_loader: Any,
    device: Any,
    dtype: Any,
    prompt: str,
    max_batches: int,
    mask_loss_kwargs: dict[str, float],
    diffusion_weight: float,
    mask_weight: float,
    diffusion_loss_mode: str,
    diffusion_mask_weight_scale: float,
    timestep_min: int,
    timestep_max: int | None,
) -> dict[str, float]:
    import torch

    controlnet.eval()
    if mask_head is not None:
        mask_head.eval()
    totals = {
        "val_total_loss": 0.0,
        "val_diffusion_loss": 0.0,
    }
    if mask_head is not None:
        totals.update({"val_mask_loss": 0.0, "val_mask_bce": 0.0, "val_mask_dice": 0.0})
    batches = 0
    with torch.no_grad():
        for batch in val_loader:
            diffusion_loss = diffusion_step(
                pipe,
                controlnet,
                batch,
                prompt,
                device,
                dtype,
                loss_mode=diffusion_loss_mode,
                mask_weight_scale=diffusion_mask_weight_scale,
                timestep_min=timestep_min,
                timestep_max=timestep_max,
            )
            total_loss = diffusion_weight * diffusion_loss
            totals["val_total_loss"] += scalar(total_loss)
            totals["val_diffusion_loss"] += scalar(diffusion_loss)
            if mask_head is not None:
                condition = batch["controlnet_condition"].to(device=device, dtype=torch.float32)
                mask = batch["M"].to(device=device, dtype=torch.float32)
                output = mask_head(condition)
                terms = auxiliary_mask_head_loss(output.mask_logits, mask, **mask_loss_kwargs)
                totals["val_mask_loss"] += scalar(terms["loss"])
                totals["val_mask_bce"] += scalar(terms["bce"])
                totals["val_mask_dice"] += scalar(terms["dice"])
            batches += 1
            if batches >= max_batches:
                break
    controlnet.train()
    if mask_head is not None:
        mask_head.train()
    if batches == 0:
        return totals
    return {key: value / batches for key, value in totals.items()}


def init_wandb_run(
    config: dict[str, Any],
    paths: RunPaths,
    data_metadata: dict[str, Any],
    logger: logging.Logger,
    run_id: str | None = None,
):
    wandb_config = config.get("wandb", {})
    if not bool(wandb_config.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb logging is enabled, but the wandb package is not installed") from exc

    effective_id = run_id or os.environ.get("WANDB_RUN_ID")
    init_kwargs: dict[str, Any] = {
        "project": str(wandb_config.get("project", "diffusion_blur")),
        "entity": str(wandb_config.get("entity", "Columbia_project")),
        "name": str(config.get("experiment", {}).get("name", "sd_controlnet_coco_pretrain")),
        "dir": str(paths.output_dir),
        "config": {
            "experiment": config.get("experiment", {}),
            "data": data_metadata,
            "sd_controlnet": config.get("sd_controlnet", {}),
            "training": config.get("training", {}),
        },
    }
    if effective_id:
        init_kwargs["id"] = str(effective_id)
        init_kwargs["resume"] = "allow"
    run = wandb.init(**init_kwargs)
    logger.info("wandb logging enabled run_id=%s url=%s", getattr(run, "id", None), getattr(run, "url", None))
    return run


def save_training_checkpoint(
    paths: RunPaths,
    controlnet: Any,
    mask_head: Any | None,
    optimizer: Any,
    *,
    step: int,
    epoch: int,
    config: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, str]:
    import torch

    checkpoint_root = paths.checkpoint_dir / f"step_{step:06d}"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    controlnet_path = checkpoint_root / "controlnet"
    mask_head_path = checkpoint_root / "aux_mask_head.pt"
    optimizer_path = checkpoint_root / "optimizer.pt"
    metadata_path = checkpoint_root / "metadata.json"
    controlnet.save_pretrained(controlnet_path)
    aux_mask_head_value = None
    if mask_head is not None:
        torch.save(
            {
                "model_state_dict": mask_head.state_dict(),
                "hidden_channels": int(config["sd_controlnet"].get("mask_head_channels", 16)),
                "step": step,
                "epoch": epoch,
            },
            mask_head_path,
        )
        aux_mask_head_value = str(mask_head_path)
    torch.save({"optimizer_state_dict": optimizer.state_dict(), "step": step, "epoch": epoch}, optimizer_path)
    metadata = {
        "step": step,
        "epoch": epoch,
        "controlnet": str(controlnet_path),
        "aux_mask_head": aux_mask_head_value,
        "optimizer": str(optimizer_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (paths.checkpoint_dir / "latest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("saved checkpoint step=%s epoch=%s path=%s", step, epoch, checkpoint_root)
    return {
        "root": str(checkpoint_root),
        "controlnet": str(controlnet_path),
        "aux_mask_head": aux_mask_head_value,
        "optimizer": str(optimizer_path),
        "metadata": str(metadata_path),
    }


def _resolve_resume_checkpoint_root(resume: Path, checkpoint_dir: Path) -> Path:
    """Return the directory of a single saved step (contains metadata.json, controlnet/, etc.)."""
    p = resolve_project_path(resume)
    if not p.exists():
        raise FileNotFoundError(f"resume path not found: {p}")
    if p.is_file():
        p = p.parent
    if p.name == "controlnet" and p.parent.is_dir() and p.parent.name.startswith("step_"):
        return p.parent
    if p.name.startswith("step_") and (p / "metadata.json").is_file():
        return p
    latest = p / "latest.json"
    if p.name == "checkpoint" and latest.is_file():
        meta = json.loads(latest.read_text(encoding="utf-8"))
        root = Path(meta.get("controlnet", "")).parent
        if root.is_dir() and (root / "metadata.json").is_file():
            return root
    if p.is_dir() and p.name == "checkpoint":
        alt_latest = checkpoint_dir / "latest.json"
        if alt_latest.is_file():
            meta = json.loads(alt_latest.read_text(encoding="utf-8"))
            root = Path(meta.get("controlnet", "")).parent
            if root.is_dir() and (root / "metadata.json").is_file():
                return root
    raise FileNotFoundError(
        f"Could not resolve resume checkpoint from {resume!s}; expected a step_* dir or checkpoint/ with latest.json"
    )


def _load_resumed_training_state(
    resume_root: Path,
    config: dict[str, Any],
    controlnet: Any,
    mask_head: Any | None,
    optimizer: Any,
    device: Any,
    dtype: Any,
    logger: logging.Logger,
) -> tuple[int, int]:
    """Load controlnet, mask head, and optimizer from a step_* save. Returns (start_step, start_epoch)."""
    import torch
    from diffusers import ControlNetModel

    metadata_path = resume_root / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing {metadata_path}")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    cn_path = Path(str(meta.get("controlnet", resume_root / "controlnet")))
    head_value = meta.get("aux_mask_head")
    head_path = Path(str(head_value)) if head_value else resume_root / "aux_mask_head.pt"
    opt_path = Path(str(meta.get("optimizer", resume_root / "optimizer.pt")))

    sd_config = config["sd_controlnet"]
    local_files_only = bool(sd_config.get("local_files_only", False))
    cache_dir = sd_config.get("cache_dir")
    variant = sd_config.get("variant")
    cn_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": local_files_only,
        "cache_dir": cache_dir,
    }
    if variant:
        cn_kwargs["variant"] = variant
    try:
        reloaded = ControlNetModel.from_pretrained(cn_path, **cn_kwargs).to(device=device, dtype=dtype)
    except OSError as exc:
        if not variant:
            raise
        logger.warning("resumed ControlNet variant=%s unavailable; retrying without variant (%s)", variant, exc)
        cn_kwargs.pop("variant", None)
        reloaded = ControlNetModel.from_pretrained(cn_path, **cn_kwargs).to(device=device, dtype=dtype)
    controlnet.load_state_dict(reloaded.state_dict())

    head_blob = {}
    if mask_head is not None:
        if not head_path.is_file():
            raise FileNotFoundError(f"missing {head_path}")
        head_blob = torch.load(head_path, map_location="cpu")
        mask_head.load_state_dict(head_blob["model_state_dict"], strict=True)
    opt_blob = torch.load(opt_path, map_location="cpu")
    optimizer.load_state_dict(opt_blob["optimizer_state_dict"])

    start_step = int(head_blob.get("step", opt_blob.get("step", 0)) or 0)
    start_epoch = int(head_blob.get("epoch", opt_blob.get("epoch", 0)) or 0)
    if start_step <= 0 and "step" in meta:
        try:
            start_step = int(meta["step"])
        except (TypeError, ValueError):
            pass
    if start_epoch <= 0 and "epoch" in meta:
        try:
            start_epoch = int(meta["epoch"])
        except (TypeError, ValueError):
            pass

    logger.info(
        "resumed from %s (step=%s epoch=%s); reloaded controlnet+mask+optimizer; pipe weights unchanged from base SD",
        resume_root,
        start_step,
        start_epoch,
    )
    return start_step, start_epoch


def _last_step_in_loss_curve(loss_curve: Path) -> int | None:
    if not loss_curve.is_file() or loss_curve.stat().st_size == 0:
        return None
    last_step: int | None = None
    with loss_curve.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row or "step" not in row or row["step"] in {"", None}:
                continue
            try:
                last_step = int(row["step"])
            except (TypeError, ValueError):
                continue
    return last_step


def run_training(config: dict[str, Any], args: argparse.Namespace, paths: RunPaths, logger: logging.Logger) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    seed = int(config["experiment"].get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.local_files_only:
        config["sd_controlnet"]["local_files_only"] = True
        config["sd_controlnet"]["allow_downloads"] = False

    train_dataset, val_dataset, data_metadata = prepare_datasets(config, args)
    batch_size = int(config["training"].get("batch_size", 1))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    pipe, controlnet, dtype = load_sd_controlnet_components(config, device, logger)

    for module in (pipe.vae, pipe.unet, pipe.text_encoder):
        module.requires_grad_(False)
        module.eval()
    controlnet.train()
    mask_head_enabled = bool(config["sd_controlnet"].get("mask_head_enabled", True))
    mask_head = None
    if mask_head_enabled:
        mask_head = ControlNetAuxMaskHead(
            in_channels=3,
            hidden_channels=int(config["sd_controlnet"].get("mask_head_channels", 16)),
        ).to(device=device, dtype=torch.float32)
        mask_head.train()
    mask_head_checkpoint = config["sd_controlnet"].get("mask_head_checkpoint")
    if mask_head is not None and mask_head_checkpoint and not args.resume:
        head_path = resolve_project_path(mask_head_checkpoint)
        head_blob = torch.load(head_path, map_location="cpu")
        mask_head.load_state_dict(head_blob["model_state_dict"], strict=True)
        logger.info("loaded auxiliary mask head checkpoint=%s", head_path)

    optimizer_params = list(controlnet.parameters())
    if mask_head is not None:
        optimizer_params.extend(mask_head.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=float(config["training"].get("learning_rate", 1e-4)),
    )
    start_step = 0
    resume_root: Path | None = None
    if args.resume:
        resume_root = _resolve_resume_checkpoint_root(Path(args.resume), paths.checkpoint_dir)
        if args.max_steps is not None:
            logger.warning("--max-steps is ignored when --resume is set; using the configured run length instead")
        start_step, _ = _load_resumed_training_state(
            resume_root, config, controlnet, mask_head, optimizer, device, dtype, logger
        )
        if start_step <= 0:
            raise RuntimeError("resume failed: could not read a positive start step from the checkpoint")
    epochs = int(args.epochs or config["training"].get("epochs", 1))
    gradient_accumulation_steps = max(1, int(config["training"].get("gradient_accumulation_steps", 1)))
    micro_batches_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(micro_batches_per_epoch / gradient_accumulation_steps)
    configured_max_steps = config["training"].get("max_steps")
    if args.resume is None and args.max_steps is not None:
        max_steps = int(args.max_steps)
    elif configured_max_steps is None:
        max_steps = steps_per_epoch * epochs
    else:
        configured_max_steps = int(configured_max_steps)
        max_steps = steps_per_epoch * epochs if configured_max_steps <= 0 else configured_max_steps
    if start_step > 0 and start_step >= max_steps:
        raise RuntimeError(f"resume step {start_step} is already at or past max_steps {max_steps}")
    diffusion_weight = float(config["training"].get("diffusion_loss_weight", 1.0))
    mask_weight = float(config["training"].get("mask_loss_weight", 0.1))
    if mask_head is None:
        mask_weight = 0.0
    mask_loss_kwargs = {
        "bce_weight": float(config["training"].get("mask_bce_weight", 1.0)),
        "dice_weight": float(config["training"].get("mask_dice_weight", 0.5)),
    }
    diffusion_loss_mode = str(config["training"].get("diffusion_loss_mode", "full"))
    diffusion_mask_weight_scale = float(config["training"].get("diffusion_mask_weight_scale", 0.0))
    timestep_min = int(config["training"].get("timestep_min", 0) or 0)
    timestep_max_config = config["training"].get("timestep_max")
    timestep_max = None if timestep_max_config is None else int(timestep_max_config)
    prompt = str(config["training"].get("prompt", "local deblur restoration"))
    align_target_step = start_step
    resume_ckpt_step = start_step
    if args.resume and paths.loss_curve.is_file():
        last_csv_step = _last_step_in_loss_curve(paths.loss_curve)
        if last_csv_step is not None and last_csv_step > resume_ckpt_step:
            align_batches = int(last_csv_step - resume_ckpt_step)
            logger.info(
                "resume alignment: loss_curve ends at step %s but checkpoint is step %s; running %s silent optimizer steps to match",
                last_csv_step,
                resume_ckpt_step,
                align_batches,
            )
            train_iter = iter(train_loader)
            for _ in range(align_batches):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                diffusion_loss = diffusion_step(
                    pipe,
                    controlnet,
                    batch,
                    prompt,
                    device,
                    dtype,
                    loss_mode=diffusion_loss_mode,
                    mask_weight_scale=diffusion_mask_weight_scale,
                    timestep_min=timestep_min,
                    timestep_max=timestep_max,
                )
                total_loss = diffusion_weight * diffusion_loss
                if mask_head is not None:
                    condition = batch["controlnet_condition"].to(device=device, dtype=torch.float32)
                    mask = batch["M"].to(device=device, dtype=torch.float32)
                    mask_output = mask_head(condition)
                    mask_terms = auxiliary_mask_head_loss(mask_output.mask_logits, mask, **mask_loss_kwargs)
                    total_loss = total_loss + mask_weight * mask_terms["loss"]
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()
            align_target_step = last_csv_step
        elif last_csv_step is not None and last_csv_step < resume_ckpt_step:
            logger.warning(
                "loss_curve last step %s is before checkpoint step %s; training logs may overlap or be inconsistent",
                last_csv_step,
                resume_ckpt_step,
            )
    if args.resume and align_target_step > 0 and (align_target_step % steps_per_epoch) != 0:
        logger.info(
            "note: resume is not on an epoch boundary; DataLoader order restarts at the start of the current "
            "epoch, so the already-seen prefix of the epoch is repeated (weights+optimizer are aligned to step %s).",
            align_target_step,
        )
    validation_batches = int(config["training"].get("validation_batches", 1))
    validation_interval_steps = int(config["training"].get("validation_interval_steps", steps_per_epoch))
    validation_interval_steps = max(1, validation_interval_steps)
    warmup_steps = max(0, int(config["training"].get("warmup_steps", 0) or 0))
    checkpoint_interval_steps = int(config["training"].get("checkpoint_interval_steps", 0) or 0)
    checkpoint_at_epoch_end = bool(config["training"].get("checkpoint_at_epoch_end", True))
    checkpoint_mid_epoch = bool(config["training"].get("checkpoint_mid_epoch", False))
    checkpoint_keep_last_n = int(config["training"].get("checkpoint_keep_last_n", 0) or 0)
    explicit_checkpoint_steps = {
        int(value)
        for value in config["training"].get("checkpoint_steps", [])
        if int(value) > 0
    }
    wandb_log_interval_steps = int(config.get("wandb", {}).get("log_interval_steps", 10))
    wandb_log_interval_steps = max(1, wandb_log_interval_steps)
    wandb_run = init_wandb_run(
        config,
        paths,
        data_metadata,
        logger,
        run_id=args.wandb_run_id,
    )
    log_append = bool(getattr(args, "log_append", False))
    loss_mode = "a" if log_append else "w"
    write_header = True
    if log_append and paths.loss_curve.is_file() and paths.loss_curve.stat().st_size > 0:
        write_header = False
    rows: list[dict[str, Any]] = []
    checkpoint_paths: list[dict[str, str]] = []
    start = time.time()
    step = align_target_step
    resume_epoch = 1 + (align_target_step // steps_per_epoch) if align_target_step > 0 else 1
    if resume_epoch < 1:
        resume_epoch = 1
    if resume_epoch > epochs:
        raise RuntimeError(
            f"resume epoch {resume_epoch} is beyond training.epochs {epochs} (align_target_step={align_target_step} steps/epoch={steps_per_epoch})"
        )
    with paths.loss_curve.open(loss_mode, newline="", encoding="utf-8") as handle:
        fieldnames = [
            "step",
            "epoch",
            "sample_id",
            "train_total_loss",
            "train_diffusion_loss",
            "val_total_loss",
            "val_diffusion_loss",
            "learning_rate",
            "used_baseline_fallback",
        ]
        if mask_head is not None:
            fieldnames[5:5] = ["train_mask_loss", "train_mask_bce", "train_mask_dice"]
            fieldnames[10:10] = ["val_mask_loss", "val_mask_bce", "val_mask_dice"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(resume_epoch, epochs + 1):
            accum_count = 0
            accum_sample_id = ""
            accum_total_loss = 0.0
            accum_diffusion_loss = 0.0
            accum_mask_loss = 0.0
            accum_mask_bce = 0.0
            accum_mask_dice = 0.0
            for micro_index, batch in enumerate(train_loader, start=1):
                sample_id = batch["sample_id"][0] if isinstance(batch["sample_id"], list) else str(batch["sample_id"])
                diffusion_loss = diffusion_step(
                    pipe,
                    controlnet,
                    batch,
                    prompt,
                    device,
                    dtype,
                    loss_mode=diffusion_loss_mode,
                    mask_weight_scale=diffusion_mask_weight_scale,
                    timestep_min=timestep_min,
                    timestep_max=timestep_max,
                )
                mask_terms = None
                total_loss = diffusion_weight * diffusion_loss
                if mask_head is not None:
                    condition = batch["controlnet_condition"].to(device=device, dtype=torch.float32)
                    mask = batch["M"].to(device=device, dtype=torch.float32)
                    mask_output = mask_head(condition)
                    mask_terms = auxiliary_mask_head_loss(mask_output.mask_logits, mask, **mask_loss_kwargs)
                    total_loss = total_loss + mask_weight * mask_terms["loss"]
                (total_loss / gradient_accumulation_steps).backward()
                accum_count += 1
                accum_sample_id = sample_id
                accum_total_loss += scalar(total_loss)
                accum_diffusion_loss += scalar(diffusion_loss)
                if mask_terms is not None:
                    accum_mask_loss += scalar(mask_terms["loss"])
                    accum_mask_bce += scalar(mask_terms["bce"])
                    accum_mask_dice += scalar(mask_terms["dice"])
                is_epoch_micro_end = micro_index == micro_batches_per_epoch
                should_optimizer_step = accum_count >= gradient_accumulation_steps or is_epoch_micro_end
                if not should_optimizer_step:
                    continue
                learning_rate = float(config["training"].get("learning_rate", 1e-4))
                if warmup_steps > 0:
                    learning_rate = learning_rate * min(1.0, float(step + 1) / float(warmup_steps))
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                should_validate = (step + 1) % validation_interval_steps == 0 or (step + 1) == max_steps
                val_metrics = (
                    evaluate_validation_loss(
                        pipe,
                        controlnet,
                        mask_head,
                        val_loader,
                        device,
                        dtype,
                        prompt,
                        validation_batches,
                        mask_loss_kwargs,
                        diffusion_weight,
                        mask_weight,
                        diffusion_loss_mode,
                        diffusion_mask_weight_scale,
                        timestep_min,
                        timestep_max,
                    )
                    if should_validate
                    else {}
                )
                row = {
                    "step": step + 1,
                    "epoch": epoch,
                    "sample_id": accum_sample_id,
                    "train_total_loss": accum_total_loss / accum_count,
                    "train_diffusion_loss": accum_diffusion_loss / accum_count,
                    "val_total_loss": "",
                    "val_diffusion_loss": "",
                    **val_metrics,
                    "learning_rate": learning_rate,
                    "used_baseline_fallback": False,
                }
                if mask_terms is not None:
                    row.update(
                        {
                            "train_mask_loss": accum_mask_loss / accum_count,
                            "train_mask_bce": accum_mask_bce / accum_count,
                            "train_mask_dice": accum_mask_dice / accum_count,
                            "val_mask_loss": row.get("val_mask_loss", ""),
                            "val_mask_bce": row.get("val_mask_bce", ""),
                            "val_mask_dice": row.get("val_mask_dice", ""),
                        }
                    )
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                should_log_wandb = (step + 1) % wandb_log_interval_steps == 0 or should_validate or (step + 1) == max_steps
                if wandb_run is not None and should_log_wandb:
                    wandb_run.log({key: value for key, value in row.items() if value != ""}, step=step + 1)
                if should_validate:
                    if mask_head is None:
                        logger.info(
                            "step=%s epoch=%s total=%.6f diffusion=%.6f val_total=%.6f val_diffusion=%.6f fallback=False",
                            row["step"],
                            row["epoch"],
                            row["train_total_loss"],
                            row["train_diffusion_loss"],
                            row["val_total_loss"],
                            row["val_diffusion_loss"],
                        )
                    else:
                        logger.info(
                            "step=%s epoch=%s total=%.6f diffusion=%.6f mask=%.6f val_total=%.6f val_diffusion=%.6f val_mask=%.6f fallback=False",
                            row["step"],
                            row["epoch"],
                            row["train_total_loss"],
                            row["train_diffusion_loss"],
                            row["train_mask_loss"],
                            row["val_total_loss"],
                            row["val_diffusion_loss"],
                            row["val_mask_loss"],
                        )
                else:
                    if mask_head is None:
                        logger.info(
                            "step=%s epoch=%s total=%.6f diffusion=%.6f fallback=False",
                            row["step"],
                            row["epoch"],
                            row["train_total_loss"],
                            row["train_diffusion_loss"],
                        )
                    else:
                        logger.info(
                            "step=%s epoch=%s total=%.6f diffusion=%.6f mask=%.6f fallback=False",
                            row["step"],
                            row["epoch"],
                            row["train_total_loss"],
                            row["train_diffusion_loss"],
                            row["train_mask_loss"],
                        )
                step += 1
                is_epoch_end = step % steps_per_epoch == 0
                is_final_step = step >= max_steps
                is_mid_epoch = checkpoint_mid_epoch and (step % steps_per_epoch) == max(1, steps_per_epoch // 2)
                should_checkpoint = (
                    step in explicit_checkpoint_steps
                    or (checkpoint_interval_steps > 0 and step % checkpoint_interval_steps == 0)
                    or is_mid_epoch
                    or (checkpoint_at_epoch_end and is_epoch_end)
                )
                if should_checkpoint:
                    checkpoint_paths.append(
                        save_training_checkpoint(
                            paths,
                            controlnet,
                            mask_head,
                            optimizer,
                            step=step,
                            epoch=epoch,
                            config=config,
                            logger=logger,
                        )
                    )
                    if checkpoint_keep_last_n > 0:
                        while len(checkpoint_paths) > checkpoint_keep_last_n:
                            stale_checkpoint = checkpoint_paths.pop(0)
                            stale_root = Path(stale_checkpoint["root"])
                            if stale_root.exists():
                                shutil.rmtree(stale_root)
                                logger.info("removed stale checkpoint path=%s keep_last_n=%s", stale_root, checkpoint_keep_last_n)
                if is_final_step:
                    break
                accum_count = 0
                accum_sample_id = ""
                accum_total_loss = 0.0
                accum_diffusion_loss = 0.0
                accum_mask_loss = 0.0
                accum_mask_bce = 0.0
                accum_mask_dice = 0.0
            if step >= max_steps:
                break

    controlnet_path = paths.checkpoint_dir / "controlnet"
    if bool(config["training"].get("save_controlnet", True)):
        controlnet.save_pretrained(controlnet_path)
    mask_head_path = paths.checkpoint_dir / "aux_mask_head.pt"
    aux_mask_head_artifact = None
    if mask_head is not None:
        torch.save(
            {
                "model_state_dict": mask_head.state_dict(),
                "hidden_channels": int(config["sd_controlnet"].get("mask_head_channels", 16)),
                "step": max_steps,
            },
            mask_head_path,
        )
        aux_mask_head_artifact = str(mask_head_path)
    config_copy = paths.output_dir / "config_used.yaml"
    shutil.copy2(resolve_project_path(args.config), config_copy)
    final_row = rows[-1] if rows else {}
    log_append_flag = bool(getattr(args, "log_append", False))
    summary = {
        "status": "completed",
        "label": config["experiment"].get("label", "sd_controlnet_coco_smoke"),
        "limitation": config["experiment"].get(
            "limitation",
            "Bounded SD + ControlNet smoke/pretraining run; not a converged full training checkpoint.",
        ),
        "used_baseline_fallback": False,
        "model": {
            "base_sd_checkpoint": config["sd_controlnet"]["base_sd_checkpoint"],
            "controlnet_init": config["sd_controlnet"].get("controlnet_init", "from_unet"),
            "controlnet_checkpoint": config["sd_controlnet"].get("controlnet_checkpoint"),
            "dtype": str(dtype),
            "device": str(device),
        },
        "data": data_metadata,
        "training": {
            "steps": max_steps,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "micro_batches_per_epoch": micro_batches_per_epoch,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "warmup_steps": warmup_steps,
            "resumed": bool(args.resume),
            "resume_checkpoint": str(resume_root) if args.resume and resume_root is not None else None,
            "resume_checkpoint_step": start_step if start_step > 0 else None,
            "resume_log_step": align_target_step if args.resume and align_target_step > 0 else None,
            "log_append": log_append_flag,
            "checkpoint_interval_steps": checkpoint_interval_steps,
            "checkpoint_at_epoch_end": checkpoint_at_epoch_end,
            "checkpoint_mid_epoch": checkpoint_mid_epoch,
            "checkpoint_keep_last_n": checkpoint_keep_last_n,
            "batch_size": batch_size,
            "learning_rate": float(config["training"].get("learning_rate", 1e-4)),
            "diffusion_loss_mode": diffusion_loss_mode,
            "diffusion_mask_weight_scale": diffusion_mask_weight_scale,
            "timestep_min": timestep_min,
            "timestep_max": timestep_max,
            "prompt": prompt,
            "mask_head_enabled": mask_head is not None,
            "elapsed_seconds": time.time() - start,
            "final_metrics": final_row,
        },
        "wandb": {
            "enabled": wandb_run is not None,
            "project": config.get("wandb", {}).get("project"),
            "entity": config.get("wandb", {}).get("entity"),
            "run_id": None if wandb_run is None else getattr(wandb_run, "id", None),
            "url": None if wandb_run is None else getattr(wandb_run, "url", None),
        },
        "artifacts": {
            "checkpoint_dir": str(paths.checkpoint_dir),
            "controlnet": str(controlnet_path),
            "aux_mask_head": aux_mask_head_artifact,
            "loss_curve": str(paths.loss_curve),
            "log": str(paths.log_file),
            "config": str(config_copy),
            "intermediate_checkpoints": checkpoint_paths,
            "latest_checkpoint": str(paths.checkpoint_dir / "latest.json"),
        },
    }
    paths.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.finish()
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.resume and args.max_steps is not None:
        args.max_steps = None
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.subset_train_count is not None:
        config.setdefault("data", {})["subset_train_count"] = args.subset_train_count
    if args.validation_manifest is not None:
        config.setdefault("data", {})["val_manifest"] = args.validation_manifest
    if args.resume:
        args.log_append = True
    if args.wandb_run_id:
        os.environ["WANDB_RUN_ID"] = str(args.wandb_run_id)
    output_dir = args.output_dir or config.get("experiment", {}).get("output_dir", "output/training/sd_controlnet_coco_pretrain")
    paths = make_paths(output_dir)
    log_file_mode = "a" if args.log_append else "w"
    logger = configure_logging("local_deblur.sd_controlnet_coco", paths.log_file, file_mode=log_file_mode)
    try:
        summary = run_training(config, args, paths, logger)
        print(json.dumps({"status": summary["status"], "summary": str(paths.summary)}, indent=2))
    except Exception as exc:
        logger.exception("SD + ControlNet smoke pretraining blocked")
        write_blocked_report(paths, config, stage="run_training", exc=exc)
        print(json.dumps({"status": "blocked", "summary": str(paths.summary), "error": str(exc)}, indent=2))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
