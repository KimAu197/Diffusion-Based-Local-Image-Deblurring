#!/usr/bin/env python
"""Bounded SD + ControlNet pretraining smoke on the COCO synthetic local-blur set."""

from __future__ import annotations

import argparse
import csv
import json
import logging
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
from local_deblur.logging_utils import configure_logging
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

    def __init__(self, manifest: str | Path, *, indices: list[int], image_size: int):
        self.dataset = ManifestDeblurDataset(manifest)
        self.indices = list(indices)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[self.indices[index]]
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
    parser.add_argument("--subset-train-count", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
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


def prepare_datasets(config: dict[str, Any], args: argparse.Namespace) -> tuple[SDControlNetCocoDataset, SDControlNetCocoDataset, dict[str, Any]]:
    data_config = config["data"]
    manifest = resolve_project_path(data_config["manifest"])
    full_dataset = ManifestDeblurDataset(manifest)
    train_indices, val_indices = deterministic_split_indices(
        len(full_dataset),
        val_fraction=float(data_config.get("val_fraction", 0.1)),
        seed=int(data_config.get("split_seed", 42)),
    )
    subset_train_count = int(args.subset_train_count or data_config.get("subset_train_count", 8))
    subset_val_count = int(data_config.get("subset_val_count", 2))
    train_indices = train_indices[:subset_train_count]
    val_indices = val_indices[:subset_val_count]
    image_size = int(data_config.get("image_size", 64))
    metadata = {
        "manifest": str(manifest),
        "total_samples": len(full_dataset),
        "train_samples_used": len(train_indices),
        "val_samples_used": len(val_indices),
        "image_size": image_size,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    return (
        SDControlNetCocoDataset(manifest, indices=train_indices, image_size=image_size),
        SDControlNetCocoDataset(manifest, indices=val_indices, image_size=image_size),
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


def diffusion_step(pipe: Any, controlnet: Any, batch: dict[str, Any], prompt: str, device: Any, dtype: Any):
    import torch
    import torch.nn.functional as F

    target = batch["target"].to(device=device, dtype=dtype)
    condition = batch["controlnet_condition"].to(device=device, dtype=dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(target * 2.0 - 1.0).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            pipe.scheduler.config.num_train_timesteps,
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
    return F.mse_loss(noise_pred.float(), noise.float())


def evaluate_mask_head(mask_head: Any, val_loader: Any, device: Any, max_batches: int, mask_loss_kwargs: dict[str, float]) -> dict[str, float]:
    import torch

    mask_head.eval()
    totals = {"val_mask_loss": 0.0, "val_mask_bce": 0.0, "val_mask_dice": 0.0}
    batches = 0
    with torch.no_grad():
        for batch in val_loader:
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
    mask_head.train()
    if batches == 0:
        return totals
    return {key: value / batches for key, value in totals.items()}


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
    mask_head = ControlNetAuxMaskHead(
        in_channels=3,
        hidden_channels=int(config["sd_controlnet"].get("mask_head_channels", 16)),
    ).to(device=device, dtype=torch.float32)
    mask_head.train()

    optimizer = torch.optim.AdamW(
        list(controlnet.parameters()) + list(mask_head.parameters()),
        lr=float(config["training"].get("learning_rate", 1e-4)),
    )
    max_steps = int(args.max_steps or config["training"].get("max_steps", 2))
    diffusion_weight = float(config["training"].get("diffusion_loss_weight", 1.0))
    mask_weight = float(config["training"].get("mask_loss_weight", 0.1))
    mask_loss_kwargs = {
        "bce_weight": float(config["training"].get("mask_bce_weight", 1.0)),
        "dice_weight": float(config["training"].get("mask_dice_weight", 0.5)),
    }
    prompt = str(config["training"].get("prompt", "local deblur restoration"))
    validation_batches = int(config["training"].get("validation_batches", 1))
    rows: list[dict[str, Any]] = []
    start = time.time()
    step = 0
    epoch = 1
    with paths.loss_curve.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "step",
            "epoch",
            "sample_id",
            "train_total_loss",
            "train_diffusion_loss",
            "train_mask_loss",
            "train_mask_bce",
            "train_mask_dice",
            "val_mask_loss",
            "val_mask_bce",
            "val_mask_dice",
            "learning_rate",
            "used_baseline_fallback",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        while step < max_steps:
            for batch in train_loader:
                sample_id = batch["sample_id"][0] if isinstance(batch["sample_id"], list) else str(batch["sample_id"])
                diffusion_loss = diffusion_step(pipe, controlnet, batch, prompt, device, dtype)
                condition = batch["controlnet_condition"].to(device=device, dtype=torch.float32)
                mask = batch["M"].to(device=device, dtype=torch.float32)
                mask_output = mask_head(condition)
                mask_terms = auxiliary_mask_head_loss(mask_output.mask_logits, mask, **mask_loss_kwargs)
                total_loss = diffusion_weight * diffusion_loss + mask_weight * mask_terms["loss"]
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

                val_metrics = evaluate_mask_head(mask_head, val_loader, device, validation_batches, mask_loss_kwargs)
                row = {
                    "step": step + 1,
                    "epoch": epoch,
                    "sample_id": sample_id,
                    "train_total_loss": scalar(total_loss),
                    "train_diffusion_loss": scalar(diffusion_loss),
                    "train_mask_loss": scalar(mask_terms["loss"]),
                    "train_mask_bce": scalar(mask_terms["bce"]),
                    "train_mask_dice": scalar(mask_terms["dice"]),
                    **val_metrics,
                    "learning_rate": float(config["training"].get("learning_rate", 1e-4)),
                    "used_baseline_fallback": False,
                }
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                logger.info(
                    "step=%s epoch=%s total=%.6f diffusion=%.6f mask=%.6f val_mask=%.6f fallback=False",
                    row["step"],
                    row["epoch"],
                    row["train_total_loss"],
                    row["train_diffusion_loss"],
                    row["train_mask_loss"],
                    row["val_mask_loss"],
                )
                step += 1
                if step >= max_steps:
                    break
            epoch += 1

    controlnet_path = paths.checkpoint_dir / "controlnet"
    if bool(config["training"].get("save_controlnet", True)):
        controlnet.save_pretrained(controlnet_path)
    mask_head_path = paths.checkpoint_dir / "aux_mask_head.pt"
    torch.save(
        {
            "model_state_dict": mask_head.state_dict(),
            "hidden_channels": int(config["sd_controlnet"].get("mask_head_channels", 16)),
            "step": max_steps,
        },
        mask_head_path,
    )
    config_copy = paths.output_dir / "config_used.yaml"
    shutil.copy2(resolve_project_path(args.config), config_copy)
    final_row = rows[-1] if rows else {}
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
            "batch_size": batch_size,
            "learning_rate": float(config["training"].get("learning_rate", 1e-4)),
            "prompt": prompt,
            "elapsed_seconds": time.time() - start,
            "final_metrics": final_row,
        },
        "artifacts": {
            "checkpoint_dir": str(paths.checkpoint_dir),
            "controlnet": str(controlnet_path),
            "aux_mask_head": str(mask_head_path),
            "loss_curve": str(paths.loss_curve),
            "log": str(paths.log_file),
            "config": str(config_copy),
        },
    }
    paths.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps
    if args.subset_train_count is not None:
        config.setdefault("data", {})["subset_train_count"] = args.subset_train_count
    output_dir = args.output_dir or config.get("experiment", {}).get("output_dir", "output/training/sd_controlnet_coco_pretrain")
    paths = make_paths(output_dir)
    logger = configure_logging("local_deblur.sd_controlnet_coco", paths.log_file)
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
