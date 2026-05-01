"""Stage 1 standalone blur-mask prediction training."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from local_deblur.data.datasets import ManifestDeblurDataset
from local_deblur.data.tensor_dataset import TensorManifestDeblurDataset, deterministic_split_indices
from local_deblur.data.transforms import load_mask, mask_to_array
from local_deblur.logging_utils import configure_logging
from local_deblur.models.mask_predictor import MaskPredictorUNet
from local_deblur.paths import resolve_project_path
from local_deblur.training.losses import dice_loss_from_logits


@dataclass
class MaskStage1Result:
    output_dir: Path
    checkpoint_path: Path
    summary_path: Path
    metrics: dict[str, float]


def unique_output_dir(path: str | Path) -> Path:
    base = resolve_project_path(path)
    if not base.exists():
        base.mkdir(parents=True)
        return base
    if not any(base.iterdir()):
        return base
    suffix = datetime.now().strftime("%H%M")
    candidate = base.with_name(f"{base.name}_{suffix}")
    index = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{suffix}_{index}")
        index += 1
    candidate.mkdir(parents=True)
    return candidate


def resource_snapshot() -> dict[str, Any]:
    import torch

    snapshot: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "devices": [],
        "nvidia_smi": None,
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            props = torch.cuda.get_device_properties(idx)
            snapshot["devices"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_gb": round(total_bytes / 1024**3, 3),
                    "free_gb": round(free_bytes / 1024**3, 3),
                    "allocated_gb": round(torch.cuda.memory_allocated(idx) / 1024**3, 3),
                }
            )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        snapshot["nvidia_smi"] = completed.stdout.strip() or completed.stderr.strip()
    except Exception as exc:
        snapshot["nvidia_smi"] = f"unavailable: {exc}"
    return snapshot


def recommended_run_parameters(snapshot: dict[str, Any], requested: dict[str, Any]) -> dict[str, Any]:
    free_gb = 0.0
    if snapshot.get("devices"):
        free_gb = max(float(device.get("free_gb", 0.0)) for device in snapshot["devices"])
    proceed = bool(snapshot.get("cuda_available")) and free_gb >= 4.0
    return {
        "requested_image_size": requested.get("image_size"),
        "requested_batch_size": requested.get("batch_size"),
        "recommended_device": "cuda" if proceed else "cpu",
        "recommended_batch_size": requested.get("batch_size") if proceed else min(int(requested.get("batch_size", 1)), 2),
        "proceed": proceed or not snapshot.get("cuda_available"),
        "reason": "sufficient GPU memory" if proceed else "GPU unavailable or low free memory; use smaller CPU/GPU run",
    }


def _sobel_edges(mask: Any):
    import torch
    import torch.nn.functional as F

    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=mask.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=mask.device).view(1, 1, 3, 3)
    gx = F.conv2d(mask, kernel_x, padding=1)
    gy = F.conv2d(mask, kernel_y, padding=1)
    return (gx.square() + gy.square() + 1e-8).sqrt()


def boundary_loss(mask_prob: Any, target_mask: Any):
    import torch.nn.functional as F

    pred_edges = _sobel_edges(mask_prob)
    target_edges = _sobel_edges(target_mask)
    return F.l1_loss(pred_edges, target_edges)


def _per_sample_normalize(value: Any):
    dims = tuple(range(1, value.ndim))
    min_value = value.amin(dim=dims, keepdim=True)
    max_value = value.amax(dim=dims, keepdim=True)
    return (value - min_value) / (max_value - min_value + 1e-8)


def build_mask_predictor_input(Ib: Any, mode: str):
    import torch
    import torch.nn.functional as F

    if mode == "rgb":
        return Ib
    if mode != "rgb_blur_cues":
        raise ValueError(f"Unsupported Stage 1 input feature mode: {mode}")
    gray = Ib.mean(dim=1, keepdim=True)
    low_pass = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
    detail = (gray - low_pass).abs()
    inverse_detail = 1.0 - _per_sample_normalize(detail)
    edge = _per_sample_normalize(_sobel_edges(gray))
    return torch.cat([Ib, inverse_detail, edge], dim=1)


def batch_mask_metrics(mask_prob: Any, mask_logits: Any, target_mask: Any, threshold: float = 0.5) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    pred = mask_prob >= threshold
    target = target_mask >= threshold
    intersection = (pred & target).sum(dim=(1, 2, 3)).float()
    union = (pred | target).sum(dim=(1, 2, 3)).float()
    pred_count = pred.sum(dim=(1, 2, 3)).float()
    target_count = target.sum(dim=(1, 2, 3)).float()
    iou = (intersection / (union + 1e-8)).mean()
    dice = ((2.0 * intersection) / (pred_count + target_count + 1e-8)).mean()
    bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)

    pred_edges = _sobel_edges(pred.float()) > 0.1
    target_edges = _sobel_edges(target.float()) > 0.1
    edge_intersection = (pred_edges & target_edges).sum(dim=(1, 2, 3)).float()
    edge_union = (pred_edges | target_edges).sum(dim=(1, 2, 3)).float()
    boundary_iou = (edge_intersection / (edge_union + 1e-8)).mean()
    return {
        "mask_iou": float(iou.detach().cpu()),
        "mask_dice": float(dice.detach().cpu()),
        "mask_bce": float(bce.detach().cpu()),
        "boundary_iou": float(boundary_iou.detach().cpu()),
        "pred_mask_mean": float(mask_prob.mean().detach().cpu()),
        "target_mask_mean": float(target_mask.mean().detach().cpu()),
        "pred_positive_fraction": float(pred.float().mean().detach().cpu()),
        "target_positive_fraction": float(target.float().mean().detach().cpu()),
    }


def _tensor_to_pil_rgb(tensor: Any) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")


def _tensor_to_pil_mask(tensor: Any) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 1).squeeze(0).numpy()
    return Image.fromarray((array * 255.0).astype(np.uint8), mode="L").convert("RGB")


def _overlay_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    overlay = rgb.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] * (1.0 - 0.45 * m) + 255.0 * 0.45 * m, 0, 255)
    overlay[..., 1] = np.clip(overlay[..., 1] * (1.0 - 0.35 * m), 0, 255)
    overlay[..., 2] = np.clip(overlay[..., 2] * (1.0 - 0.35 * m), 0, 255)
    return Image.fromarray(overlay.astype(np.uint8), mode="RGB")


def save_validation_grids(batch: dict[str, Any], mask_prob: Any, output_dir: Path, *, max_images: int = 4) -> list[str]:
    saved: list[str] = []
    log_dir = output_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = batch["sample_id"]
    if isinstance(sample_ids, str):
        sample_ids = [sample_ids]
    count = min(max_images, mask_prob.shape[0])
    for idx in range(count):
        image = _tensor_to_pil_rgb(batch["Ib"][idx])
        gt = _tensor_to_pil_mask(batch["M"][idx])
        prob = _tensor_to_pil_mask(mask_prob[idx])
        binary = _tensor_to_pil_mask((mask_prob[idx] >= 0.5).float())
        overlay = _overlay_mask(image, binary.convert("L"))
        grid = Image.new("RGB", (image.width * 5, image.height))
        for col, tile in enumerate([image, gt, prob, binary, overlay]):
            grid.paste(tile, (col * image.width, 0))
        safe_id = str(sample_ids[idx]).replace("/", "-")
        path = log_dir / f"{safe_id}_mask_grid.png"
        grid.save(path)
        saved.append(str(path))
    return saved


def validate_mask_manifest(manifest_path: Path) -> dict[str, Any]:
    dataset = ManifestDeblurDataset(manifest_path)
    mask_means: list[float] = []
    empty_ids: list[str] = []
    for record in dataset.records:
        if record.mask_path is None:
            empty_ids.append(record.sample_id)
            continue
        mean = float(mask_to_array(load_mask(record.mask_path)).mean())
        mask_means.append(mean)
        if mean <= 0.0:
            empty_ids.append(record.sample_id)
    if empty_ids:
        raise ValueError(f"Found empty or missing masks in Stage 1 data: {empty_ids[:10]}")
    return {
        "samples_checked": len(mask_means),
        "mask_mean_min": min(mask_means) if mask_means else 0.0,
        "mask_mean_max": max(mask_means) if mask_means else 0.0,
        "mask_mean_avg": sum(mask_means) / max(1, len(mask_means)),
        "empty_masks": len(empty_ids),
    }


def evaluate_mask_model(
    model: Any,
    loader: Any,
    device: Any,
    *,
    input_features: str,
    max_batches: int = 0,
    output_dir: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], list[str]]:
    import torch

    model.eval()
    totals: dict[str, float] = {}
    predictions: list[dict[str, Any]] = []
    grids: list[str] = []
    batches = 0
    with torch.no_grad():
        for batch in loader:
            Ib = batch["Ib"].to(device, non_blocking=True)
            M = batch["M"].to(device, non_blocking=True)
            output = model(build_mask_predictor_input(Ib, input_features))
            metrics = batch_mask_metrics(output.mask_prob, output.mask_logits, M)
            batch_size = Ib.shape[0]
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            sample_ids = batch["sample_id"]
            if isinstance(sample_ids, str):
                sample_ids = [sample_ids]
            for idx, sample_id in enumerate(sample_ids):
                sample_metrics = batch_mask_metrics(output.mask_prob[idx : idx + 1], output.mask_logits[idx : idx + 1], M[idx : idx + 1])
                predictions.append({"sample_id": str(sample_id), **sample_metrics})
            if output_dir is not None and not grids:
                cpu_batch = {"sample_id": sample_ids, "Ib": batch["Ib"].cpu(), "M": batch["M"].cpu()}
                grids.extend(save_validation_grids(cpu_batch, output.mask_prob.cpu(), output_dir))
            batches += 1
            if max_batches and batches >= max_batches:
                break
    model.train()
    total_samples = max(1, len(predictions))
    return {key: value / total_samples for key, value in totals.items()}, predictions, grids


def run_mask_stage1(config: dict[str, Any]) -> MaskStage1Result:
    import torch
    from torch.utils.data import DataLoader
    from torch.nn import functional as F

    training_config = config.get("training", {})
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    loss_config = config.get("losses", {})

    train_manifest = resolve_project_path(data_config.get("train_manifest") or data_config.get("manifest"))
    val_manifest = resolve_project_path(data_config.get("val_manifest")) if data_config.get("val_manifest") else None
    image_size = int(training_config.get("image_size", 256))
    split_seed = int(training_config.get("split_seed", 2026))
    val_fraction = float(training_config.get("val_fraction", 0.1))
    batch_size = int(training_config.get("batch_size", 4))
    num_workers = int(training_config.get("num_workers", 0))
    max_steps = int(training_config.get("max_steps", 50))
    val_interval = max(1, int(training_config.get("validation_interval", 10)))
    val_max_batches = int(training_config.get("validation_batches", 0))
    seed = int(training_config.get("seed", 42))
    output_dir = unique_output_dir(training_config.get("output_dir", "output/training/mask_stage1"))
    logger = configure_logging("local_deblur.mask_stage1", output_dir / "logging.log")

    resource_before = resource_snapshot()
    recommendation = recommended_run_parameters(resource_before, {"image_size": image_size, "batch_size": batch_size})
    logger.info("resource_before=%s", json.dumps(resource_before, ensure_ascii=True))
    logger.info("resource_recommendation=%s", json.dumps(recommendation, ensure_ascii=True))

    torch.manual_seed(seed)
    if val_manifest is not None:
        full_dataset = None
        train_dataset = TensorManifestDeblurDataset(train_manifest, image_size=image_size, include_segmentation=False)
        val_dataset = TensorManifestDeblurDataset(val_manifest, image_size=image_size, include_segmentation=False)
        data_audit = {
            "train": validate_mask_manifest(train_manifest),
            "val": validate_mask_manifest(val_manifest),
        }
        split_source = "provided_train_val_manifests"
    else:
        full_dataset = TensorManifestDeblurDataset(train_manifest, image_size=image_size, include_segmentation=False)
        train_indices, val_indices = deterministic_split_indices(len(full_dataset), val_fraction=val_fraction, seed=split_seed)
        train_dataset = TensorManifestDeblurDataset(train_manifest, image_size=image_size, include_segmentation=False, indices=train_indices)
        val_dataset = TensorManifestDeblurDataset(train_manifest, image_size=image_size, include_segmentation=False, indices=val_indices)
        data_audit = {"combined": validate_mask_manifest(train_manifest)}
        split_source = "deterministic_split"
    split_metadata = {
        "manifest": str(train_manifest),
        "train_manifest": str(train_manifest),
        "val_manifest": None if val_manifest is None else str(val_manifest),
        "split_source": split_source,
        "total_samples": len(train_dataset) + len(val_dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "split_seed": split_seed,
        "val_fraction": val_fraction,
        "image_size": image_size,
        "data_audit": data_audit,
    }
    logger.info("split_metadata=%s", json.dumps(split_metadata, ensure_ascii=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    base_channels = int(model_config.get("base_channels", 16))
    input_features = str(model_config.get("input_features", "rgb"))
    in_channels = 5 if input_features == "rgb_blur_cues" else 3
    model = MaskPredictorUNet(in_channels=in_channels, base_channels=base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 5e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    bce_weight = float(loss_config.get("bce_weight", 1.0))
    dice_weight = float(loss_config.get("dice_weight", 1.0))
    boundary_weight = float(loss_config.get("boundary_weight", 0.1))
    positive_weight = float(loss_config.get("positive_weight", 4.0))
    pos_weight = torch.tensor([positive_weight], device=device)

    curve_path = output_dir / "loss_curve.csv"
    best_checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"
    best_iou = -1.0
    final_metrics: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    model.train()
    step = 0
    epoch = 1
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "step",
            "epoch",
            "train_loss",
            "train_bce",
            "train_dice_loss",
            "train_boundary_loss",
            "val_mask_iou",
            "val_mask_dice",
            "val_mask_bce",
            "val_boundary_iou",
            "learning_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        while step < max_steps:
            for batch in train_loader:
                Ib = batch["Ib"].to(device, non_blocking=True)
                M = batch["M"].to(device, non_blocking=True)
                output = model(build_mask_predictor_input(Ib, input_features))
                bce = F.binary_cross_entropy_with_logits(output.mask_logits, M, pos_weight=pos_weight)
                dice = dice_loss_from_logits(output.mask_logits, M)
                edge = boundary_loss(output.mask_prob, M)
                loss = bce_weight * bce + dice_weight * dice + boundary_weight * edge
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                step += 1
                should_validate = step % val_interval == 0 or step == max_steps
                val_metrics = None
                if should_validate:
                    val_metrics, _, _ = evaluate_mask_model(
                        model,
                        val_loader,
                        device,
                        input_features=input_features,
                        max_batches=val_max_batches,
                    )
                    final_metrics = val_metrics
                    if val_metrics["mask_iou"] > best_iou:
                        best_iou = val_metrics["mask_iou"]
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "step": step,
                                "epoch": epoch,
                                "model": "MaskPredictorUNet",
                                "base_channels": base_channels,
                                "in_channels": in_channels,
                                "input_features": input_features,
                                "split": split_metadata,
                                "config": config,
                                "best_mask_iou": best_iou,
                                "input_contract": "Ib only",
                            },
                            best_checkpoint_path,
                        )
                row = {
                    "step": step,
                    "epoch": epoch,
                    "train_loss": float(loss.detach().cpu()),
                    "train_bce": float(bce.detach().cpu()),
                    "train_dice_loss": float(dice.detach().cpu()),
                    "train_boundary_loss": float(edge.detach().cpu()),
                    "val_mask_iou": "" if val_metrics is None else val_metrics["mask_iou"],
                    "val_mask_dice": "" if val_metrics is None else val_metrics["mask_dice"],
                    "val_mask_bce": "" if val_metrics is None else val_metrics["mask_bce"],
                    "val_boundary_iou": "" if val_metrics is None else val_metrics["boundary_iou"],
                    "learning_rate": float(training_config.get("learning_rate", 5e-4)),
                }
                writer.writerow(row)
                handle.flush()
                rows.append(row)
                logger.info(
                    "step=%s epoch=%s train_loss=%.6f val_iou=%s",
                    step,
                    epoch,
                    row["train_loss"],
                    "NA" if val_metrics is None else f"{val_metrics['mask_iou']:.6f}",
                )
                if step >= max_steps:
                    break
            epoch += 1

    final_metrics, predictions, grids = evaluate_mask_model(
        model,
        val_loader,
        device,
        input_features=input_features,
        max_batches=0,
        output_dir=output_dir,
    )
    resource_after = resource_snapshot()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": max_steps,
            "epoch": epoch,
            "model": "MaskPredictorUNet",
            "base_channels": base_channels,
            "in_channels": in_channels,
            "input_features": input_features,
            "split": split_metadata,
            "config": config,
            "final_metrics": final_metrics,
            "input_contract": "Ib only",
        },
        last_checkpoint_path,
    )
    if not best_checkpoint_path.exists():
        torch.save(torch.load(last_checkpoint_path, map_location="cpu"), best_checkpoint_path)
        best_iou = final_metrics["mask_iou"]

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in final_metrics.items():
            writer.writerow({"metric": key, "value": value})

    answer_path = output_dir / "answer.json"
    answer_path.write_text(
        json.dumps(
            {
                "predictions": predictions,
                "visual_grids": grids,
                "threshold": 0.5,
                "input_contract": "Ib only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        "model": "MaskPredictorUNet",
        "input_contract": "Ib only",
        "input_features": input_features,
        "target": "M",
        "stage": "stage1_mask_prediction",
        "achieved_iou_target_0_8": final_metrics.get("mask_iou", 0.0) >= 0.8,
        "final_metrics": final_metrics,
        "best_mask_iou": best_iou,
        "output_dir": str(output_dir),
        "checkpoints": {"best": str(best_checkpoint_path), "last": str(last_checkpoint_path)},
        "loss_curve": str(curve_path),
        "metrics_csv": str(metrics_path),
        "answer_json": str(answer_path),
        "split": split_metadata,
        "resource_before": resource_before,
        "resource_after": resource_after,
        "resource_recommendation": recommendation,
        "visual_grids": grids,
        "constraint_check": "Model receives only Ib and never receives GT M as input.",
    }
    training_summary_path = output_dir / "training_summary.json"
    training_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_txt = output_dir / "summary.txt"
    summary_txt.write_text(
        "\n".join(
            [
                "Stage 1 Mask Predictor Summary",
                f"Model: MaskPredictorUNet",
                f"Input: Ib only",
                f"Input features: {input_features}",
                f"Train manifest: {train_manifest}",
                f"Val manifest: {val_manifest}",
                f"Samples: total={len(train_dataset) + len(val_dataset)} train={len(train_dataset)} val={len(val_dataset)}",
                f"Mask IoU: {final_metrics.get('mask_iou', 0.0):.6f}",
                f"Mask Dice: {final_metrics.get('mask_dice', 0.0):.6f}",
                f"Mask BCE: {final_metrics.get('mask_bce', 0.0):.6f}",
                f"Boundary IoU: {final_metrics.get('boundary_iou', 0.0):.6f}",
                f"IoU target > 0.8 achieved: {summary['achieved_iou_target_0_8']}",
                "Constraint check: Model receives only Ib and never receives GT M as input.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return MaskStage1Result(output_dir=output_dir, checkpoint_path=last_checkpoint_path, summary_path=summary_txt, metrics=final_metrics)
