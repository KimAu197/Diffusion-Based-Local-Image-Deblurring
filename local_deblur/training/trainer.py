"""Minimal trainer wiring data, pipeline, losses, and checkpoint metadata."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from local_deblur.data.datasets import load_dataset
from local_deblur.logging_utils import configure_logging
from local_deblur.models.pipeline import LocalDeblurPipeline
from local_deblur.paths import ensure_directory

from .losses import binary_cross_entropy_mask, charbonnier_loss, combined_trainable_loss, masked_l1


@dataclass
class TrainingResult:
    output_dir: Path
    checkpoint_path: Path
    steps: int
    final_loss: float


class LocalDeblurTrainer:
    def __init__(
        self,
        *,
        phase: str,
        manifest: str | None = None,
        output_dir: str = "output/training",
        dry_run: bool = True,
        image_size: int = 512,
        trainable_baseline: bool = False,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
        base_channels: int = 32,
        val_fraction: float = 0.1,
        split_seed: int = 42,
        validation_batches: int = 4,
        num_workers: int = 0,
        loss_weights: dict | None = None,
    ):
        if phase not in {"pretrain", "finetune"}:
            raise ValueError("phase must be 'pretrain' or 'finetune'")
        self.phase = phase
        self.output_dir = ensure_directory(output_dir)
        self.dry_run = dry_run
        self.trainable_baseline = trainable_baseline
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.base_channels = base_channels
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.validation_batches = validation_batches
        self.num_workers = num_workers
        self.loss_weights = loss_weights or {}
        self.manifest = manifest
        if trainable_baseline:
            from local_deblur.data.tensor_dataset import TensorManifestDeblurDataset, deterministic_split_indices

            if manifest is None:
                raise ValueError("A manifest path is required for trainable_baseline=True")
            full_dataset = TensorManifestDeblurDataset(manifest, image_size=image_size, include_segmentation=True)
            train_indices, val_indices = deterministic_split_indices(
                len(full_dataset),
                val_fraction=val_fraction,
                seed=split_seed,
            )
            self.dataset = TensorManifestDeblurDataset(
                manifest,
                image_size=image_size,
                include_segmentation=True,
                indices=train_indices,
            )
            self.val_dataset = TensorManifestDeblurDataset(
                manifest,
                image_size=image_size,
                include_segmentation=True,
                indices=val_indices,
            )
            self.split_metadata = {
                "split_seed": split_seed,
                "val_fraction": val_fraction,
                "total_samples": len(full_dataset),
                "train_samples": len(self.dataset),
                "val_samples": len(self.val_dataset),
            }
            self.pipeline = None
        else:
            self.dataset = load_dataset(manifest, dry_run=dry_run and manifest is None, count=1, size=image_size)
            self.val_dataset = None
            self.split_metadata = None
            self.pipeline = LocalDeblurPipeline.load(dry_run=True)
        self.logger = configure_logging("local_deblur.train", self.output_dir / "training.log")

    def run(self, max_steps: int = 1, seed: int = 42) -> TrainingResult:
        if self.trainable_baseline:
            return self._run_trainable(max_steps=max_steps, seed=seed)
        if not self.dry_run:
            raise RuntimeError("Full training requires explicit user confirmation before execution")
        steps = max(1, max_steps)
        final_loss = 0.0
        curve_path = self.output_dir / "loss_curve.csv"
        with curve_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["step", "epoch", "sample_id", "loss", "restoration_loss", "mask_bce", "used_fallback"],
            )
            writer.writeheader()
            for step in range(steps):
                sample_index = step % len(self.dataset)
                sample = self.dataset[sample_index]
                output = self.pipeline(sample)
                target = sample.target or sample.blurred
                restoration_loss = 0.8 * masked_l1(output.image, target, sample.mask) + 0.2 * charbonnier_loss(
                    output.image, target, sample.mask
                )
                mask_loss = binary_cross_entropy_mask(output.predicted_mask, sample.mask) if output.predicted_mask else 0.0
                loss = restoration_loss + 0.1 * mask_loss
                final_loss = loss
                epoch = step // len(self.dataset) + 1
                writer.writerow(
                    {
                        "step": step + 1,
                        "epoch": epoch,
                        "sample_id": sample.sample_id,
                        "loss": loss,
                        "restoration_loss": restoration_loss,
                        "mask_bce": mask_loss,
                        "used_fallback": output.metadata["used_fallback"],
                    }
                )
                self.logger.info(
                    "step=%s epoch=%s phase=%s loss=%.6f restoration_loss=%.6f mask_bce=%.6f fallback=%s",
                    step + 1,
                    epoch,
                    self.phase,
                    loss,
                    restoration_loss,
                    mask_loss,
                    output.metadata["used_fallback"],
                )

        checkpoint_path = self.output_dir / f"{self.phase}_dry_run_checkpoint.json"
        metadata = {
            "phase": self.phase,
            "dry_run": self.dry_run,
            "seed": seed,
            "steps": steps,
            "dataset_size": len(self.dataset),
            "loss_curve": str(curve_path),
            "final_loss": final_loss,
            "mask_head": "enabled",
            "mask_loss_weight": 0.1,
            "note": "Checkpoint-like metadata for smoke validation; no model weights are stored.",
        }
        checkpoint_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return TrainingResult(self.output_dir, checkpoint_path, steps, final_loss)

    def _loss_kwargs(self) -> dict:
        restoration = self.loss_weights.get("restoration", {})
        mask_head = self.loss_weights.get("mask_head", {})
        return {
            "image_weight": float(self.loss_weights.get("image_weight", 1.0)),
            "mask_weight": float(self.loss_weights.get("mask_weight", mask_head.get("weight", 0.1))),
            "restoration_l1_weight": float(restoration.get("masked_l1_weight", 0.8)),
            "restoration_charbonnier_weight": float(restoration.get("charbonnier_weight", 0.2)),
            "mask_bce_weight": float(mask_head.get("bce_weight", 1.0)),
            "mask_dice_weight": float(mask_head.get("dice_weight", 0.5)),
        }

    @staticmethod
    def _scalar_terms(terms: dict) -> dict[str, float]:
        return {key: float(value.detach().cpu()) for key, value in terms.items()}

    @staticmethod
    def _mask_iou(mask_prob, target_mask, threshold: float = 0.5) -> float:
        pred = mask_prob >= threshold
        target = target_mask >= threshold
        intersection = (pred & target).sum().float()
        union = (pred | target).sum().float()
        return float((intersection / (union + 1e-8)).detach().cpu())

    @staticmethod
    def _psnr(restored, target) -> float:
        mse = float(((restored - target) ** 2).mean().detach().cpu())
        if mse <= 0.0:
            return 99.0
        return float(10.0 * math.log10(1.0 / mse))

    def _evaluate_trainable(self, model, loader, device, loss_kwargs: dict) -> dict[str, float]:
        import torch

        model.eval()
        totals: dict[str, float] = {
            "loss": 0.0,
            "restoration_loss": 0.0,
            "mask_loss": 0.0,
            "mask_bce": 0.0,
            "mask_dice": 0.0,
            "psnr": 0.0,
            "mask_iou": 0.0,
        }
        batches = 0
        with torch.no_grad():
            for batch in loader:
                Ib = batch["Ib"].to(device, non_blocking=True)
                M = batch["M"].to(device, non_blocking=True)
                S = batch["S"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                output = model(Ib, M, S)
                terms = combined_trainable_loss(output.restored, target, output.mask_logits, M, **loss_kwargs)
                scalars = self._scalar_terms(terms)
                for key in ("loss", "restoration_loss", "mask_loss", "mask_bce", "mask_dice"):
                    totals[key] += scalars[key]
                totals["psnr"] += self._psnr(output.restored, target)
                totals["mask_iou"] += self._mask_iou(output.mask_prob, M)
                batches += 1
                if batches >= self.validation_batches:
                    break
        model.train()
        if batches == 0:
            return {key: 0.0 for key in totals}
        return {key: value / batches for key, value in totals.items()}

    def _write_loss_plot(self, rows: list[dict], curve_path: Path) -> Path:
        png_path = self.output_dir / "loss_curve.png"
        try:
            import matplotlib.pyplot as plt

            steps = [row["step"] for row in rows]
            train_loss = [row["train_total_loss"] for row in rows]
            val_loss = [row["val_total_loss"] for row in rows if row["val_total_loss"] != ""]
            val_steps = [row["step"] for row in rows if row["val_total_loss"] != ""]
            plt.figure(figsize=(8, 5))
            plt.plot(steps, train_loss, label="train total loss")
            if val_loss:
                plt.plot(val_steps, val_loss, marker="o", label="val total loss")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("ConditionalLocalDeblurNet training")
            plt.legend()
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close()
            return png_path
        except Exception as exc:
            fallback_path = self.output_dir / "loss_curve_plot_unavailable.txt"
            fallback_path.write_text(
                f"matplotlib was unavailable or failed while plotting {curve_path}: {exc}\n",
                encoding="utf-8",
            )
            return fallback_path

    def _run_trainable(self, max_steps: int = 1, seed: int = 42) -> TrainingResult:
        import torch
        from torch.utils.data import DataLoader

        from local_deblur.models.conditional_unet import ConditionalLocalDeblurNet

        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ConditionalLocalDeblurNet(base_channels=self.base_channels, use_segmentation=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        generator = torch.Generator()
        generator.manual_seed(seed)
        pin_memory = device.type == "cuda"
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            generator=generator,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

        steps = max(1, max_steps)
        final_loss = 0.0
        curve_path = self.output_dir / "loss_curve.csv"
        last_checkpoint_path = self.output_dir / "last.pt"
        best_checkpoint_path = self.output_dir / "best.pt"
        loss_kwargs = self._loss_kwargs()
        validation_interval = max(1, min(5, steps))
        best_val_loss = float("inf")
        rows: list[dict] = []
        model.train()
        with curve_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "epoch",
                    "sample_id",
                    "train_total_loss",
                    "train_restoration_loss",
                    "train_mask_loss",
                    "mask_bce",
                    "mask_dice",
                    "val_total_loss",
                    "val_restoration_loss",
                    "val_mask_loss",
                    "val_mask_bce",
                    "val_mask_dice",
                    "val_psnr",
                    "val_mask_iou",
                    "learning_rate",
                    "used_fallback",
                ],
            )
            writer.writeheader()
            step = 0
            epoch = 1
            while step < steps:
                for batch in loader:
                    Ib = batch["Ib"].to(device)
                    M = batch["M"].to(device)
                    S = batch["S"].to(device)
                    target = batch["target"].to(device)

                    output = model(Ib, M, S)
                    terms = combined_trainable_loss(output.restored, target, output.mask_logits, M, **loss_kwargs)
                    optimizer.zero_grad(set_to_none=True)
                    terms["loss"].backward()
                    optimizer.step()

                    scalars = self._scalar_terms(terms)
                    final_loss = scalars["loss"]
                    should_validate = (step + 1) % validation_interval == 0 or (step + 1) == steps
                    val_metrics = (
                        self._evaluate_trainable(model, val_loader, device, loss_kwargs)
                        if should_validate and self.val_dataset is not None
                        else None
                    )
                    if val_metrics is not None and val_metrics["loss"] < best_val_loss:
                        best_val_loss = val_metrics["loss"]
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "step": step + 1,
                                "phase": self.phase,
                                "base_channels": self.base_channels,
                                "split": self.split_metadata,
                                "loss_weights": loss_kwargs,
                                "best_val_loss": best_val_loss,
                            },
                            best_checkpoint_path,
                        )
                    sample_id = batch["sample_id"][0] if isinstance(batch["sample_id"], list) else str(batch["sample_id"])
                    row = {
                        "step": step + 1,
                        "epoch": epoch,
                        "sample_id": sample_id,
                        "train_total_loss": final_loss,
                        "train_restoration_loss": scalars["restoration_loss"],
                        "train_mask_loss": scalars["mask_loss"],
                        "mask_bce": scalars["mask_bce"],
                        "mask_dice": scalars["mask_dice"],
                        "val_total_loss": "" if val_metrics is None else val_metrics["loss"],
                        "val_restoration_loss": "" if val_metrics is None else val_metrics["restoration_loss"],
                        "val_mask_loss": "" if val_metrics is None else val_metrics["mask_loss"],
                        "val_mask_bce": "" if val_metrics is None else val_metrics["mask_bce"],
                        "val_mask_dice": "" if val_metrics is None else val_metrics["mask_dice"],
                        "val_psnr": "" if val_metrics is None else val_metrics["psnr"],
                        "val_mask_iou": "" if val_metrics is None else val_metrics["mask_iou"],
                        "learning_rate": self.learning_rate,
                        "used_fallback": False,
                    }
                    writer.writerow(row)
                    handle.flush()
                    rows.append(row)
                    self.logger.info(
                        "step=%s epoch=%s phase=%s train_loss=%.6f train_restoration=%.6f train_mask=%.6f val_loss=%s val_psnr=%s fallback=False",
                        step + 1,
                        epoch,
                        self.phase,
                        final_loss,
                        scalars["restoration_loss"],
                        scalars["mask_loss"],
                        "NA" if val_metrics is None else f"{val_metrics['loss']:.6f}",
                        "NA" if val_metrics is None else f"{val_metrics['psnr']:.3f}",
                    )
                    step += 1
                    if step >= steps:
                        break
                epoch += 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": steps,
                "phase": self.phase,
                "base_channels": self.base_channels,
                "split": self.split_metadata,
                "loss_weights": loss_kwargs,
                "final_loss": final_loss,
            },
            last_checkpoint_path,
        )
        if not best_checkpoint_path.exists():
            torch.save(torch.load(last_checkpoint_path, map_location="cpu"), best_checkpoint_path)
            best_val_loss = final_loss
        plot_artifact = self._write_loss_plot(rows, curve_path)
        summary_path = self.output_dir / "training_summary.json"
        final_val_metrics = next((row for row in reversed(rows) if row["val_total_loss"] != ""), {})
        summary_path.write_text(
            json.dumps(
                {
                    "phase": self.phase,
                    "dry_run": self.dry_run,
                    "trainable_baseline": True,
                    "seed": seed,
                    "device": str(device),
                    "manifest": self.manifest,
                    "steps": steps,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "base_channels": self.base_channels,
                    "image_size": getattr(self.dataset, "image_size", None),
                    "split": self.split_metadata,
                    "loss_weights": loss_kwargs,
                    "final_train_loss": final_loss,
                    "best_val_loss": best_val_loss,
                    "final_validation": final_val_metrics,
                    "loss_curve": str(curve_path),
                    "loss_curve_artifact": str(plot_artifact),
                    "checkpoints": {"best": str(best_checkpoint_path), "last": str(last_checkpoint_path)},
                    "note": "Image resizing is controlled by config image_size to keep the run budget practical.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        metadata_path = last_checkpoint_path.with_suffix(".json")
        metadata_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
        return TrainingResult(self.output_dir, last_checkpoint_path, steps, final_loss)
