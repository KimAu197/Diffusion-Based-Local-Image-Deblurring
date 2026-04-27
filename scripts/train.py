#!/usr/bin/env python
"""Smoke-safe training entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_deblur.config import load_yaml_config
from local_deblur.paths import resolve_project_path
from local_deblur.training.trainer import LocalDeblurTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--config", default="configs/train_pretrain.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trainable-baseline", action="store_true", help="Train compact PyTorch ConditionalLocalDeblurNet.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if not args.dry_run and not args.trainable_baseline:
        raise SystemExit("Full training requires confirmation from the main agent/user. Use --dry-run for smoke validation.")
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    trainable_config = config.get("trainable_baseline", {})
    image_size = int(training_config.get("image_size", 512))
    batch_size = int(args.batch_size or trainable_config.get("batch_size", 4))
    learning_rate = float(args.learning_rate or trainable_config.get("learning_rate", 1e-3))
    base_channels = int(trainable_config.get("base_channels", 32))
    max_steps = int(args.max_steps or training_config.get("max_steps", 1))
    seed = int(args.seed if args.seed is not None else training_config.get("seed", 42))
    split_seed = int(args.split_seed if args.split_seed is not None else training_config.get("split_seed", seed))
    val_fraction = float(args.val_fraction if args.val_fraction is not None else training_config.get("val_fraction", 0.1))
    validation_batches = int(training_config.get("validation_batches", 4))
    num_workers = int(training_config.get("num_workers", 0))
    manifest = args.manifest or data_config.get("synthetic_manifest") or data_config.get("manifest")
    output_dir = resolve_project_path(args.output_dir or training_config.get("output_dir", "output/training"))
    trainer = LocalDeblurTrainer(
        phase=str(training_config.get("phase", args.phase)),
        manifest=manifest,
        output_dir=str(output_dir),
        dry_run=args.dry_run,
        image_size=image_size,
        trainable_baseline=args.trainable_baseline,
        batch_size=batch_size,
        learning_rate=learning_rate,
        base_channels=base_channels,
        val_fraction=val_fraction,
        split_seed=split_seed,
        validation_batches=validation_batches,
        num_workers=num_workers,
        loss_weights=config.get("losses", {}),
    )
    result = trainer.run(max_steps=max_steps, seed=seed)
    mode = "Trainable baseline" if args.trainable_baseline else "Dry-run"
    print(f"{mode} training complete: steps={result.steps} final_loss={result.final_loss:.6f}")
    print(f"Checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
