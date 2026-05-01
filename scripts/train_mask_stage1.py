#!/usr/bin/env python
"""Train the Stage 1 standalone Ib-to-mask predictor."""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_deblur.config import deep_merge, load_yaml_config
from local_deblur.paths import PROJECT_ROOT
from local_deblur.training.mask_stage1 import run_mask_stage1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_mask_stage1.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Use a tiny dry output directory and very few steps.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = load_yaml_config(args.config)
    overrides: dict = {"training": {}, "data": {}}
    if args.manifest is not None:
        overrides["data"]["manifest"] = args.manifest
    if args.train_manifest is not None:
        overrides["data"]["train_manifest"] = args.train_manifest
    if args.val_manifest is not None:
        overrides["data"]["val_manifest"] = args.val_manifest
    if args.output_dir is not None:
        overrides["training"]["output_dir"] = args.output_dir
    if args.max_steps is not None:
        overrides["training"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        overrides["training"]["batch_size"] = args.batch_size
    if args.image_size is not None:
        overrides["training"]["image_size"] = args.image_size
    if args.learning_rate is not None:
        overrides["training"]["learning_rate"] = args.learning_rate
    if args.seed is not None:
        overrides["training"]["seed"] = args.seed
    if args.dry_run:
        overrides["training"]["output_dir"] = "output/training/dry_mask_stage1"
        overrides["training"]["max_steps"] = min(int(config.get("training", {}).get("max_steps", 2)), 2)
        overrides["training"]["validation_interval"] = 1
        overrides["training"]["validation_batches"] = 1
    return deep_merge(config, overrides)


def move_incomplete(output_dir: Path | None, reason: str) -> None:
    if output_dir is None or not output_dir.exists():
        return
    incomplete_root = PROJECT_ROOT / "incomplete"
    incomplete_root.mkdir(parents=True, exist_ok=True)
    target = incomplete_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_mask_stage1"
    shutil.move(str(output_dir), str(target))
    print(f"Moved incomplete Stage 1 artifacts to {target} because {reason}")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    output_dir: Path | None = None
    try:
        configured_output = config.get("training", {}).get("output_dir")
        output_dir = PROJECT_ROOT / configured_output if configured_output and not Path(configured_output).is_absolute() else Path(configured_output)
        result = run_mask_stage1(config)
    except Exception as exc:
        move_incomplete(output_dir, str(exc))
        raise
    print(f"Stage 1 mask training complete: {result.output_dir}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Summary: {result.summary_path}")
    print(f"Mask IoU: {result.metrics.get('mask_iou', 0.0):.6f}")


if __name__ == "__main__":
    main()
