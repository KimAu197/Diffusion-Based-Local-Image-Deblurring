#!/usr/bin/env python
"""Run local deblurring inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_deblur.config import load_yaml_config
from local_deblur.inference import run_inference
from local_deblur.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="Blurred image path (Ib).")
    parser.add_argument("--mask", default=None, help="Blur mask path (M).")
    parser.add_argument("--segmentation", default=None, help="Optional segmentation map path (S).")
    parser.add_argument("--checkpoint", default=None, help="Optional local checkpoint path.")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--output", default="output/inference/dry_run_output.png")
    parser.add_argument("--mask-output", default="output/inference/dry_run_predicted_mask.png")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_yaml_config(args.config)
    if not args.dry_run and args.checkpoint is None:
        raise SystemExit("Full inference requires a local checkpoint and user confirmation. Use --dry-run for smoke validation.")
    output, mask_output = run_inference(
        image_path=args.image,
        mask_path=args.mask,
        segmentation_path=args.segmentation,
        checkpoint=args.checkpoint,
        output_path=resolve_project_path(args.output),
        mask_output_path=resolve_project_path(args.mask_output) if args.mask_output else None,
        dry_run=args.dry_run,
    )
    print(f"Wrote inference output: {output}")
    if mask_output:
        print(f"Wrote predicted blur mask: {mask_output}")


if __name__ == "__main__":
    main()
