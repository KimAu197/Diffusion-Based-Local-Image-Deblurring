#!/usr/bin/env python
"""Run local deblurring inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_deblur.config import load_yaml_config
from local_deblur.inference import run_inference, run_sd_controlnet_inference
from local_deblur.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=None, help="Blurred image path (Ib).")
    parser.add_argument("--mask", default=None, help="Optional blur mask path for fallback/compatibility; SD + ControlNet inference predicts its own mask.")
    parser.add_argument("--segmentation", default=None, help="Optional segmentation map path (S).")
    parser.add_argument("--checkpoint", default=None, help="Trained ControlNet checkpoint path; kept as an alias for --controlnet-checkpoint.")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--mask-output", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Use the lightweight fallback path instead of SD + ControlNet.")
    parser.add_argument("--base-sd-checkpoint", default=None)
    parser.add_argument("--controlnet-checkpoint", default=None)
    parser.add_argument("--mask-head-checkpoint", default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--no-background-preserve", action="store_true")
    return parser.parse_args()


def _get(config: dict, section: str, key: str, default=None):
    value = config.get(section, {}).get(key, default)
    return default if value is None else value


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    image = args.image or _get(config, "inputs", "image")
    mask = args.mask or _get(config, "inputs", "mask")
    segmentation = args.segmentation or _get(config, "inputs", "segmentation")
    output_path = resolve_project_path(args.output or _get(config, "inference", "output", "output/inference/sd_controlnet_output.png"))
    mask_output_value = args.mask_output if args.mask_output is not None else _get(config, "inference", "mask_output")
    mask_output_path = resolve_project_path(mask_output_value) if mask_output_value else None
    dry_run = bool(args.dry_run or _get(config, "inference", "dry_run", False))

    if dry_run:
        output, mask_output = run_inference(
            image_path=image,
            mask_path=mask,
            segmentation_path=segmentation,
            checkpoint=args.checkpoint or _get(config, "inputs", "checkpoint"),
            output_path=output_path,
            mask_output_path=mask_output_path,
            dry_run=True,
        )
        mode = "fallback"
    else:
        sd_config = dict(config.get("sd_controlnet", {}))
        sd_config["base_sd_checkpoint"] = args.base_sd_checkpoint or sd_config.get("base_sd_checkpoint")
        sd_config["controlnet_checkpoint"] = args.controlnet_checkpoint or args.checkpoint or sd_config.get("controlnet_checkpoint")
        sd_config["mask_head_checkpoint"] = args.mask_head_checkpoint or sd_config.get("mask_head_checkpoint")
        if args.device:
            sd_config["device"] = args.device
        if args.dtype:
            sd_config["dtype"] = args.dtype
            sd_config["precision"] = args.dtype
        if args.image_size:
            sd_config["image_size"] = args.image_size

        missing = [key for key in ("base_sd_checkpoint", "controlnet_checkpoint") if not sd_config.get(key)]
        if missing:
            raise SystemExit(f"SD + ControlNet inference missing required config values: {', '.join(missing)}")

        output, mask_output = run_sd_controlnet_inference(
            image_path=image,
            mask_path=mask,
            segmentation_path=segmentation,
            output_path=output_path,
            mask_output_path=mask_output_path,
            sd_controlnet_config=sd_config,
            prompt=args.prompt or _get(config, "inference", "prompt", "local deblur restoration"),
            negative_prompt=args.negative_prompt if args.negative_prompt is not None else _get(config, "inference", "negative_prompt"),
            num_inference_steps=args.num_inference_steps or int(_get(config, "inference", "num_inference_steps", 50)),
            guidance_scale=args.guidance_scale if args.guidance_scale is not None else float(_get(config, "inference", "guidance_scale", 7.5)),
            strength=args.strength if args.strength is not None else float(_get(config, "inference", "strength", 0.8)),
            seed=args.seed if args.seed is not None else _get(config, "inference", "seed"),
            preserve_background=not args.no_background_preserve and bool(_get(config, "runtime", "preserve_background", True)),
        )
        mode = "sd_controlnet"

    print(f"Inference mode: {mode}")
    print(f"Wrote inference output: {output}")
    if mask_output:
        print(f"Wrote predicted blur mask: {mask_output}")


if __name__ == "__main__":
    main()
