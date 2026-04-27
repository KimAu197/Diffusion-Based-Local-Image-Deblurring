#!/usr/bin/env python
"""Convert a local ReLoBlur directory into project manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_deblur.data.transforms import load_mask, load_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Path containing ReLoBlur dataset/train and dataset/test.")
    parser.add_argument("--masks-root", required=True, help="Path containing ReLoBlur masks/train and masks/test.")
    parser.add_argument("--output-dir", default="output/datasets/reloblur")
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--validate-images", action="store_true")
    return parser.parse_args()


def build_records(dataset_root: Path, masks_root: Path, split: str, *, validate_images: bool) -> list[dict]:
    sharp_files = sorted((dataset_root / split).glob("*/*/*sharp.png"))
    records: list[dict] = []
    for sharp_path in sharp_files:
        blur_path = Path(str(sharp_path).replace("sharp", "blur"))
        rel = sharp_path.relative_to(dataset_root)
        mask_rel = Path(str(rel).replace("_sharp", ""))
        mask_path = masks_root / mask_rel
        if not blur_path.exists() or not mask_path.exists():
            continue
        if validate_images:
            blurred = load_rgb(blur_path)
            target = load_rgb(sharp_path)
            mask = load_mask(mask_path)
            if blurred.size != target.size or blurred.size != mask.size:
                raise ValueError(f"Size mismatch for {sharp_path}: {blurred.size}, {target.size}, {mask.size}")
        sample_id = str(rel).replace("/", "_").replace("_sharp.png", "")
        records.append(
            {
                "sample_id": sample_id,
                "Ib": str(blur_path.resolve()),
                "M": str(mask_path.resolve()),
                "target": str(sharp_path.resolve()),
                "S": str(mask_path.resolve()),
                "metadata": {
                    "source": "ReLoBlur",
                    "split": split,
                    "relative_path": str(rel),
                },
            }
        )
    return records


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    masks_root = Path(args.masks_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "test"] if args.split == "all" else [args.split]
    summary: dict[str, int] = {}
    for split in splits:
        records = build_records(dataset_root, masks_root, split, validate_images=args.validate_images)
        manifest = output_dir / f"{split}_manifest.json"
        manifest.write_text(json.dumps({"samples": records}, indent=2), encoding="utf-8")
        summary[split] = len(records)
        print(f"{split}: {len(records)} samples -> {manifest}")
    (output_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
