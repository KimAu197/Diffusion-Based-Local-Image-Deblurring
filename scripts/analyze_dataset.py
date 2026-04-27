#!/usr/bin/env python
"""Summarize a local deblurring manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--black-border-check", type=int, default=100)
    return parser.parse_args()


def has_black_border(path: str, threshold: int = 8, max_fraction: float = 0.25) -> bool:
    image = np.asarray(Image.open(path).convert("RGB"))
    strips = [image[:8], image[-8:], image[:, :8], image[:, -8:]]
    fraction = sum(float((strip.mean(axis=-1) < threshold).mean()) for strip in strips) / len(strips)
    return fraction > max_fraction


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    samples = payload["samples"]
    blur_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    mask_means: list[float] = []
    grouped = 0
    black_border_failures = 0
    for index, sample in enumerate(samples):
        metadata = sample.get("metadata", {})
        blur_counts[str(metadata.get("blur_variant", "unknown"))] += 1
        for category in metadata.get("categories", []):
            category_counts[str(category)] += 1
        if metadata.get("grouped_motion_object"):
            grouped += 1
        if "mask_mean" in metadata:
            mask_means.append(float(metadata["mask_mean"]))
        if index < args.black_border_check and has_black_border(sample["target"]):
            black_border_failures += 1

    stats = {
        "manifest": str(manifest.resolve()),
        "samples": len(samples),
        "grouped_motion_object_samples": grouped,
        "blur_counts": dict(blur_counts),
        "top_categories": category_counts.most_common(20),
        "mask_mean_min": min(mask_means) if mask_means else None,
        "mask_mean_max": max(mask_means) if mask_means else None,
        "mask_mean_avg": sum(mask_means) / len(mask_means) if mask_means else None,
        "black_border_failures_checked": black_border_failures,
        "black_border_checked_samples": min(args.black_border_check, len(samples)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
