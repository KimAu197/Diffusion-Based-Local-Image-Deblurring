"""Evaluation output directory and serialization helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from local_deblur.data.transforms import array_to_image, image_to_array, save_image
from local_deblur.logging_utils import configure_logging
from local_deblur.paths import dated_result_dir


def create_eval_output(
    round_name: str,
    model: str,
    dataset: str,
    count: int,
    *,
    output_root: str | Path | None = None,
) -> tuple[Path, object]:
    root = dated_result_dir(round_name, model, dataset, count, root=output_root)
    (root / "log").mkdir(parents=True, exist_ok=True)
    logger = configure_logging("local_deblur.eval", root / "logging.log")
    return root, logger


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    metric_names = sorted(rows[0]["metrics"].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", *metric_names])
        writer.writeheader()
        for row in rows:
            writer.writerow({"sample_id": row["sample_id"], **row["metrics"]})


def write_answer_json(path: Path, rows: list[dict]) -> None:
    payload = [
        {
            "sample_id": row["sample_id"],
            "question": "Restore the locally blurred region while preserving the background.",
            "answer": row.get("prediction_path", ""),
            "input": row.get("input_path", ""),
            "target": row.get("target_path", ""),
            "ground_truth_mask": row.get("mask_path", ""),
            "predicted_mask": row.get("predicted_mask_path", ""),
            "visual_grid": row.get("grid_path", ""),
            "metrics": row["metrics"],
            "metadata": row["metadata"],
            "sample_metadata": row.get("sample_metadata", {}),
        }
        for row in rows
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(
    path: Path,
    rows: list[dict],
    *,
    model: str,
    dataset: str,
    dry_run: bool,
    checkpoint: str | None = None,
    split: str = "val",
    manifest: str | None = None,
) -> None:
    averages: dict[str, float] = {}
    if rows:
        for key in rows[0]["metrics"]:
            averages[key] = sum(row["metrics"][key] for row in rows) / len(rows)
    sample_metadata = rows[0].get("sample_metadata", {}) if rows else {}
    first_metadata = rows[0].get("metadata", {}) if rows else {}
    is_sd_controlnet = "ControlNet" in model or first_metadata.get("base_sd_checkpoint")
    is_reloblur = "reloblur" in dataset.lower() or "reloblur" in str(manifest or sample_metadata.get("manifest", "")).lower()
    if is_sd_controlnet:
        interpretation = "Interpretation: trained SD 1.5 + Tile ControlNet local-deblur checkpoint evaluated with real diffusers img2img inference."
        if is_reloblur:
            limitation = "Limitation: sampled ReLoBlur test subset; use count=0 for a full ReLoBlur test-set benchmark."
        else:
            limitation = "Limitation: small synthetic COCO validation subset; these are not ReLoBlur benchmark results."
    else:
        interpretation = "Interpretation: trained ConditionalLocalDeblurNet baseline on the synthetic validation split."
        limitation = "Limitation: these are not full diffusion/ControlNet or ReLoBlur benchmark results."
    title = "ReLoBlur Local Deblur Evaluation Summary" if is_reloblur else "Synthetic Local Deblur Validation Summary"
    lines = [
        title,
        f"model: {model}",
        f"dataset: {dataset}",
        f"manifest: {manifest or sample_metadata.get('manifest', '')}",
        f"split: {split}",
        f"split_seed: {sample_metadata.get('split_seed', '')}",
        f"val_fraction: {sample_metadata.get('val_fraction', '')}",
        f"checkpoint: {checkpoint or ''}",
        f"dry_run: {dry_run}",
        f"samples: {len(rows)}",
        f"image_size: {sample_metadata.get('image_size', '')}",
        interpretation,
        limitation,
        "LBAG reference context: PSNR 34.71 / SSIM 0.967 (not claimed as this run's result).",
        "Mask debug metrics report whether the model identified the blurred region before restoration.",
        "",
        "Averages:",
    ]
    lines.extend(f"{key}: {value:.6f}" for key, value in sorted(averages.items()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _label(image: Image.Image, text: str) -> Image.Image:
    labeled = image.convert("RGB")
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return labeled


def _error_image(prediction: Image.Image, target: Image.Image) -> Image.Image:
    error = np.abs(image_to_array(prediction) - image_to_array(target))
    return array_to_image(np.clip(error * 4.0, 0.0, 1.0))


def _write_grid(path: Path, row: dict) -> Path:
    panels = [
        _label(row["input"], "blurred input"),
        _label(row["mask"].convert("RGB"), "GT mask"),
        _label(row["predicted_mask"].convert("RGB") if row.get("predicted_mask") else Image.new("RGB", row["input"].size), "pred mask"),
        _label(row["prediction"], "restored output"),
        _label(row["target"], "target"),
        _label(_error_image(row["prediction"], row["target"]), "abs error x4"),
    ]
    width, height = panels[0].size
    grid = Image.new("RGB", (width * 3, height * 2), color=(255, 255, 255))
    for idx, panel in enumerate(panels):
        grid.paste(panel, ((idx % 3) * width, (idx // 3) * height))
    return save_image(grid, path)


def write_eval_outputs(
    root: Path,
    rows: list[dict],
    *,
    model: str,
    dataset: str,
    dry_run: bool,
    checkpoint: str | None = None,
    split: str = "val",
    manifest: str | None = None,
    visual_limit: int = 12,
) -> None:
    for row in rows:
        prediction_path = save_image(row["prediction"], root / "log" / f"{row['sample_id']}_prediction.png")
        row["prediction_path"] = str(prediction_path)
        row["input_path"] = str(save_image(row["input"], root / "log" / f"{row['sample_id']}_input.png"))
        row["target_path"] = str(save_image(row["target"], root / "log" / f"{row['sample_id']}_target.png"))
        row["mask_path"] = str(save_image(row["mask"], root / "log" / f"{row['sample_id']}_gt_mask.png"))
        if row.get("predicted_mask") is not None:
            mask_path = save_image(row["predicted_mask"], root / "log" / f"{row['sample_id']}_predicted_mask.png")
            row["predicted_mask_path"] = str(mask_path)
    for row in rows[: max(0, visual_limit)]:
        grid_path = _write_grid(root / "log" / f"{row['sample_id']}_grid.png", row)
        row["grid_path"] = str(grid_path)
    write_metrics_csv(root / "metrics.csv", rows)
    write_answer_json(root / "answer.json", rows)
    write_summary(
        root / "summary.txt",
        rows,
        model=model,
        dataset=dataset,
        dry_run=dry_run,
        checkpoint=checkpoint,
        split=split,
        manifest=manifest,
    )
