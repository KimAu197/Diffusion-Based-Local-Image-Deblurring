"""Apply NAFNet deblurring inside predicted masks and write full-image outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from local_deblur.data.transforms import array_to_image, image_to_array, load_mask, load_rgb, save_image
from local_deblur.eval.metrics import metric_bundle
from local_deblur.paths import dated_result_dir, resolve_project_path


DEFAULT_ANSWER = (
    "results/reloblur_predmask_infra_eval_SD15-TileControlNet-ReLoBlur-PredMask_"
    "reloblur-test_5_0430/answer.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer", default=DEFAULT_ANSWER, help="Existing eval answer.json with input/mask paths.")
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/models/NAFNet-GoPro-width64.pth")
    parser.add_argument("--nafnet-root", default="external/NAFNet")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--round", default="reloblur_predmask_nafnet_mask_postprocess")
    parser.add_argument("--model", default="NAFNet-GoPro-width64-MaskComposite")
    parser.add_argument("--dataset", default="reloblur-test")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--blend", type=float, default=0.65, help="Blend NAFNet result with input before mask compositing.")
    parser.add_argument("--mask-blur-radius", type=float, default=2.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resource_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"cpu_count": None, "nvidia_smi": ""}
    try:
        import os

        snapshot["cpu_count"] = os.cpu_count()
    except Exception as exc:
        snapshot["cpu_error"] = repr(exc)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        snapshot["nvidia_smi"] = result.stdout.strip()
    except Exception as exc:
        snapshot["nvidia_smi_error"] = repr(exc)
    return snapshot


def import_nafnet(nafnet_root: Path):
    sys.path.insert(0, str(nafnet_root.resolve()))
    from basicsr.models.archs.NAFNet_arch import NAFNetLocal

    return NAFNetLocal


def normalize_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("params_ema", "params", "state_dict", "net_g"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("NAFNet checkpoint must be a state-dict-like object.")

    normalized = {}
    for key, value in checkpoint.items():
        clean = str(key)
        for prefix in ("module.", "net_g.", "model."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        normalized[clean] = value
    return normalized


def load_model(checkpoint_path: Path, nafnet_root: Path, device: str) -> torch.nn.Module:
    NAFNetLocal = import_nafnet(nafnet_root)
    model = NAFNetLocal(
        img_channel=3,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1],
        train_size=(1, 3, 256, 256),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = normalize_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"NAFNet checkpoint load warning: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 20 or len(unexpected) > 20:
            raise RuntimeError("Checkpoint does not look compatible with NAFNet-width64.")
    model.to(device).eval()
    return model


def image_to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    array = image_to_array(image)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    return tensor


def run_nafnet(model: torch.nn.Module, image: Image.Image, device: str) -> Image.Image:
    with torch.no_grad():
        output = model(image_to_tensor(image, device)).detach().clamp(0.0, 1.0)[0].cpu()
    array = output.permute(1, 2, 0).numpy()
    return array_to_image(array)


def compose_masked(input_image: Image.Image, nafnet_image: Image.Image, mask: Image.Image, *, blend: float, radius: float) -> Image.Image:
    input_array = image_to_array(input_image)
    nafnet_array = image_to_array(nafnet_image.resize(input_image.size, Image.Resampling.BICUBIC))
    blended = array_to_image(np.clip(input_array * (1.0 - blend) + nafnet_array * blend, 0.0, 1.0))
    alpha = mask.convert("L").resize(input_image.size, Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius=radius))
    return Image.composite(blended.convert("RGB"), input_image.convert("RGB"), alpha)


def label(image: Image.Image, text: str) -> Image.Image:
    labeled = image.convert("RGB")
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return labeled


def write_grid(path: Path, row: dict[str, Any]) -> Path:
    error = array_to_image(np.clip(np.abs(image_to_array(row["prediction"]) - image_to_array(row["target"])) * 4.0, 0.0, 1.0))
    panels = [
        label(row["input"], "blurred input"),
        label(row["predicted_mask"].convert("RGB"), "pred mask"),
        label(row["nafnet_full"], "NAFNet full"),
        label(row["prediction"], "masked full output"),
        label(row["target"], "target"),
        label(error, "abs error x4"),
    ]
    width, height = panels[0].size
    grid = Image.new("RGB", (width * 3, height * 2), color=(255, 255, 255))
    for idx, panel in enumerate(panels):
        grid.paste(panel, ((idx % 3) * width, (idx // 3) * height))
    return save_image(grid, path)


def first_existing(row: dict[str, Any], *keys: str) -> Path | None:
    for key in keys:
        value = row.get(key)
        if value and Path(value).exists():
            return Path(value)
    return None


def write_outputs(root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    (root / "log").mkdir(parents=True, exist_ok=True)
    metric_names = sorted(rows[0]["metrics"].keys()) if rows else []
    with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", *metric_names])
        writer.writeheader()
        for row in rows:
            writer.writerow({"sample_id": row["sample_id"], **row["metrics"]})

    answer_rows = []
    for row in rows:
        answer_rows.append(
            {
                "sample_id": row["sample_id"],
                "answer": row["prediction_path"],
                "input": row["input_path"],
                "target": row["target_path"],
                "ground_truth_mask": row["mask_path"],
                "predicted_mask": row["predicted_mask_path"],
                "nafnet_full": row["nafnet_full_path"],
                "visual_grid": row["grid_path"],
                "metrics": row["metrics"],
                "metadata": row["metadata"],
                "sample_metadata": row.get("sample_metadata", {}),
            }
        )
    (root / "answer.json").write_text(json.dumps(answer_rows, indent=2), encoding="utf-8")

    averages = {key: sum(row["metrics"][key] for row in rows) / len(rows) for key in metric_names} if rows else {}
    lines = [
        "NAFNet Mask Postprocess Summary",
        f"model: {args.model}",
        f"checkpoint: {args.checkpoint}",
        f"samples: {len(rows)}",
        f"blend: {args.blend}",
        f"mask_blur_radius: {args.mask_blur_radius}",
        "Interpretation: NAFNet-GoPro-width64 is applied to the full input, then only the predicted-mask region is composited back into the full image.",
        "",
        "Averages:",
    ]
    lines.extend(f"{key}: {value:.6f}" for key, value in averages.items())
    (root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    answer_path = resolve_project_path(args.answer)
    checkpoint_path = Path(args.checkpoint)
    nafnet_root = resolve_project_path(args.nafnet_root)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing NAFNet checkpoint: {checkpoint_path}")
    if not nafnet_root.exists():
        raise FileNotFoundError(f"Missing NAFNet source root: {nafnet_root}")

    rows_in = json.loads(answer_path.read_text(encoding="utf-8"))
    rows_in = rows_in[: args.count] if args.count else rows_in
    output_root = dated_result_dir(args.round, args.model, args.dataset, len(rows_in), root=args.output_root)
    (output_root / "resource_before.json").write_text(json.dumps(resource_snapshot(), indent=2), encoding="utf-8")

    model = load_model(checkpoint_path, nafnet_root, args.device)
    rows = []
    for idx, source in enumerate(rows_in):
        input_path = first_existing(source, "input") or Path(source["sample_metadata"]["input_path"])
        target_path = first_existing(source, "target") or Path(source["sample_metadata"]["target_path"])
        mask_path = first_existing(source, "ground_truth_mask") or Path(source["sample_metadata"]["mask_path"])
        pred_mask_path = first_existing(source, "predicted_mask")
        if pred_mask_path is None:
            raise FileNotFoundError(f"Missing predicted mask for sample {source.get('sample_id', idx)}")

        input_image = load_rgb(input_path)
        target = load_rgb(target_path).resize(input_image.size, Image.Resampling.BICUBIC)
        gt_mask = load_mask(mask_path).resize(input_image.size, Image.Resampling.NEAREST)
        predicted_mask = load_mask(pred_mask_path).resize(input_image.size, Image.Resampling.BICUBIC)
        nafnet_full = run_nafnet(model, input_image, args.device)
        prediction = compose_masked(input_image, nafnet_full, predicted_mask, blend=args.blend, radius=args.mask_blur_radius)
        metrics = metric_bundle(prediction, target, gt_mask)
        for key, value in source.get("metrics", {}).items():
            metrics[f"source_{key}"] = float(value)
        rows.append(
            {
                "sample_id": source.get("sample_id", f"sample_{idx:03d}"),
                "input": input_image,
                "target": target,
                "mask": gt_mask,
                "predicted_mask": predicted_mask,
                "nafnet_full": nafnet_full,
                "prediction": prediction,
                "metrics": metrics,
                "metadata": {
                    "checkpoint": str(checkpoint_path),
                    "nafnet_root": str(nafnet_root),
                    "device": args.device,
                    "blend": args.blend,
                    "mask_blur_radius": args.mask_blur_radius,
                    "mask_source": "stage1_predicted_mask_from_answer_json",
                },
                "sample_metadata": source.get("sample_metadata", {}),
            }
        )

    for row in rows:
        prefix = output_root / "log" / row["sample_id"]
        row["prediction_path"] = str(save_image(row["prediction"], f"{prefix}_nafnet_masked_prediction.png"))
        row["nafnet_full_path"] = str(save_image(row["nafnet_full"], f"{prefix}_nafnet_full.png"))
        row["input_path"] = str(save_image(row["input"], f"{prefix}_input.png"))
        row["target_path"] = str(save_image(row["target"], f"{prefix}_target.png"))
        row["mask_path"] = str(save_image(row["mask"], f"{prefix}_gt_mask.png"))
        row["predicted_mask_path"] = str(save_image(row["predicted_mask"], f"{prefix}_predicted_mask.png"))
        row["grid_path"] = str(write_grid(Path(f"{prefix}_grid.png"), row))

    write_outputs(output_root, rows, args)
    (output_root / "resource_after.json").write_text(json.dumps(resource_snapshot(), indent=2), encoding="utf-8")
    (output_root / "logging.log").write_text(f"NAFNet mask postprocess completed: {output_root}\n", encoding="utf-8")
    print(output_root)


if __name__ == "__main__":
    main()
