#!/usr/bin/env python
"""Prepare synthetic local deblurring data or a dry-run fixture."""

from __future__ import annotations

import argparse
import json
import zipfile
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

from local_deblur.data.cropping import mask_centered_crop
from local_deblur.data.synthetic_blur import (
    apply_defocus_local_blur,
    apply_gaussian_local_blur,
    apply_motion_local_blur,
    make_arbitrary_mask,
    write_dry_run_artifacts,
)
from local_deblur.data.transforms import mask_to_array, save_image
from local_deblur.paths import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-images", default=None, help="COCO image directory or zip file.")
    parser.add_argument("--coco-instances", default=None, help="COCO instances JSON file or annotations zip.")
    parser.add_argument("--annotation-split", choices=["train", "val"], default="val")
    parser.add_argument("--global-blur-dir", default=None, help="Optional global blur dataset root.")
    parser.add_argument("--reloblur-dir", default=None, help="Optional ReLoBlur dataset root.")
    parser.add_argument("--output-dir", default="output/synthetic_dry_run", help="Ignored output directory.")
    parser.add_argument("--count", type=int, default=2, help="Number of samples to generate. Use 0 for all available source images.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true", help="Generate synthetic fixtures without external datasets.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prefer-categories",
        default="person,car,motorcycle,bicycle,dog,bird,bus,train,truck,horse,sports ball,skateboard",
        help="Comma-separated COCO categories prioritized for local motion blur.",
    )
    parser.add_argument(
        "--skip-categories",
        default="chair,dining table,couch,bed,refrigerator",
        help="Comma-separated COCO categories to skip.",
    )
    parser.add_argument("--min-mask-frac", type=float, default=0.02, help="Minimum mask area fraction after crop.")
    parser.add_argument("--max-mask-frac", type=float, default=0.35, help="Maximum mask area fraction after crop.")
    parser.add_argument("--soft-mask-radius", type=float, default=5.0, help="Gaussian feather radius for mask boundaries.")
    parser.add_argument("--blur-types", default="motion,gaussian,defocus", help="Comma-separated blur variants to mix.")
    parser.add_argument(
        "--attach-categories",
        default="backpack,handbag,umbrella,tie,suitcase,cell phone,skis,snowboard,skateboard,sports ball",
        help="Categories to merge into nearby person groups.",
    )
    parser.add_argument("--max-samples-per-image", type=int, default=2)
    parser.add_argument("--merge-distance-frac", type=float, default=0.08)
    parser.add_argument("--edge-margin-frac", type=float, default=0.02)
    return parser.parse_args()


def _iter_image_records(source: Path):
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = sorted(name for name in archive.namelist() if Path(name).suffix.lower() in suffixes and not name.endswith("/"))
            for name in names:
                with archive.open(name) as handle:
                    image = Image.open(BytesIO(handle.read())).convert("RGB")
                    yield Path(name).stem, image
        return

    if source.is_dir():
        paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in suffixes)
        for path in paths:
            yield path.stem, Image.open(path).convert("RGB")
        return

    raise FileNotFoundError(f"Unsupported image source: {source}")


def _load_coco_annotations(source: Path, split: str) -> tuple[dict, dict[int, str], dict[int, dict], dict[int, list[dict]]]:
    annotation_name = f"annotations/instances_{split}2017.json"
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            with archive.open(annotation_name) as handle:
                payload = json.load(handle)
    elif source.is_file():
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError(f"Unsupported COCO annotation source: {source}")

    categories = {item["id"]: item["name"] for item in payload["categories"]}
    images = {item["id"]: item for item in payload["images"]}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)
    return payload, categories, images, annotations_by_image


def _annotation_to_mask(annotation: dict, size: tuple[int, int]) -> Image.Image | None:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list):
        return None
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentation:
        if len(polygon) >= 6:
            points = [(polygon[index], polygon[index + 1]) for index in range(0, len(polygon), 2)]
            draw.polygon(points, fill=255)
    return mask if mask.getbbox() else None


def _mask_bbox(mask: Image.Image) -> tuple[int, int, int, int] | None:
    return mask.convert("L").getbbox()


def _bbox_area(box: tuple[int, int, int, int] | None) -> float:
    if box is None:
        return 0.0
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def _bbox_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    return float(max(0, right - left) * max(0, bottom - top))


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return float((dx * dx + dy * dy) ** 0.5)


def _bbox_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _touches_edge(box: tuple[int, int, int, int], size: tuple[int, int], margin_frac: float) -> bool:
    width, height = size
    margin = int(min(width, height) * margin_frac)
    return box[0] <= margin or box[1] <= margin or box[2] >= width - margin or box[3] >= height - margin


def _merge_masks(size: tuple[int, int], masks: list[Image.Image]) -> Image.Image:
    merged = Image.new("L", size, 0)
    for mask in masks:
        merged = Image.composite(Image.new("L", size, 255), merged, mask.convert("L"))
    return merged


def _has_black_border(image: Image.Image, threshold: int = 8, max_fraction: float = 0.25) -> bool:
    arr = np.asarray(image.convert("RGB"))
    strips = [arr[:8, :, :], arr[-8:, :, :], arr[:, :8, :], arr[:, -8:, :]]
    black_fraction = sum(float((strip.mean(axis=-1) < threshold).mean()) for strip in strips) / len(strips)
    return black_fraction > max_fraction


def _safe_mask_centered_crop(
    image: Image.Image,
    mask: Image.Image,
    *,
    size: int,
    segmentation: Image.Image | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image | None] | None:
    rgb = image.convert("RGB")
    mask_l = mask.convert("L")
    box = _mask_bbox(mask_l)
    if box is None:
        return None
    width, height = rgb.size
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    crop_side = int(max(size, max(box_w, box_h) * 2.2))
    crop_side = min(crop_side, width, height)
    if crop_side < max(box_w, box_h):
        return None
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    left = min(max(0, cx - crop_side // 2), width - crop_side)
    top = min(max(0, cy - crop_side // 2), height - crop_side)
    crop_box = (left, top, left + crop_side, top + crop_side)
    cropped_target = rgb.crop(crop_box)
    if _has_black_border(cropped_target):
        return None

    def crop_one(img: Image.Image | None, is_mask: bool) -> Image.Image | None:
        if img is None:
            return None
        mode = "L" if is_mask else "RGB"
        resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BICUBIC
        return img.convert(mode).crop(crop_box).resize((size, size), resample)

    return crop_one(rgb, False), crop_one(mask_l, True), crop_one(segmentation, True)


def _read_image_from_zip(archive: zipfile.ZipFile, file_name: str) -> Image.Image:
    candidates = [file_name, f"val2017/{file_name}", f"train2017/{file_name}"]
    for candidate in candidates:
        if candidate in archive.namelist():
            with archive.open(candidate) as handle:
                return Image.open(BytesIO(handle.read())).convert("RGB")
    raise FileNotFoundError(f"Image {file_name} not found in zip")


def _iter_semantic_coco_records(
    image_source: Path,
    annotation_source: Path,
    *,
    split: str,
    prefer_categories: set[str],
    skip_categories: set[str],
    min_mask_frac: float,
    max_mask_frac: float,
):
    _, categories, images, annotations_by_image = _load_coco_annotations(annotation_source, split)
    if not (image_source.is_file() and image_source.suffix.lower() == ".zip"):
        raise FileNotFoundError("Semantic COCO generation currently expects an image zip source")

    with zipfile.ZipFile(image_source) as archive:
        for image_id in sorted(annotations_by_image):
            image_info = images[image_id]
            image = None
            for annotation in annotations_by_image[image_id]:
                category = categories[annotation["category_id"]]
                if annotation.get("iscrowd") or category in skip_categories or category not in prefer_categories:
                    continue
                image_area = float(image_info["width"] * image_info["height"])
                raw_area_frac = float(annotation.get("area", 0.0)) / max(1.0, image_area)
                if raw_area_frac <= 0:
                    continue
                if image is None:
                    image = _read_image_from_zip(archive, image_info["file_name"])
                mask = _annotation_to_mask(annotation, image.size)
                if mask is None:
                    continue
                yield image_info, annotation, category, image.copy(), mask


def _iter_semantic_coco_groups(
    image_source: Path,
    annotation_source: Path,
    *,
    split: str,
    prefer_categories: set[str],
    skip_categories: set[str],
    attach_categories: set[str],
    max_samples_per_image: int,
    merge_distance_frac: float,
    edge_margin_frac: float,
):
    _, categories, images, annotations_by_image = _load_coco_annotations(annotation_source, split)
    if not (image_source.is_file() and image_source.suffix.lower() == ".zip"):
        raise FileNotFoundError("Semantic COCO generation currently expects an image zip source")

    independent_categories = (prefer_categories | {"bus", "train", "truck", "horse"}) - attach_categories
    with zipfile.ZipFile(image_source) as archive:
        for image_id in sorted(annotations_by_image):
            image_info = images[image_id]
            image = None
            instances: list[dict] = []
            for annotation in annotations_by_image[image_id]:
                category = categories[annotation["category_id"]]
                if annotation.get("iscrowd") or category in skip_categories or category not in independent_categories | attach_categories:
                    continue
                if image is None:
                    image = _read_image_from_zip(archive, image_info["file_name"])
                mask = _annotation_to_mask(annotation, image.size)
                box = _mask_bbox(mask) if mask is not None else None
                if mask is None or box is None or _touches_edge(box, image.size, edge_margin_frac):
                    continue
                instances.append({"annotation": annotation, "category": category, "mask": mask, "box": box})
            if image is None or not instances:
                continue

            groups: list[dict] = []
            used_attach: set[int] = set()
            for inst in instances:
                category = inst["category"]
                if category not in independent_categories:
                    continue
                group_masks = [inst["mask"]]
                group_categories = [category]
                group_ids = [inst["annotation"]["id"]]
                group_box = inst["box"]
                if category == "person":
                    for attach in instances:
                        attach_id = attach["annotation"]["id"]
                        if attach_id in used_attach or attach["category"] not in attach_categories:
                            continue
                        inter = _bbox_intersection(group_box, attach["box"])
                        close = _bbox_distance(group_box, attach["box"]) <= min(image.size) * merge_distance_frac
                        if inter > 0 or close:
                            group_masks.append(attach["mask"])
                            group_categories.append(attach["category"])
                            group_ids.append(attach_id)
                            group_box = _bbox_union(group_box, attach["box"])
                            used_attach.add(attach_id)
                groups.append(
                    {
                        "masks": group_masks,
                        "categories": group_categories,
                        "annotation_ids": group_ids,
                        "box": group_box,
                    }
                )

            merged_groups: list[dict] = []
            for group in groups:
                merged = False
                for existing in merged_groups:
                    close = _bbox_distance(existing["box"], group["box"]) <= min(image.size) * merge_distance_frac
                    overlap = _bbox_intersection(existing["box"], group["box"]) > 0
                    if close or overlap:
                        existing["masks"].extend(group["masks"])
                        existing["categories"].extend(group["categories"])
                        existing["annotation_ids"].extend(group["annotation_ids"])
                        existing["box"] = _bbox_union(existing["box"], group["box"])
                        merged = True
                        break
                if not merged:
                    merged_groups.append(group)

            selected: list[dict] = []
            for group in sorted(merged_groups, key=lambda item: _bbox_area(item["box"]), reverse=True):
                if len(selected) >= max_samples_per_image:
                    break
                if any(_bbox_intersection(group["box"], prev["box"]) > 0 for prev in selected):
                    continue
                group_mask = _merge_masks(image.size, group["masks"])
                selected.append({**group, "mask": group_mask})

            for group in selected:
                yield image_info, group, image.copy()


def _apply_blur_variant(image: Image.Image, mask: Image.Image, blur_type: str) -> Image.Image:
    if blur_type == "gaussian":
        return apply_gaussian_local_blur(image, mask, radius=5.0)
    if blur_type == "motion":
        return apply_motion_local_blur(image, mask, radius=5)
    return apply_defocus_local_blur(image, mask, radius=4)


def write_coco_synthetic_artifacts(source: Path, output_dir: Path, *, count: int, size: int, seed: int) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    limit = None if count == 0 else max(1, count)
    for index, (source_id, image) in enumerate(_iter_image_records(source)):
        if limit is not None and len(records) >= limit:
            break
        sample_id = f"coco_synth_{index:06d}"
        mask = make_arbitrary_mask(image.size, seed=seed + index)
        target, mask, segmentation, _ = mask_centered_crop(image, mask, size=size, segmentation=mask, target=None)
        if target is None or mask is None:
            continue
        blurred = _apply_blur_variant(target, mask, ["gaussian", "motion", "defocus"][index % 3])
        segmentation = segmentation or mask

        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        blurred_path = save_image(blurred, sample_dir / "Ib.png")
        mask_path = save_image(mask, sample_dir / "M.png")
        target_path = save_image(target, sample_dir / "target.png")
        segmentation_path = save_image(segmentation, sample_dir / "S.png")
        records.append(
            {
                "sample_id": sample_id,
                "Ib": str(blurred_path),
                "M": str(mask_path),
                "target": str(target_path),
                "S": str(segmentation_path),
                "metadata": {
                    "source": "coco",
                    "source_id": source_id,
                    "mask_mean": float(mask_to_array(mask).mean()),
                    "blur_variant": ["gaussian", "motion", "defocus"][index % 3],
                },
            }
        )
    if not records:
        raise RuntimeError(f"No images were processed from {source}")
    return records


def write_coco_semantic_artifacts(
    image_source: Path,
    annotation_source: Path,
    output_dir: Path,
    *,
    count: int,
    size: int,
    seed: int,
    split: str,
    prefer_categories: set[str],
    skip_categories: set[str],
    min_mask_frac: float,
    max_mask_frac: float,
    soft_mask_radius: float,
    blur_types: list[str],
    attach_categories: set[str],
    max_samples_per_image: int,
    merge_distance_frac: float,
    edge_margin_frac: float,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    limit = None if count == 0 else max(1, count)
    for index, (image_info, group, image) in enumerate(
        _iter_semantic_coco_groups(
            image_source,
            annotation_source,
            split=split,
            prefer_categories=prefer_categories,
            skip_categories=skip_categories,
            attach_categories=attach_categories,
            max_samples_per_image=max_samples_per_image,
            merge_distance_frac=merge_distance_frac,
            edge_margin_frac=edge_margin_frac,
        )
    ):
        if limit is not None and len(records) >= limit:
            break
        crop_result = _safe_mask_centered_crop(image, group["mask"], size=size, segmentation=group["mask"])
        if crop_result is None:
            continue
        target, crop_mask, segmentation = crop_result
        mask_mean = float(mask_to_array(crop_mask).mean())
        if mask_mean < min_mask_frac or mask_mean > max_mask_frac:
            continue
        soft_mask = crop_mask.filter(ImageFilter.GaussianBlur(radius=soft_mask_radius))
        segmentation = segmentation or crop_mask
        blur_type = blur_types[(seed + index) % len(blur_types)]
        blurred = _apply_blur_variant(target, soft_mask, blur_type)

        sample_id = f"coco_semantic_{len(records):06d}"
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        blurred_path = save_image(blurred, sample_dir / "Ib.png")
        mask_path = save_image(soft_mask, sample_dir / "M.png")
        target_path = save_image(target, sample_dir / "target.png")
        segmentation_path = save_image(segmentation, sample_dir / "S.png")
        records.append(
            {
                "sample_id": sample_id,
                "Ib": str(blurred_path),
                "M": str(mask_path),
                "target": str(target_path),
                "S": str(segmentation_path),
                "metadata": {
                    "source": "coco",
                    "source_id": Path(image_info["file_name"]).stem,
                    "annotation_ids": list(dict.fromkeys(group["annotation_ids"])),
                    "categories": list(dict.fromkeys(group["categories"])),
                    "category": "+".join(sorted(set(group["categories"]))),
                    "mask_mean": mask_mean,
                    "blur_variant": blur_type,
                    "semantic_mask": True,
                    "grouped_motion_object": True,
                    "soft_mask_radius": soft_mask_radius,
                },
            }
        )
    if not records:
        raise RuntimeError(f"No semantic COCO samples were processed from {image_source}")
    return records


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    if args.dry_run:
        records = write_dry_run_artifacts(output_dir, count=args.count, size=args.image_size)
    else:
        if not args.coco_images:
            raise SystemExit("--coco-images is required unless --dry-run is used")
        if args.coco_instances:
            prefer_categories = {item.strip() for item in args.prefer_categories.split(",") if item.strip()}
            skip_categories = {item.strip() for item in args.skip_categories.split(",") if item.strip()}
            attach_categories = {item.strip() for item in args.attach_categories.split(",") if item.strip()}
            blur_types = [item.strip() for item in args.blur_types.split(",") if item.strip()]
            records = write_coco_semantic_artifacts(
                Path(args.coco_images),
                Path(args.coco_instances),
                output_dir,
                count=args.count,
                size=args.image_size,
                seed=args.seed,
                split=args.annotation_split,
                prefer_categories=prefer_categories,
                skip_categories=skip_categories,
                min_mask_frac=args.min_mask_frac,
                max_mask_frac=args.max_mask_frac,
                soft_mask_radius=args.soft_mask_radius,
                blur_types=blur_types,
                attach_categories=attach_categories,
                max_samples_per_image=args.max_samples_per_image,
                merge_distance_frac=args.merge_distance_frac,
                edge_margin_frac=args.edge_margin_frac,
            )
        else:
            records = write_coco_synthetic_artifacts(Path(args.coco_images), output_dir, count=args.count, size=args.image_size, seed=args.seed)
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps({"samples": records}, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} samples")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
