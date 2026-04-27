# Task 002: Data And Synthetic Blur Utilities
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: pending_

## Objective
Implement dataset contracts and synthetic blur utilities for local image deblurring samples with blurred image, blur mask, optional segmentation, and sharp target.

## Context
- Relevant files: [`/root/autodl-tmp/project/local_deblur/data/__init__.py`, `/root/autodl-tmp/project/local_deblur/data/types.py`, `/root/autodl-tmp/project/local_deblur/data/datasets.py`, `/root/autodl-tmp/project/local_deblur/data/synthetic_blur.py`, `/root/autodl-tmp/project/local_deblur/data/cropping.py`, `/root/autodl-tmp/project/local_deblur/data/transforms.py`, `/root/autodl-tmp/project/scripts/prepare_synthetic_data.py`, `/root/autodl-tmp/project/configs/data_synthetic.yaml`]
- Current state: Task 001 should create the base package and config/logging utilities.
- Dependencies: task-001
- Domino assumptions: Inputs are locally blurred image `Ib`, blur mask `M`, optional segmentation map `S`, and sharp target; synthetic data should support COCO-style instance masks, global blur dataset augmentation, arbitrary/object-aware masks, and mask-centered 512x512 crops preserving aspect ratio.
- User decisions: Build the full research workflow scaffold without requiring datasets now; keep dry-run or generated sample support; do not hard-code unavailable dataset paths.

## Instructions
1. Define a typed sample contract for local deblurring records, including paths and loaded tensors/images where appropriate.
2. Implement dataset classes/loaders that can read manifest-based samples and can run in dry-run mode with generated small images/masks.
3. Implement synthetic blur utilities for Gaussian/motion/defocus-like blur kernels applied inside masks.
4. Add utilities for creating arbitrary masks and object-aware masks from instance-mask inputs when available.
5. Add mask-centered crop utilities for 512x512 crops while preserving aspect ratio and cleanly padding/resizing as needed.
6. Add `scripts/prepare_synthetic_data.py` as a CLI scaffold that writes manifests and dry-run artifacts under ignored output paths.
7. Keep data utilities separate from training/evaluation loops.

## Acceptance Criteria
- [ ] Data modules import without real datasets.
- [ ] Dry-run data generation produces at least one coherent sample with `Ib`, `M`, optional `S`, and target fields.
- [ ] Synthetic blur functions apply blur only or primarily to masked regions and preserve background by default.
- [ ] Crop utility supports mask-centered 512x512 output.
- [ ] `scripts/prepare_synthetic_data.py` exposes CLI arguments for COCO/global/ReLoBlur-style inputs and a dry-run mode.

## Output
- Modified files: [`/root/autodl-tmp/project/local_deblur/data/__init__.py`, `/root/autodl-tmp/project/local_deblur/data/types.py`, `/root/autodl-tmp/project/local_deblur/data/datasets.py`, `/root/autodl-tmp/project/local_deblur/data/synthetic_blur.py`, `/root/autodl-tmp/project/local_deblur/data/cropping.py`, `/root/autodl-tmp/project/local_deblur/data/transforms.py`, `/root/autodl-tmp/project/scripts/prepare_synthetic_data.py`, `/root/autodl-tmp/project/configs/data_synthetic.yaml`]
- Result summary: write under `## Result`

## Result
Files changed: `local_deblur/data/`, `scripts/prepare_synthetic_data.py`, `configs/data_synthetic.yaml`.

Acceptance criteria: typed sample contracts, manifest loading, generated dry-run samples, synthetic blur utilities, arbitrary/object masks, mask-centered crops, transforms, and dry-run artifact writing are implemented.

Constraint check: data utilities import without real datasets; full dataset preparation is gated and the script exits unless `--dry-run` is used.
