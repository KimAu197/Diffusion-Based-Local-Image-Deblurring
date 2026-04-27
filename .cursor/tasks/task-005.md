# Task 005: Inference Script And Examples
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: pending_

## Objective
Add an inference CLI that restores a locally blurred image from `Ib`, `M`, and optional `S` using the shared pipeline interface.

## Context
- Relevant files: [`/root/autodl-tmp/project/scripts/infer.py`, `/root/autodl-tmp/project/local_deblur/inference.py`, `/root/autodl-tmp/project/configs/inference.yaml`, `/root/autodl-tmp/project/examples/inference_manifest.json`]
- Current state: Task 002 should provide data transforms/sample contracts; task 003 should provide the model pipeline wrapper.
- Dependencies: task-001, task-002, task-003
- Domino assumptions: Output should restore the sharp local region while preserving background and boundary transitions; inference must work in dry-run/fallback mode if checkpoints are unavailable.
- User decisions: Include training and inference scripts in the deliverable; keep examples practical and runnable without real datasets.

## Instructions
1. Add `local_deblur/inference.py` with reusable inference functions for loading images/masks/optional segmentation, invoking `LocalDeblurPipeline`, and saving outputs.
2. Add `scripts/infer.py` CLI accepting `--image`, `--mask`, optional `--segmentation`, `--checkpoint`, `--config`, `--output`, and `--dry-run`.
3. Ensure dry-run mode can generate a synthetic sample if no image path is provided and write an output image under an ignored directory.
4. Preserve background outside the mask through the pipeline/post-processing interface.
5. Add `configs/inference.yaml` and a small example manifest showing expected fields.
6. Avoid duplicating model/data utilities already provided by earlier tasks.

## Acceptance Criteria
- [ ] Inference CLI exposes all required inputs: blurred image, blur mask, optional segmentation, checkpoint/config, output, and dry-run.
- [ ] Dry-run inference can produce an output artifact without real checkpoints.
- [ ] Full-mode code path validates required input files and reports useful errors.
- [ ] Output image shape and path handling are documented or asserted.
- [ ] Background preservation uses shared post-processing/pipeline helpers.

## Output
- Modified files: [`/root/autodl-tmp/project/scripts/infer.py`, `/root/autodl-tmp/project/local_deblur/inference.py`, `/root/autodl-tmp/project/configs/inference.yaml`, `/root/autodl-tmp/project/examples/inference_manifest.json`]
- Result summary: write under `## Result`

## Result
Files changed: `local_deblur/inference.py`, `scripts/infer.py`, `configs/inference.yaml`, `examples/inference_manifest.json`.

Acceptance criteria: reusable inference, CLI inputs for image/mask/segmentation/checkpoint/config/output/dry-run, synthetic dry-run output, and background-preserving pipeline use are implemented.

Constraint check: full inference without a checkpoint is rejected; dry-run needs no checkpoint or dataset.
