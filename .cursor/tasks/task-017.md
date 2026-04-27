# Task 017: ReLoBlur SD ControlNet Posttraining
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: blocked_

## Objective
Fine-tune/posttrain the SD + ControlNet model on ReLoBlur after COCO synthetic pretraining.

## Context
- Relevant files: [`/root/autodl-tmp/project/output/datasets/reloblur/`, `/root/autodl-tmp/project/configs/`, `/root/autodl-tmp/project/output/training/`]
- Current state: ReLoBlur collection is planned in task-014; COCO pretrain is planned in task-016.
- Dependencies: task-014, task-015, task-016
- Domino assumptions: This task can only run if ReLoBlur manifests and a COCO-pretrained checkpoint are available.
- User decisions: Posttrain on real ReLoBlur after COCO synthetic pretraining.

## Instructions
1. Use ReLoBlur train manifest from task-014.
2. Initialize from the COCO pretrain checkpoint from task-016.
3. Run a bounded fine-tuning/posttraining experiment and save checkpoints/logs/curves.
4. Validate on ReLoBlur test/validation split when available.
5. If ReLoBlur is unavailable, mark the task blocked with the exact missing data requirement.

## Acceptance Criteria
- [ ] ReLoBlur posttraining uses real ReLoBlur data, not synthetic fallback.
- [ ] Training artifacts and summary are saved.
- [ ] Validation metrics are recorded when test data is available.
- [ ] Limitations and blocked states are explicit.

## Output
- Modified files: [`/root/autodl-tmp/project/configs/`, `/root/autodl-tmp/project/output/training/`, `/root/autodl-tmp/project/.cursor/tasks/task-017.md`]
- Result summary: write under `## Result`

## Result

Outcome:
- Still blocked. ReLoBlur posttraining cannot run yet because the real ReLoBlur data is unavailable:
  - ReLoBlur train/test/mask data is not local, and official Google Drive download failed with `Network is unreachable`.
  - COCO SD + ControlNet smoke/pretraining now has a local checkpoint from task-016, but this task requires real ReLoBlur manifests before fine-tuning.

Artifact:
- `output/training/reloblur_posttrain_blocked.json`
- Available COCO SD + ControlNet checkpoint: `output/training/sd15_tile_controlnet_coco_pretrain/checkpoint/controlnet/diffusion_pytorch_model.safetensors`
- Available auxiliary mask-head checkpoint: `output/training/sd15_tile_controlnet_coco_pretrain/checkpoint/aux_mask_head.pt`

Required to continue:
- Place extracted ReLoBlur `dataset/` and `masks/` locally, then run `scripts/prepare_reloblur_manifest.py`.
- Rerun ReLoBlur posttraining after train/test manifests exist.

Constraint check:
- Did not use synthetic fallback as ReLoBlur.
- Did not label any baseline result as ReLoBlur posttraining.
- No ReLoBlur training was run without the required real data.
