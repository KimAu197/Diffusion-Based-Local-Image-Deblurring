# Task 016: COCO Synthetic SD ControlNet Pretraining
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: completed_

## Objective
Run a bounded SD + ControlNet pretraining experiment on the COCO synthetic 5K local-blur dataset and save checkpoints/metrics.

## Context
- Relevant files: [`/root/autodl-tmp/project/output/datasets/coco2017_train_grouped_localblur_5k/manifest.json`, `/root/autodl-tmp/project/configs/`, `/root/autodl-tmp/project/scripts/train.py`, `/root/autodl-tmp/project/output/training/`]
- Current state: 5K synthetic grouped local-blur data is ready and validated.
- Dependencies: task-013, task-015
- Domino assumptions: Use bounded compute first; if full SD checkpoints cannot be downloaded/accessed, record the block and preserve runnable config.
- User decisions: Pretrain on COCO synthetic before ReLoBlur posttraining.

## Instructions
1. Add/run a COCO SD + ControlNet pretrain config with explicit checkpoint/model names or local paths.
2. Use the 5K manifest and train/validation split.
3. Save checkpoint(s), training logs, loss curves, and a training summary.
4. Keep the run bounded; do not consume unbounded GPU time.
5. If checkpoint download/access is blocked, stop with a concrete block report rather than silently falling back to the PyTorch baseline.

## Acceptance Criteria
- [ ] COCO pretrain command/config is reproducible.
- [ ] A checkpoint or clear blocked-checkpoint report exists.
- [ ] Training summary and logs are saved.
- [ ] No fallback-only result is mislabeled as SD + ControlNet.

## Output
- Modified files: [`/root/autodl-tmp/project/configs/`, `/root/autodl-tmp/project/output/training/`, `/root/autodl-tmp/project/.cursor/tasks/task-016.md`]
- Result summary: write under `## Result`

## Previous Result

Files changed:
- `configs/train_sd_controlnet_coco.yaml`
- `scripts/train_sd_controlnet_coco.py`
- `.cursor/tasks/task-016.md`

Outcome:
- Blocked before training because the requested diffusers-backed tiny Stable Diffusion checkpoint could not be loaded.
- Attempted checkpoint: `hf-internal-testing/tiny-stable-diffusion-pipe`.
- ControlNet plan in the runnable script: initialize `diffusers.ControlNetModel.from_unet(pipe.unet)` from the tiny SD UNet, then train ControlNet residuals plus `ControlNetAuxMaskHead` for a bounded 2-step smoke run on the COCO synthetic 5K manifest.
- Exact blocker: Hugging Face Hub network access failed with `[Errno 101] Network is unreachable`; diffusers then tried local cache and reported the model was not cached locally.
- No baseline/fallback training was run or labeled as SD + ControlNet.
- No ReLoBlur dependency was used.

Commands:
- `python -m py_compile "scripts/train_sd_controlnet_coco.py"`
- `python "scripts/train_sd_controlnet_coco.py" --config "configs/train_sd_controlnet_coco.yaml"`

Artifacts:
- Blocked summary: `output/training/sd_controlnet_coco_pretrain/training_summary.json`
- Log: `output/training/sd_controlnet_coco_pretrain/logging.log`
- Runnable config: `configs/train_sd_controlnet_coco.yaml`
- Runnable script: `scripts/train_sd_controlnet_coco.py`

Metrics:
- No `loss_curve.csv` and no checkpoint were produced because checkpoint loading failed before the training loop.
- `training_summary.json` is labeled `blocked_no_diffusers_backed_checkpoint` and records `used_baseline_fallback: false`.

## Result

Continuing after user decision: use SD 1.5 + Tile ControlNet.

Files changed:
- `configs/train_sd_controlnet_coco.yaml`
- `scripts/train_sd_controlnet_coco.py`
- `.cursor/tasks/task-016.md`

Current execution plan:
- Base SD checkpoint: `runwayml/stable-diffusion-v1-5`.
- Tile ControlNet checkpoint: `lllyasviel/control_v11f1e_sd15_tile`.
- Hugging Face cache: `/root/autodl-tmp/cache/huggingface`.
- Bounded run: 2 training steps, 8 train samples, 2 validation samples, image size 256.

Completed continuation:
- Found the partially downloaded local assets and completed the local SD + ControlNet smoke path using:
  - Base SD checkpoint: `/root/autodl-tmp/models/sd15-fp16`
  - Tile ControlNet checkpoint: `/root/autodl-tmp/models/control-v11f1e-sd15-tile`
- Added the missing official SD1.5 `unet/config.json` to the local model directory so diffusers can load the local fp16 checkpoint files.
- Updated `configs/train_sd_controlnet_coco.yaml` to use local model paths with `local_files_only: true`, `allow_downloads: false`, fp32 compute, and a conservative `1e-6` learning rate.
- Ran the bounded smoke/pretraining command successfully:
  - `HF_HUB_DISABLE_XET=1 python3 scripts/train_sd_controlnet_coco.py --config configs/train_sd_controlnet_coco.yaml`
- The first fp16 optimizer run produced `nan` at step 2 and was rejected. The fp32 rerun completed with finite losses.

Artifacts:
- Summary: `output/training/sd15_tile_controlnet_coco_pretrain/training_summary.json`
- Loss curve: `output/training/sd15_tile_controlnet_coco_pretrain/loss_curve.csv`
- ControlNet checkpoint: `output/training/sd15_tile_controlnet_coco_pretrain/checkpoint/controlnet/diffusion_pytorch_model.safetensors`
- Mask-head checkpoint: `output/training/sd15_tile_controlnet_coco_pretrain/checkpoint/aux_mask_head.pt`

Final smoke metrics:
- Step 1 total/diffusion/mask loss: `0.968189 / 0.861857 / 1.063319`
- Step 2 total/diffusion/mask loss: `0.206055 / 0.096977 / 1.090777`
- Step 2 validation mask loss/BCE/Dice: `0.977106 / 0.625439 / 0.703332`

Constraint check:
- User explicitly selected SD 1.5 + Tile ControlNet.
- No fallback-only result will be labeled as SD + ControlNet.
- Run remains bounded for smoke/pretraining validation.
