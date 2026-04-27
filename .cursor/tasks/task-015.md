# Task 015: SD ControlNet Local Deblur Integration
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: completed_

## Objective
Add an SD + ControlNet-compatible local deblurring training/inference path with spatial conditioning and a blur-mask prediction head.

## Context
- Relevant files: [`/root/autodl-tmp/project/local_deblur/models/`, `/root/autodl-tmp/project/local_deblur/training/`, `/root/autodl-tmp/project/configs/model.yaml`, `/root/autodl-tmp/project/configs/`]
- Current state: A compact `ConditionalLocalDeblurNet` baseline exists. The SD + ControlNet path is currently only a scaffold/fallback interface.
- Dependencies: task-013
- Domino assumptions: Prefer Hugging Face `diffusers` APIs; keep downloads/config explicit; support local-only checkpoint paths; preserve baseline code.
- User decisions: Integrate/train an SD + ControlNet version and retain the TA-requested blur-mask head.

## Instructions
1. Add a dedicated SD + ControlNet model/training module rather than replacing the PyTorch baseline.
2. Represent conditioning from blurred image, blur mask, optional segmentation, and background context.
3. Add a lightweight mask head path attached to ControlNet-like intermediate features where feasible; otherwise expose a compatible auxiliary head with documented limitations.
4. Add config fields for base SD checkpoint, ControlNet initialization, precision, resolution, gradient accumulation, and local-only/download policy.
5. Provide smoke tests that instantiate config and run a tiny no-download/dummy forward path.
6. Do not run full training in this implementation task unless bounded by later task instructions.

## Acceptance Criteria
- [ ] SD + ControlNet integration code imports with installed dependencies.
- [ ] Config documents checkpoints and conditioning.
- [ ] Existing baseline/fallback behavior remains intact.
- [ ] Mask-head output contract is compatible with evaluation.

## Output
- Modified files: [`/root/autodl-tmp/project/local_deblur/models/`, `/root/autodl-tmp/project/local_deblur/training/`, `/root/autodl-tmp/project/configs/`, `/root/autodl-tmp/project/.cursor/tasks/task-015.md`]
- Result summary: write under `## Result`

## Result

Files changed:
- `local_deblur/models/sd_controlnet.py`
- `local_deblur/models/__init__.py`
- `local_deblur/training/sd_controlnet.py`
- `local_deblur/training/__init__.py`
- `configs/model.yaml`
- `configs/sd_controlnet.yaml`
- `tests/test_sd_controlnet_smoke.py`
- `.cursor/tasks/task-015.md`

Implementation summary:
- Added a dedicated SD + ControlNet adapter using `diffusers.ControlNetModel` and `StableDiffusionControlNetImg2ImgPipeline` behind explicit local-only checkpoint loading.
- Added `SDControlNetConfig` with base SD checkpoint, optional ControlNet checkpoint, `local_files_only`, `allow_downloads`, dtype/precision, `image_size`, conditioning channels, and mask-head settings.
- Added ControlNet condition image utilities. RGB mapping is documented as `R=blurred grayscale/context`, `G=blur mask`, and `B=segmentation if available, otherwise inverse-mask background context`.
- Added `ControlNetAuxMaskHead` and `SDControlNetMaskOutput` with `mask_logits` and `mask_prob`, matching existing evaluation contracts. It is documented as an auxiliary head trained beside the ControlNet latent/noise objective because stable ControlNet feature hooks are not exposed uniformly across diffusers versions.
- Added training-prep helpers that build SD ControlNet tensors and auxiliary mask-head loss inputs without running training.

Acceptance criteria:
- SD + ControlNet integration imports passed for `ControlNetModel`, `StableDiffusionControlNetImg2ImgPipeline`, `SDControlNetConfig`, and `StableDiffusionControlNetLocalDeblurPipeline`.
- Config documents checkpoints, download policy, precision, conditioning shape, and mask-head behavior in `configs/sd_controlnet.yaml` and `configs/model.yaml`.
- Existing baseline/fallback imports still pass for `ConditionalLocalDeblurNet` and `LocalDeblurPipeline.load(dry_run=True)`.
- Mask-head output contract returns `mask_logits` and `mask_prob` with shape `[B, 1, H, W]`.
- Smoke checks build a ControlNet condition image from `output/synthetic_smoke/manifest.json` and verify RGB/tensor/mask-head shapes without checkpoint loading.

Commands:
- `python -m pytest "tests/test_sd_controlnet_smoke.py"` (not run: active environment has no `pytest` module)
- `python -m py_compile "local_deblur/models/sd_controlnet.py" "local_deblur/training/sd_controlnet.py" "tests/test_sd_controlnet_smoke.py"`
- `python - <<'PY' ... diffusers import, baseline import, config, manifest condition image, tensor, mask-head, and training-batch smoke checks ... PY`

Constraint check:
- No model checkpoints were downloaded.
- No training was run.
- No ReLoBlur data or requirement was added.
- Existing `ConditionalLocalDeblurNet` and fallback pipeline behavior were not replaced.
