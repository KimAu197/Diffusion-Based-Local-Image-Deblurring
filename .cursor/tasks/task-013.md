# Task 013: SD ControlNet Dependency Setup
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: completed_

## Objective
Install and verify the Python dependencies needed for Stable Diffusion + ControlNet integration without breaking existing baseline code.

## Context
- Relevant files: [`/root/autodl-tmp/project/requirements.txt`, `/root/autodl-tmp/project/configs/model.yaml`, `/root/autodl-tmp/project/docs/`]
- Current state: PyTorch is installed; recent checks showed `diffusers` and `transformers` unavailable even though they are listed in `requirements.txt`.
- Dependencies: task-012
- Domino assumptions: Use the active conda/base Python environment unless another environment is clearly active; follow pip-first installation; update `requirements.txt` if new helper packages are added.
- User decisions: Proceed toward SD + ControlNet training; avoid unnecessary downloads in dependency setup; preserve existing baseline artifacts.

## Instructions
1. Identify the active Python/conda environment and CUDA/Torch status.
2. Install missing packages needed for SD + ControlNet: `diffusers`, `transformers`, `accelerate`, `safetensors`, and a download helper such as `gdown` if needed.
3. Prefer `pip`; use an Aliyun mirror only if downloads are slow or failing.
4. Update `requirements.txt` if it lacks any installed package.
5. Run import smoke checks for `torch`, `diffusers`, `transformers`, `accelerate`, `safetensors`, and `huggingface_hub`.
6. Do not download model checkpoints in this task.

## Acceptance Criteria
- [ ] Dependency import smoke check passes.
- [ ] `requirements.txt` includes all needed packages.
- [ ] Existing fallback/baseline imports still pass.
- [ ] No model checkpoint download is performed.

## Output
- Modified files: [`/root/autodl-tmp/project/requirements.txt`, `/root/autodl-tmp/project/.cursor/tasks/task-013.md`]
- Result summary: write under `## Result`

## Result

Files changed:
- `requirements.txt`
- `.cursor/tasks/task-013.md`

Acceptance criteria:
- Dependency import smoke check passed for `torch`, `diffusers`, `transformers`, `accelerate`, `safetensors`, `huggingface_hub`, and `gdown`.
- `diffusers` ControlNet imports passed: `ControlNetModel` and `StableDiffusionControlNetPipeline`.
- CLIP image imports passed: `CLIPVisionModel` and `CLIPImageProcessor`.
- `requirements.txt` now includes `safetensors`, `huggingface_hub`, and `gdown`.
- Existing local baseline imports still pass for `local_deblur` and `ConditionalLocalDeblurNet`.

Commands:
- `python -m pip install --no-input diffusers transformers accelerate safetensors huggingface_hub gdown`
- `python - <<'PY' ... import smoke checks ... PY`

Constraint check:
- Used the active `/root/miniconda3/bin/python` environment.
- Used pip-first installation and did not download model checkpoints.
- No training or dataset download was run in this task.
