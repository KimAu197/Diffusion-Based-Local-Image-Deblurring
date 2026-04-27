# Task 003: Model And Pipeline Wrapper
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: pending_

## Objective
Implement a PyTorch/diffusers-compatible local deblurring pipeline wrapper with lightweight fallbacks for environments without pretrained checkpoints.

## Context
- Relevant files: [`/root/autodl-tmp/project/local_deblur/models/__init__.py`, `/root/autodl-tmp/project/local_deblur/models/conditioning.py`, `/root/autodl-tmp/project/local_deblur/models/pipeline.py`, `/root/autodl-tmp/project/local_deblur/models/fallback.py`, `/root/autodl-tmp/project/local_deblur/models/postprocess.py`, `/root/autodl-tmp/project/configs/model.yaml`]
- Current state: Task 001 provides shared config/logging utilities; task 002 provides sample/data contracts.
- Dependencies: task-001, task-002
- Domino assumptions: Research approach is Stable Diffusion + ControlNet-style local deblurring; spatial condition includes blurred region; CLIP image encoder semantic guidance replaces text prompts; input image/background, blur mask, and segmentation are concatenated with diffusion latent; blurred latent initialization and repaint-style post-processing should be represented as hooks.
- User decisions: Use PyTorch/diffusers-compatible components but keep the project runnable without datasets or pretrained checkpoints now.

## Instructions
1. Create model conditioning utilities that assemble image, mask, optional segmentation, and latent conditioning tensors from the task-002 sample contract.
2. Implement a `LocalDeblurPipeline` wrapper with a clear interface for `load`, `prepare_inputs`, and `__call__`/`generate`.
3. Use diffusers components when available and requested, but provide a deterministic fallback path that returns a plausible background-preserving composite for smoke tests.
4. Add placeholder integration points for ControlNet, CLIP image encoder guidance, blurred latent initialization, and repaint post-processing without forcing downloads.
5. Add post-processing helpers that preserve background outside the mask and smooth boundary transitions.
6. Add `configs/model.yaml` with model/checkpoint fields, dry-run fallback settings, and conditioning options.

## Acceptance Criteria
- [ ] Model modules import without installed checkpoints.
- [ ] Fallback pipeline can process a dry-run sample and return an image/tensor with the expected shape.
- [ ] Pipeline API is usable by later training, inference, and evaluation tasks.
- [ ] Conditioning code explicitly supports blur mask and optional segmentation.
- [ ] Comments/docstrings distinguish implemented fallback behavior from full checkpoint-backed diffusion behavior.

## Output
- Modified files: [`/root/autodl-tmp/project/local_deblur/models/__init__.py`, `/root/autodl-tmp/project/local_deblur/models/conditioning.py`, `/root/autodl-tmp/project/local_deblur/models/pipeline.py`, `/root/autodl-tmp/project/local_deblur/models/fallback.py`, `/root/autodl-tmp/project/local_deblur/models/postprocess.py`, `/root/autodl-tmp/project/configs/model.yaml`]
- Result summary: write under `## Result`

## Result
Files changed: `local_deblur/models/`, `configs/model.yaml`.

Acceptance criteria: conditioning, deterministic fallback deblurring, background-preserving post-processing, and a diffusers-compatible pipeline wrapper are implemented.

Constraint check: model imports do not require checkpoints or downloads; diffusers use is optional and local-only.
