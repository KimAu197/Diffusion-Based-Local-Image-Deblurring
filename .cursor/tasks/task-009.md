# Task 009: Trainable Conditional Local Deblur Baseline
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: pending_

## Objective
Implement a feasible trainable PyTorch conditional local-deblur baseline with an image restoration output and TA-requested blur-mask prediction head.

## Context
- Relevant files: [`/root/autodl-tmp/project/local_deblur/models/`, `/root/autodl-tmp/project/local_deblur/data/`, `/root/autodl-tmp/project/local_deblur/training/`, `/root/autodl-tmp/project/scripts/train.py`, `/root/autodl-tmp/project/configs/`]
- Current state: The existing project has diffusion/ControlNet-style scaffold and fallback probes, but current loss curves are fallback-only and not true trainable model results.
- Dependencies: task-001, task-002, task-003, task-004, task-008
- Domino assumptions: `diffusers`, `transformers`, and real ReLoBlur assets are unavailable locally; proposal-ready results should come from a compact trainable PyTorch baseline on the existing 5K synthetic grouped dataset. The diffusion/ControlNet framing remains the intended full direction, not a dependency for this result stage.
- User decisions: Continue from current progress; do not train in the Planner task; avoid large network downloads unless absolutely necessary; use input fields `Ib`, `M`, optional `S`, and `target`; include a blur-mask prediction head.

## Instructions
1. Inspect the existing model, dataset, and training scaffold before editing.
2. Add a compact model such as `ConditionalLocalDeblurNet` under `local_deblur/models/`.
3. The model must accept channels for blurred image `Ib`, mask `M`, and optional segmentation/semantic condition `S` when available.
4. The model must output a restored RGB image and a predicted blur mask/logit map.
5. Reuse existing dataset abstractions where possible; add only the minimal adapter needed to load the processed manifest at `output/datasets/coco2017_train_grouped_localblur_5k/manifest.json`.
6. Add restoration and mask-head losses in the training path, including image reconstruction loss and mask BCE/Dice-compatible terms.
7. Keep the implementation lightweight enough to train on the available RTX PRO 6000 without external downloads.
8. Preserve existing fallback/dry-run behavior for earlier scaffold commands.

## Acceptance Criteria
- [ ] A trainable PyTorch model class is implemented and importable.
- [ ] Forward pass accepts `Ib`, `M`, optional `S`, and returns restored image plus mask prediction.
- [ ] Training loss combines image restoration and blur-mask supervision.
- [ ] Existing smoke/fallback paths remain compatible.
- [ ] No large pretrained model download is introduced.

## Output
- Modified files: [`/root/autodl-tmp/project/local_deblur/models/`, `/root/autodl-tmp/project/local_deblur/data/`, `/root/autodl-tmp/project/local_deblur/training/`, `/root/autodl-tmp/project/scripts/train.py`, `/root/autodl-tmp/project/configs/`]
- Result summary: write under `## Result`

## Result
Files changed:
- `local_deblur/models/conditional_unet.py`
- `local_deblur/models/__init__.py`
- `local_deblur/data/tensor_dataset.py`
- `local_deblur/data/__init__.py`
- `local_deblur/training/losses.py`
- `local_deblur/training/trainer.py`
- `scripts/train.py`
- `configs/train_pretrain.yaml`
- `configs/model.yaml`

Acceptance criteria:
- Implemented importable `ConditionalLocalDeblurNet`, a compact PyTorch U-Net-style baseline with RGB restoration and trainable mask prediction heads.
- Forward pass accepts `Ib`, `M`, and optional `S`, returning `restored`, `mask_logits`, and `mask_prob`.
- Added `TensorManifestDeblurDataset` and `sample_to_tensors` to adapt the existing `ManifestDeblurDataset` manifest samples into BCHW-compatible torch tensors with optional resizing.
- Added trainable loss helpers combining masked image restoration terms with BCE-with-logits and Dice-compatible mask supervision.
- Preserved the old fallback smoke path and added an opt-in `--trainable-baseline` path for later experiment tasks.

Constraint check:
- No full training was run.
- No downloads or `diffusers` dependency were introduced for the trainable baseline.
- Did not modify task-010/task-011/task-012.
- Quick import/forward smoke command:
```bash
python - <<'PY'
import torch
from local_deblur.models import ConditionalLocalDeblurNet
model = ConditionalLocalDeblurNet(base_channels=8)
Ib = torch.rand(2, 3, 32, 32)
M = torch.rand(2, 1, 32, 32)
out = model(Ib, M)
print(out.restored.shape, out.mask_logits.shape, out.mask_prob.min().item() >= 0.0, out.mask_prob.max().item() <= 1.0)
PY
```
- Additional smoke commands run:
  - `python - <<'PY' ... TensorManifestDeblurDataset first sample ... PY`
  - `python scripts/train.py --dry-run --max-steps 1 --output-dir output/training/task009_fallback_smoke`
- Smoke outputs were inspected: tensor shapes matched expectations and fallback `loss_curve.csv`/checkpoint metadata were valid.
