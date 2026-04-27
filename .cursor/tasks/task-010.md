# Task 010: Train/Val Split And Final Experiment Runner
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: completed_

## Objective
Create the final training experiment configuration and runner path that trains the conditional PyTorch baseline on the existing 5K grouped synthetic local-blur dataset with a reproducible train/validation split.

## Context
- Relevant files: [`/root/autodl-tmp/project/configs/`, `/root/autodl-tmp/project/scripts/train.py`, `/root/autodl-tmp/project/local_deblur/training/`, `/root/autodl-tmp/project/local_deblur/data/`, `/root/autodl-tmp/project/output/datasets/coco2017_train_grouped_localblur_5k/manifest.json`]
- Current state: Dataset generation is complete for 5000 grouped semantic local-blur samples at 512x512 with fields `Ib`, `M`, `S`, and `target`. Dataset stats are motion 1678, gaussian 1653, defocus 1669, mask mean min/avg/max 0.0200/0.1030/0.3496, and no black-border failures in checked samples.
- Dependencies: task-009
- Domino assumptions: The final experiment should produce real trainable-model loss curves rather than fallback probes. Use local compute and local data; avoid introducing dataset or checkpoint downloads.
- User decisions: Output artifacts should live under ignored `output/` and `results/`; standardized output naming should be preserved; final results should be proposal-ready.

## Instructions
1. Add a deterministic train/validation split for the 5K manifest, such as 90/10 or another clearly documented split.
2. Add a final experiment config for `ConditionalLocalDeblurNet` with dataset path, split seed, batch size, learning rate, epoch/step budget, image size, loss weights, and output directory.
3. Update or add a training command path that writes checkpoints, `loss_curve.csv`, and a loss curve PNG under `output/training/<ROUND_NAME>/`.
4. Ensure the loss curve records train restoration loss, train mask loss, total loss, and validation metrics/loss where feasible.
5. Include resumable checkpoint behavior if simple to add, but keep the task focused on producing one clean final run.
6. Document the exact command an Executor should run for the final experiment.
7. Do not start training from this task spec itself; the assigned Executor may run it when executing the task.

## Acceptance Criteria
- [ ] Final train/validation split is deterministic and documented.
- [ ] Final experiment config exists and points to the 5K manifest.
- [ ] Training writes `loss_curve.csv`, a loss curve PNG, and model checkpoint(s) under `output/training/<ROUND_NAME>/`.
- [ ] Loss curve contains real trainable-model values, not fallback-only probes.
- [ ] The final training command is explicit and reproducible.

## Output
- Modified files: [`/root/autodl-tmp/project/configs/`, `/root/autodl-tmp/project/scripts/train.py`, `/root/autodl-tmp/project/local_deblur/training/`, `/root/autodl-tmp/project/local_deblur/data/`, `/root/autodl-tmp/project/output/training/`]
- Result summary: write under `## Result`

## Result
Files changed:
- `local_deblur/data/tensor_dataset.py`
- `local_deblur/data/__init__.py`
- `local_deblur/training/losses.py`
- `local_deblur/training/trainer.py`
- `scripts/train.py`
- `configs/train_final_baseline.yaml`

Acceptance criteria:
- Added deterministic manifest splitting via `deterministic_split_indices(total, val_fraction, seed)`. The final config uses `split_seed: 2026` and `val_fraction: 0.1`, producing 4500 train samples and 500 validation samples from `output/datasets/coco2017_train_grouped_localblur_5k/manifest.json`.
- Added `configs/train_final_baseline.yaml` for `ConditionalLocalDeblurNet` with the 5K manifest, split seed, batch size 4, learning rate 0.0005, 30-step bounded run budget, image resize 256, loss weights, and output directory `output/training/final_baseline_task010`.
- Updated `scripts/train.py` and `LocalDeblurTrainer` so `--trainable-baseline` trains the PyTorch model on CUDA when available and writes `best.pt`, `last.pt`, `loss_curve.csv`, `loss_curve.png` when matplotlib is available, `training_summary.json`, and `training.log` under `output/training/<round>/`.
- `loss_curve.csv` now records train total/restoration/mask losses, mask BCE/Dice terms, validation total/restoration/mask losses, validation PSNR, and validation mask IoU at validation intervals.
- No checkpoint or dataset downloads were introduced. `diffusers` is not used by the trainable baseline path.

Exact commands run:
```bash
python -m py_compile scripts/train.py local_deblur/training/trainer.py local_deblur/training/losses.py local_deblur/data/tensor_dataset.py
python - <<'PY'
from local_deblur.data.tensor_dataset import TensorManifestDeblurDataset, deterministic_split_indices
manifest='output/datasets/coco2017_train_grouped_localblur_5k/manifest.json'
dataset=TensorManifestDeblurDataset(manifest, image_size=32)
train, val = deterministic_split_indices(len(dataset), val_fraction=0.1, seed=2026)
print(len(dataset), len(train), len(val), train[:3], val[:3])
PY
python scripts/train.py --config configs/train_final_baseline.yaml --trainable-baseline --max-steps 2 --output-dir output/training/task010_smoke
python scripts/train.py --config configs/train_final_baseline.yaml --trainable-baseline
```

Artifacts:
- Smoke run: `output/training/task010_smoke/`
- Final run: `output/training/final_baseline_task010/`
- Final files: `best.pt`, `last.pt`, `last.json`, `loss_curve.csv`, `loss_curve.png`, `training_summary.json`, `training.log`

Final bounded-run metrics from `training_summary.json`:
- Device: `cuda`
- Steps: 30
- Train split / validation split: 4500 / 500
- Final train total loss: 0.09462973475456238
- Best validation total loss: 0.1057758778333664
- Final validation restoration loss: 0.04242873750627041
- Final validation mask loss: 0.6334713697433472
- Final validation PSNR: 33.9479978397777
- Final validation mask IoU: 0.733246922492981

Run-budget note:
- The final config resizes samples to 256 and uses a 30-step budget so the run remains practical in this environment. This is a bounded trainable-baseline result for task-010, not a full convergence run.
