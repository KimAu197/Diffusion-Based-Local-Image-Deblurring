# Task 004: Training Scripts And Configuration
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: pending_

## Objective
Add training scripts and configuration for synthetic/global pretraining and ReLoBlur fine-tuning using the shared data and model APIs.

## Context
- Relevant files: [`/root/autodl-tmp/project/local_deblur/training/__init__.py`, `/root/autodl-tmp/project/local_deblur/training/losses.py`, `/root/autodl-tmp/project/local_deblur/training/trainer.py`, `/root/autodl-tmp/project/scripts/train.py`, `/root/autodl-tmp/project/configs/train_pretrain.yaml`, `/root/autodl-tmp/project/configs/train_reloblur.yaml`]
- Current state: Task 001 should provide project utilities; task 002 should provide data utilities; task 003 should provide the pipeline/model interface.
- Dependencies: task-001, task-002, task-003
- Domino assumptions: Training is progressive: synthetic/global pretraining followed by ReLoBlur fine-tuning; full compute target is A100 40GB; scripts must include dry-run/smoke behavior when data/checkpoints are unavailable.
- User decisions: Deliver training scripts as part of the end-to-end research scaffold; do not require real datasets for basic verification.

## Instructions
1. Implement a small trainer module that wires dataset loading, model/pipeline setup, optimizer/loss placeholders, checkpoint paths, and logging.
2. Add loss helpers appropriate for image restoration scaffolding, such as masked L1/Charbonnier-style loss and optional perceptual/SSIM placeholders guarded by availability.
3. Add `scripts/train.py` CLI with phase selection for `pretrain` and `finetune`, config loading, `--dry-run`, `--max-steps`, `--output-dir`, and seed options.
4. Ensure dry-run training runs a tiny generated batch and writes logs/checkpoint metadata under ignored output paths without large downloads.
5. Add pretraining and ReLoBlur fine-tuning config files with documented dataset/checkpoint fields.
6. Keep evaluation logic out of training files.

## Acceptance Criteria
- [ ] `scripts/train.py --dry-run --max-steps 1` can execute in principle using generated data and fallback model components.
- [ ] Training config files distinguish synthetic/global pretraining from ReLoBlur fine-tuning.
- [ ] Checkpoint and log outputs are routed to ignored directories.
- [ ] Training code consumes shared data/model APIs instead of duplicating them.
- [ ] Full-training options are documented in config without forcing full training during smoke tests.

## Output
- Modified files: [`/root/autodl-tmp/project/local_deblur/training/__init__.py`, `/root/autodl-tmp/project/local_deblur/training/losses.py`, `/root/autodl-tmp/project/local_deblur/training/trainer.py`, `/root/autodl-tmp/project/scripts/train.py`, `/root/autodl-tmp/project/configs/train_pretrain.yaml`, `/root/autodl-tmp/project/configs/train_reloblur.yaml`]
- Result summary: write under `## Result`

## Result
Files changed: `local_deblur/training/`, `scripts/train.py`, `configs/train_pretrain.yaml`, `configs/train_reloblur.yaml`.

Acceptance criteria: loss helpers, smoke trainer, phase-aware training CLI, dry-run checkpoint metadata, and separate pretrain/ReLoBlur configs are implemented.

Constraint check: non-dry-run training is blocked pending user confirmation; outputs are routed under ignored `output/`.
