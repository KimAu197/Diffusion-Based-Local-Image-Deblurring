# Task 011: Proposal-Ready Evaluation And Report Artifacts
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: pending_

## Objective
Generate standardized evaluation outputs for the trained conditional local-deblur baseline, including restoration metrics, mask-head metrics, summaries, `answer.json`, and qualitative comparison grids.

## Context
- Relevant files: [`/root/autodl-tmp/project/evaluation/eval_pipeline.py`, `/root/autodl-tmp/project/local_deblur/eval/`, `/root/autodl-tmp/project/scripts/evaluation.sh`, `/root/autodl-tmp/project/configs/evaluation.yaml`, `/root/autodl-tmp/project/results/`, `/root/autodl-tmp/project/output/training/`]
- Current state: The existing evaluation scaffold supports required result files and PSNR/SSIM variants for dry-run/fallback behavior. The final result stage needs evaluation against the trained PyTorch checkpoint from task-010.
- Dependencies: task-009, task-010
- Domino assumptions: The proposal-ready result should be evaluated on the validation split of the local 5K synthetic dataset. It should not claim ReLoBlur performance unless real ReLoBlur data is actually used.
- User decisions: Required root files are `logging.log`, `metrics.csv`, `summary.txt`, `answer.json`, and `log/`; output directory format is `results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/`; metrics must include PSNR/SSIM weighted/aligned and blur-mask IoU/Dice/BCE.

## Instructions
1. Extend evaluation helpers, not `evaluation/eval_pipeline.py`, for trained checkpoint loading and final dataset split loading.
2. Keep `evaluation/eval_pipeline.py` as the evaluation loop only.
3. Compute restoration metrics: PSNR and SSIM in basic, mask-weighted, and aligned forms.
4. Compute mask-head metrics: BCE, IoU, and Dice against ground-truth blur masks.
5. Save per-sample and aggregate values in `metrics.csv` with clear column names.
6. Save detailed predictions and paths in `answer.json`, including input, target, prediction, ground-truth mask, predicted mask, and metric values where available.
7. Generate visual grids under `log/` showing at least blurred input, mask, restored output, target, absolute error or crop, and predicted mask.
8. Write `summary.txt` with a concise proposal-ready interpretation, including dataset name, split, count, checkpoint path, restoration metric averages, mask metric averages, and limitations.
9. Ensure `logging.log` records command arguments, checkpoint, dataset, split, and any skipped/failed samples.

## Acceptance Criteria
- [ ] Evaluation uses the trained conditional PyTorch checkpoint from task-010.
- [ ] Result directory follows the standardized naming format and contains all required root files plus `log/`.
- [ ] `metrics.csv` includes PSNR/SSIM basic, weighted, aligned, and mask BCE/IoU/Dice.
- [ ] `answer.json` contains detailed prediction records suitable for inspection.
- [ ] Visual grids are generated and manually inspected for plausibility.
- [ ] `summary.txt` clearly labels the result as a synthetic 5K validation baseline, not full diffusion/ReLoBlur performance.

## Output
- Modified files: [`/root/autodl-tmp/project/local_deblur/eval/`, `/root/autodl-tmp/project/evaluation/eval_pipeline.py`, `/root/autodl-tmp/project/scripts/evaluation.sh`, `/root/autodl-tmp/project/configs/evaluation.yaml`, `/root/autodl-tmp/project/results/`]
- Result summary: write under `## Result`

## Result
Files changed:
- `evaluation/eval_pipeline.py`
- `local_deblur/eval/__init__.py`
- `local_deblur/eval/checkpoint.py`
- `local_deblur/eval/dataset.py`
- `local_deblur/eval/runner.py`
- `local_deblur/eval/outputs.py`
- `configs/evaluation.yaml`
- `scripts/evaluation.sh`

Artifacts:
- Result directory: `results/task011_validation_ConditionalLocalDeblurNet-task010_synthetic5k-val_100_0427/`
- Required root files: `logging.log`, `metrics.csv`, `summary.txt`, `answer.json`
- Visual/grid output: `results/task011_validation_ConditionalLocalDeblurNet-task010_synthetic5k-val_100_0427/log/`
- Inspected visual grid: `results/task011_validation_ConditionalLocalDeblurNet-task010_synthetic5k-val_100_0427/log/coco_semantic_000000_grid.png`

Commands run:
```bash
python -m py_compile evaluation/eval_pipeline.py local_deblur/eval/checkpoint.py local_deblur/eval/dataset.py local_deblur/eval/runner.py local_deblur/eval/outputs.py
bash scripts/evaluation.sh
```

Evaluation configuration:
- Checkpoint: `output/training/final_baseline_task010/best.pt`
- Manifest: `output/datasets/coco2017_train_grouped_localblur_5k/manifest.json`
- Split: validation split from `split_seed: 2026`, `val_fraction: 0.1`
- Count: 100 bounded validation samples from the 500-sample validation split
- Image size: 256
- Device recorded in `answer.json`: `cuda`

Metrics summary from `summary.txt`:
- PSNR / SSIM: 35.538233 / 0.995112
- Weighted PSNR / SSIM: 25.249158 / 0.943050
- Aligned PSNR / SSIM: 35.538233 / 0.995112
- Aligned weighted PSNR / SSIM: 25.249158 / 0.943050
- Mask BCE / IoU / Dice: 0.293885 / 0.700077 / 0.812461

Validation checks:
- `metrics.csv` includes per-sample basic, weighted, aligned, aligned-weighted restoration metrics plus mask BCE/IoU/Dice.
- `answer.json` contains 100 detailed prediction records with generated input, target, GT mask, predicted mask, prediction, visual grid paths, metrics, checkpoint metadata, and source sample metadata.
- `logging.log` records command arguments, resolved checkpoint, manifest, split, seed, image size, and per-sample evaluation; no `ERROR`, `Traceback`, or failed-sample records were found.
- Visual grid inspection confirmed the expected panels: blurred input, GT mask, predicted mask, restored output, target, and absolute error.

Constraint check:
- No downloads were used or introduced.
- Final metrics use the trained PyTorch checkpoint, not fallback-only evaluation.
- `evaluation/eval_pipeline.py` remains loop-oriented; split loading, checkpoint inference, metrics, and serialization stay in helper modules.
- The evaluation process exited normally and no long-running process was left behind.
