# Task 006: Evaluation Package And Runner
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: pending_

## Objective
Implement the standardized evaluation package and `scripts/evaluation.sh` runner with required result directory structure and ReLoBlur-oriented metrics.

## Context
- Relevant files: [`/root/autodl-tmp/project/evaluation/__init__.py`, `/root/autodl-tmp/project/evaluation/eval_pipeline.py`, `/root/autodl-tmp/project/local_deblur/eval/__init__.py`, `/root/autodl-tmp/project/local_deblur/eval/metrics.py`, `/root/autodl-tmp/project/local_deblur/eval/alignment.py`, `/root/autodl-tmp/project/local_deblur/eval/outputs.py`, `/root/autodl-tmp/project/local_deblur/eval/runner.py`, `/root/autodl-tmp/project/scripts/evaluation.sh`, `/root/autodl-tmp/project/configs/evaluation.yaml`]
- Current state: Task 003 should provide model/pipeline API; task 005 should provide inference helpers.
- Dependencies: task-001, task-002, task-003, task-005
- Domino assumptions: Evaluation must compute PSNR and SSIM in basic, weighted, and aligned modes on ReLoBlur-style samples; output directory must be `results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/`; required files are `logging.log`, `metrics.csv`, `summary.txt`, `answer.json`, and `log/`.
- User decisions: Evaluation Python file goes under `evaluation/`; evaluation file only handles the evaluation loop; utilities/model/data functions live in separate modules; shell script under `/scripts` is named `evaluation.sh` and includes required parameters/comments and call format using `python -m evaluation.eval_pipeline`.

## Instructions
1. Create `evaluation/eval_pipeline.py` as the evaluation loop only: parse CLI args, load config, iterate samples, call shared runner/model helpers, and write final outputs.
2. Put metrics, alignment, result-directory naming, serialization, logging setup, and model/data invocation helpers under `local_deblur/eval/` or other non-`evaluation/` modules.
3. Implement PSNR and SSIM metrics with basic, weighted/mask-aware, and aligned variants. Use lightweight dependencies and clear fallbacks where needed.
4. Implement output directory creation with collision handling: append `_<HHMM>` only when the same-name directory exists.
5. Ensure every evaluation run creates root files `logging.log`, `metrics.csv`, `summary.txt`, `answer.json`, and subdirectory `log/`.
6. Add dry-run evaluation support that creates a tiny generated sample and produces reasonable non-empty result files.
7. Add `/root/autodl-tmp/project/scripts/evaluation.sh` with the exact required parameter block semantics: `MODEL`, `ROUND_NAME`, `DATASET`, `COUNT`, `MODE`, `DETAILED`, comments, and `python -m evaluation.eval_pipeline --model "$MODEL" --round "$ROUND_NAME" ...`.
8. Include LBAG baseline context around PSNR 34.71 / SSIM 0.967 in summary/reporting documentation without hard-coding it as achieved performance.

## Acceptance Criteria
- [ ] `evaluation/eval_pipeline.py` contains the evaluation loop and imports helper functions instead of duplicating utility/model/data logic.
- [ ] `scripts/evaluation.sh` exists under `/scripts`, is named exactly `evaluation.sh`, includes required comments/parameters, and calls `python -m evaluation.eval_pipeline`.
- [ ] Dry-run evaluation creates `results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/` with `logging.log`, `metrics.csv`, `summary.txt`, `answer.json`, and `log/`.
- [ ] Metrics include PSNR and SSIM in basic, weighted, and aligned modes.
- [ ] Result files are read back or validated by the task before marking complete.

## Output
- Modified files: [`/root/autodl-tmp/project/evaluation/__init__.py`, `/root/autodl-tmp/project/evaluation/eval_pipeline.py`, `/root/autodl-tmp/project/local_deblur/eval/__init__.py`, `/root/autodl-tmp/project/local_deblur/eval/metrics.py`, `/root/autodl-tmp/project/local_deblur/eval/alignment.py`, `/root/autodl-tmp/project/local_deblur/eval/outputs.py`, `/root/autodl-tmp/project/local_deblur/eval/runner.py`, `/root/autodl-tmp/project/scripts/evaluation.sh`, `/root/autodl-tmp/project/configs/evaluation.yaml`]
- Result summary: write under `## Result`

## Result
Files changed: `evaluation/`, `local_deblur/eval/`, `scripts/evaluation.sh`, `configs/evaluation.yaml`.

Acceptance criteria: evaluation loop is isolated in `evaluation/eval_pipeline.py`; helpers handle metrics, alignment, model/data calls, output naming, and serialization; dry-run creates required result files.

Constraint check: full evaluation is blocked pending user confirmation; metrics include basic, weighted, and aligned PSNR/SSIM; LBAG is included only as reference context.
