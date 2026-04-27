# Task 008: Final Review And Scoped Debugging
_Created by: Domino Planner_
_Assigned to: Reviewer_
_Status: pending_

## Objective
Review the completed scaffold end to end, run lightweight smoke checks, inspect generated outputs, and identify or fix only scoped issues needed to satisfy the workflow.

## Context
- Relevant files: [`/root/autodl-tmp/project/.gitignore`, `/root/autodl-tmp/project/requirements.txt`, `/root/autodl-tmp/project/pyproject.toml`, `/root/autodl-tmp/project/local_deblur/`, `/root/autodl-tmp/project/evaluation/`, `/root/autodl-tmp/project/scripts/`, `/root/autodl-tmp/project/configs/`, `/root/autodl-tmp/project/docs/`, `/root/autodl-tmp/project/README.md`, `/root/autodl-tmp/project/examples/`]
- Current state: Tasks 001 through 007 should have implemented the project scaffold and docs.
- Dependencies: task-001, task-002, task-003, task-004, task-005, task-006, task-007
- Domino assumptions: Final verification must inspect code and generated result files, not only rely on command exit codes; repairs must be surgical and limited to issues that block acceptance.
- User decisions: Test before delivery; loop until output is correct when feasible; do not ask the user to manually run anything; report unrelated issues instead of silently fixing them.

## Instructions
1. Re-read `.cursor/domino-plan.md` and all completed task results.
2. Inspect actual files for consistency with the proposal and user constraints.
3. Run import/smoke commands that do not require real datasets or checkpoints, including dry-run data prep, dry-run inference, and dry-run evaluation if implemented.
4. Read generated `summary.txt`, `metrics.csv`, `answer.json`, and `logging.log` from the dry-run evaluation result and check whether values and contents are reasonable.
5. Check `.gitignore` for required entries and confirm generated outputs are routed under ignored directories.
6. If small scoped bugs block acceptance, fix them and document the fixes. If larger design issues remain, report them clearly without broad refactoring.
7. Write a reviewer-style result with verdict, criteria review, additional findings, verification evidence, and constraint check.

## Acceptance Criteria
- [ ] Final review explicitly checks all tasks against `.cursor/domino-plan.md`.
- [ ] Lightweight smoke checks run without real datasets or pretrained checkpoints.
- [ ] Dry-run evaluation output files are manually inspected and summarized.
- [ ] Any fixes are limited to acceptance-blocking issues.
- [ ] Result includes `Verdict`, `Criteria review`, `Additional findings`, `Verification`, and `Constraint check`.

## Output
- Modified files: [`/root/autodl-tmp/project/.gitignore`, `/root/autodl-tmp/project/requirements.txt`, `/root/autodl-tmp/project/pyproject.toml`, `/root/autodl-tmp/project/local_deblur/`, `/root/autodl-tmp/project/evaluation/`, `/root/autodl-tmp/project/scripts/`, `/root/autodl-tmp/project/configs/`, `/root/autodl-tmp/project/docs/`, `/root/autodl-tmp/project/README.md`, `/root/autodl-tmp/project/examples/`]
- Result summary: write under `## Result`

## Result

### Verdict
Smoke verification passed. The project scaffold is complete enough to prepare synthetic dry-run data, run a one-step dry-run training loop, run fallback inference, and run standardized dry-run evaluation without real datasets, checkpoints, downloads, or full GPU experiments.

### Criteria review
- Final review checked the implemented scaffold against `.cursor/domino-plan.md`, including the new smoke-first/full-experiment-confirmation gate.
- Lightweight smoke commands ran successfully without real datasets or pretrained checkpoints.
- Dry-run evaluation produced and was manually inspected for `logging.log`, `metrics.csv`, `summary.txt`, `answer.json`, and `log/dry_run_000_prediction.png`.
- Generated outputs are under ignored `output/` and `results/` directories.
- No acceptance-blocking scoped fixes were needed during final review.

### Additional findings
- The diffusion/ControlNet path is intentionally scaffolded and local-only; checkpoint-backed training or inference still requires user-provided data/checkpoint choices.
- Dry-run metrics are high because the deterministic fallback preserves most background and operates on one synthetic sample. The report correctly states that LBAG PSNR/SSIM is reference context, not an achieved result.

### Verification
- Ran `python -m compileall -q local_deblur evaluation scripts`.
- Ran `python scripts/prepare_synthetic_data.py --dry-run --count 1 --image-size 64 --output-dir output/synthetic_verify`.
- Ran `python scripts/train.py --dry-run --max-steps 1 --output-dir output/training_verify`.
- Ran `python scripts/infer.py --dry-run --output output/inference/verify.png`.
- Ran `python -m evaluation.eval_pipeline --model fallback-local-deblur --round smokeVerify --dataset dry-run --count 1 --mode standard --detailed true --dry-run`.
- Inspected `/root/autodl-tmp/project/results/smokeVerify_fallback-local-deblur_dry-run_1_0427/metrics.csv`, `summary.txt`, `answer.json`, and `logging.log`.
- Read linter diagnostics for `local_deblur`, `evaluation`, and `scripts`; no linter errors were reported.

### Constraint check
- No full experiment, real dataset run, checkpoint download, or substantial GPU job was launched.
- Full experiment execution is paused until explicit user confirmation.
- Changes remained scoped to the proposal-driven project scaffold, docs, configs, task records, and smoke outputs.
