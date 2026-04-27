# Task 007: Documentation And Usage Examples
_Created by: Domino Planner_
_Assigned to: Executor_
_Status: pending_

## Objective
Document the complete local image deblurring workflow, expected project layout, commands, dataset assumptions, and examples.

## Context
- Relevant files: [`/root/autodl-tmp/project/README.md`, `/root/autodl-tmp/project/docs/dataset_format.md`, `/root/autodl-tmp/project/docs/research_plan.md`, `/root/autodl-tmp/project/docs/evaluation.md`, `/root/autodl-tmp/project/examples/README.md`]
- Current state: Earlier tasks should add project code, configs, scripts, inference examples, and evaluation outputs.
- Dependencies: task-001, task-002, task-003, task-004, task-005, task-006
- Domino assumptions: Documentation should align with `proposal.pdf`: Stable Diffusion + ControlNet-style local deblurring, CLIP image semantic guidance, synthetic pretraining, global blur augmentation, ReLoBlur fine-tuning, PSNR/SSIM evaluation, qualitative comparisons, and ablations.
- User decisions: Complete the entire flow based on the proposal; include practical docs/examples without requiring datasets now; target LBAG baseline context is PSNR 34.71 / SSIM 0.967.

## Instructions
1. Write `README.md` covering project purpose, setup, package layout, quick dry-run commands, data preparation, training, inference, evaluation, and outputs.
2. Add dataset documentation for local blur sample fields: `Ib`, `M`, optional `S`, and sharp target; include COCO/global blur/ReLoBlur layout assumptions.
3. Add a concise research plan doc linking implementation components to proposal ideas, including synthetic pretraining, progressive training, blurred latent initialization, repaint post-processing, qualitative comparisons, and ablations.
4. Add evaluation documentation explaining basic/weighted/aligned PSNR and SSIM, output files, result directory naming, and LBAG baseline context.
5. Add example notes showing dry-run commands and full-data command templates.
6. Keep docs consistent with actual script names and config paths created by earlier tasks.

## Acceptance Criteria
- [ ] `README.md` gives a new user a coherent path from setup to dry-run evaluation.
- [ ] Docs accurately describe dataset fields and optional segmentation.
- [ ] Evaluation docs reflect the required result structure and metrics.
- [ ] Research-plan docs map proposal requirements to implemented scaffold modules.
- [ ] Commands in docs match actual script/module names.

## Output
- Modified files: [`/root/autodl-tmp/project/README.md`, `/root/autodl-tmp/project/docs/dataset_format.md`, `/root/autodl-tmp/project/docs/research_plan.md`, `/root/autodl-tmp/project/docs/evaluation.md`, `/root/autodl-tmp/project/examples/README.md`]
- Result summary: write under `## Result`

## Result
Files changed: `README.md`, `docs/dataset_format.md`, `docs/research_plan.md`, `docs/evaluation.md`, `examples/README.md`.

Acceptance criteria: setup, dry-run commands, dataset fields, research plan, evaluation outputs/metrics, examples, and full-run templates are documented.

Constraint check: docs state the smoke-first gate and require confirmation before full experiments, checkpoint downloads, dataset-scale runs, or substantial GPU training.
