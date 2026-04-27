# Domino Plan: Diffusion-Based Local Image Deblurring

_Created by: Domino Planner_
_Strategy: Sequential_
_Workspace: `/root/autodl-tmp/project`_

## Goal
Create a practical, minimal, coherent Python research project scaffold for Diffusion-Based Local Image Deblurring based on `proposal.pdf`, covering data preparation, synthetic blur generation, model/pipeline wrapping, training, inference, evaluation, documentation, and final review.

## Chosen Strategy
Sequential.

This project should be built in dependency order because later tasks need stable interfaces from earlier tasks:

1. Establish package structure, project metadata, requirements, and ignore rules.
2. Add data contracts and synthetic blur utilities.
3. Add model and pipeline wrappers around PyTorch/diffusers-compatible components with dry-run fallbacks.
4. Add training and inference scripts that consume the shared data/model APIs.
5. Add evaluation package and required shell runner following the user's evaluation architecture.
6. Add documentation and examples.
7. Run final review/debug pass across the complete scaffold.

## Assumptions
- The repo is currently mostly empty and should receive a minimal but runnable scaffold rather than a full research implementation.
- Datasets and pretrained checkpoints may not be present now, so scripts must support dry-run, tiny synthetic smoke tests, or graceful validation modes.
- The implementation should be compatible with PyTorch and diffusers-style components while avoiding mandatory large downloads during smoke tests.
- The core research design is Stable Diffusion + ControlNet-style local deblurring with spatial conditioning, CLIP image semantic guidance, input/background preservation, blur mask support, optional segmentation maps, progressive training, and ReLoBlur evaluation.
- The user's evaluation architecture rule is binding: `evaluation/` contains the evaluation loop only, utilities/model/data helpers live outside it, `scripts/evaluation.sh` has the required parameter block, and outputs follow `results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/`.
- `.gitignore` must exclude `cache/`, `output/`, `results/`, `*.log`, `*.pyc`, and `*.tmp` while preserving existing entries.
- Edits should stay scoped to project code, docs, and project metadata.
- No task should require real datasets, real checkpoints, or manual user setup to pass its basic acceptance criteria.
- Current proposal-ready results should be obtained from the available 5K grouped synthetic local-blur dataset using a feasible trainable PyTorch conditional local-deblur baseline with a blur-mask prediction head, because `diffusers`, `transformers`, and real ReLoBlur assets are not locally available.
- The final result stage should preserve the diffusion/ControlNet framing as the intended full research direction while clearly labeling the produced numbers as a trained conditional PyTorch baseline result.

## User Decisions
- Use workspace `/root/autodl-tmp/project`.
- Base the workflow on `proposal.pdf` for Diffusion-Based Local Image Deblurring.
- Create a complete end-to-end project scaffold, not just notes.
- Do not implement actual project code inside the Planner task beyond planning and task files.
- Use the standardized evaluation architecture and output structure described in the user rules.
- Use A100 40GB as the target compute assumption for documented full training, while keeping smoke tests lightweight.
- Target LBAG baseline context is PSNR 34.71 / SSIM 0.967 for evaluation comparison documentation.
- Implement all project code and run only a lightweight smoke run first; before full training/evaluation experiments, stop and ask the user to confirm operational choices.
- Continue from current progress and obtain final proposal-ready results without asking the user to manually run old slash commands.
- Use `/root/autodl-tmp/project/output/datasets/coco2017_train_grouped_localblur_5k/manifest.json` as the primary available dataset for the final result stage.
- Do not run training in the Planner task; create executable task specifications for Executor/Reviewer roles.

## Final Results Stage
The scaffold and synthetic data generation are complete enough to move from fallback-only probes to a real trainable baseline. The final result stage should implement and run a compact PyTorch model, tentatively `ConditionalLocalDeblurNet`, that consumes blurred image `Ib`, blur mask `M`, and optional segmentation `S`, then predicts both the restored sharp image and the blur mask.

This stage intentionally avoids depending on unavailable `diffusers`, `transformers`, or ReLoBlur downloads. It should produce proposal-ready evidence from the local 5K synthetic grouped dataset: train/validation loss curves, restoration metrics, mask-head metrics, qualitative grids, and a concise summary that can be cited in a proposal. Documentation should state that the diffusion/ControlNet design remains the full intended method, while the reported artifacts are from the feasible trained conditional baseline.

Required final artifacts:
- `output/training/<ROUND_NAME>/loss_curve.csv` and a loss curve PNG.
- `results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/logging.log`.
- `results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/metrics.csv`.
- `results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/summary.txt`.
- `results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/answer.json`.
- `results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/log/` with visual comparison grids.

The evaluation must include PSNR/SSIM in basic, mask-weighted, and aligned forms, plus mask IoU, Dice, and BCE for the TA-requested blur-mask prediction head.

## SD + ControlNet ReLoBlur Stage
The next stage targets the full proposal direction requested by the user: integrate and train a Stable Diffusion + ControlNet-style local deblurring model, collect ReLoBlur, pretrain on the existing COCO synthetic local-blur data, then posttrain/fine-tune on real ReLoBlur data.

This stage should remain practical and checkpointed. First, set up `diffusers`, `transformers`, `accelerate`, `safetensors`, and dataset download helpers. Next, collect and normalize ReLoBlur into the same `Ib`, `M`, `S`, `target` manifest contract. Then add SD + ControlNet integration with spatial conditioning and the TA-requested blur-mask head. Finally, run bounded COCO pretraining and ReLoBlur posttraining when data and checkpoint access are available, evaluating each checkpoint with the standard result format.

If ReLoBlur download is blocked by access, license, Google Drive quota, or Baidu credentials, the workflow should stop with one concrete question instead of guessing. If pretrained SD/ControlNet checkpoints are gated or unavailable, use local-only downloads first and report the exact blocked asset.

New task queue:
- `task-013.md`: Dependency/environment setup for SD + ControlNet.
- `task-014.md`: ReLoBlur collection, conversion, and manifest validation.
- `task-015.md`: SD + ControlNet local-deblur integration with mask head.
- `task-016.md`: COCO synthetic pretraining run/config/checkpoint.
- `task-017.md`: ReLoBlur posttraining/fine-tuning run/config/checkpoint.
- `task-018.md`: Evaluation/report comparing baseline, COCO pretrain, and ReLoBlur posttrain.
- `task-019.md`: Final verification and updated proposal summary.

## Implementation Phases
### Phase 1: Project Foundation
- Create package skeleton, CLI/module layout, configuration files, requirements, and `.gitignore` updates.
- Define common conventions for paths, optional dependencies, logging, dry-run behavior, and result directories.

### Phase 2: Data And Synthetic Blur
- Add dataset abstractions for local blur samples with `Ib`, `M`, optional `S`, and sharp target.
- Add synthetic blur utilities for COCO-style instance masks, global blur augmentation, arbitrary-shape masks, and mask-centered 512x512 crops preserving aspect ratio.
- Provide smoke-test fixtures generated in memory or under ignored output paths.

### Phase 3: Model And Pipeline
- Add a PyTorch/diffusers-compatible local deblurring pipeline wrapper.
- Include lightweight fallback modules so import, configuration, and dry-run inference work without pretrained checkpoints.
- Represent spatial conditions, CLIP image encoder guidance, latent concatenation conditions, blurred latent initialization hooks, and repaint post-processing hooks.

### Phase 4: Training
- Add training script(s) for synthetic/global pretraining and ReLoBlur fine-tuning.
- Include config-driven CLI parameters, dry-run mode, checkpoint directory handling, logging, and minimal smoke execution.

### Phase 5: Inference
- Add inference CLI for a blurred image, mask, optional segmentation map, and output image.
- Support dry-run/fallback mode and full mode when compatible checkpoints are provided.

### Phase 6: Evaluation
- Add `evaluation/` package where the evaluation pipeline owns only the evaluation loop.
- Move metrics, alignment, output directory naming, serialization, data loading, and model invocation helpers into separate modules.
- Add `scripts/evaluation.sh` with required parameter comments and call format.
- Guarantee `logging.log`, `metrics.csv`, `summary.txt`, `answer.json`, and `log/` are created in the required result directory.

### Phase 7: Documentation And Examples
- Add `README.md` and example configs/commands explaining dataset layout, synthetic data preparation, training phases, inference, evaluation, outputs, and expected ReLoBlur baseline comparison.

### Phase 8: Final Review And Debug
- Re-read the implemented files, run available smoke tests/commands, inspect produced result artifacts, and repair only scoped issues.
- Stop after smoke verification and ask for confirmation before launching any full experiment that may download checkpoints, use real datasets, or consume substantial GPU time.

## Task Queue
- `task-001.md`: Project/package skeleton, requirements, and `.gitignore`
- `task-002.md`: Data contracts and synthetic blur utilities
- `task-003.md`: Model and pipeline wrapper
- `task-004.md`: Training scripts and configuration
- `task-005.md`: Inference script and examples
- `task-006.md`: Evaluation package and `scripts/evaluation.sh`
- `task-007.md`: Documentation and usage examples
- `task-008.md`: Final review, smoke tests, and scoped debugging
- `task-009.md`: Trainable conditional PyTorch local-deblur baseline and mask head
- `task-010.md`: Train/validation split and final experiment runner/config
- `task-011.md`: Proposal-ready evaluation/report generation with metrics and visual grids
- `task-012.md`: Final verification and proposal result summary

## Acceptance Criteria
- `.cursor/tasks/` contains standard task specs with explicit dependencies, acceptance criteria, relevant files, `Domino assumptions`, and `User decisions`.
- The plan uses strategy `Sequential`.
- Planned tasks cover project/package skeleton, requirements, `.gitignore`, data and synthetic blur utilities, model/pipeline wrapper, training, inference, evaluation package, `scripts/evaluation.sh`, documentation/examples, and final review/debug.
- Evaluation requirements are represented exactly enough for executors to implement without reinterpreting the user rule.
- Planner does not implement actual project code beyond this plan and task specs.
- Smoke run is completed and inspected before any full experiment starts.
- Full experiment execution is gated on explicit user confirmation.
- Final result tasks define a path to real trainable baseline metrics using the existing 5K synthetic dataset without requiring large downloads.

## Result
Planning updated for the final proposal-ready result stage. Added the working assumption that the immediate deliverable should be a trained conditional PyTorch local-deblur baseline with a blur-mask head on the existing 5K synthetic grouped dataset, while preserving the diffusion/ControlNet design as the full research direction.

Created follow-on task specifications `task-009.md` through `task-012.md` to cover implementation, experiment setup, evaluation/report generation, and final verification. The Planner task did not launch training or require manual slash-command execution.
