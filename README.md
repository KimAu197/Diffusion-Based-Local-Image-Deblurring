# Diffusion-Based Local Image Deblurring

This repository contains the implementation scaffold, experiment artifacts, and final analysis report for a proposal-style study on diffusion-based local image deblurring. The main research direction is to restore only locally blurred regions while preserving the original background, using blur masks, Stable Diffusion, ControlNet-style spatial conditioning, and ReLoBlur fine-tuning.

## Project Summary

The project explored a multi-stage local deblurring pipeline:

- Synthetic COCO local-blur data generation with blur masks.
- Stage 1 blur-mask prediction from blurred input images.
- Stable Diffusion 1.5 + Tile ControlNet local restoration.
- COCO pretraining followed by ReLoBlur fine-tuning.
- NAFNet preprocessing and direct mask-region restoration as a stronger structure-preserving baseline.
- Low-noise diffusion refinement experiments with reduced timestep ranges.

The final conclusion is analytical rather than performance-claim oriented: the full SD + ControlNet route was implemented end to end, but it did not produce sufficiently stable real local deblurring. The main failure mode is the trade-off between diffusion-model generation and image fidelity. Motion blur loses information, so diffusion models can hallucinate plausible details that are not faithful to the original scene.

## Final Report

The final report is in `report/`:

- `report/experiment_summary.tex`: LaTeX source for Overleaf.
- `report/experiment_summary.md`: Markdown source.
- `report/experiment_summary.html`: HTML version.
- `report/metrics_overview.csv`: compact metric table used by the report.
- `report/assets/`: representative figures used by the report.

For Overleaf, upload `report/experiment_summary.tex` and the `report/assets/` directory, then compile with pdfLaTeX.

## Key Findings

- The lightweight synthetic baseline performs well on COCO synthetic validation, showing that local blur supervision is learnable in-distribution.
- Stage 1 mask prediction works reasonably on COCO but transfers less reliably to ReLoBlur.
- Direct SD + ControlNet restoration is unstable because the model can change local semantics and texture instead of faithfully reconstructing the original content.
- NAFNet-style preprocessing helps preserve structure, but it does not fully solve severe local blur.
- Low-timestep diffusion is a more promising direction because it treats diffusion as local refinement rather than full reconstruction.

## Repository Layout

- `local_deblur/`: data contracts, model wrappers, training helpers, inference, metrics, and evaluation utilities.
- `evaluation/`: evaluation loop entry point.
- `scripts/`: data preparation, training, inference, and evaluation scripts.
- `configs/`: training, inference, and evaluation configurations.
- `docs/`: project notes and data/status summaries.
- `report/`: final report source and selected report assets.
- `results/`: selected result `log/` visualizations and `metrics.csv` files only.

## Artifact Policy

Large generated artifacts are intentionally excluded from GitHub:

- Checkpoints and model weights are not uploaded.
- `output/`, `data/`, `cache/`, `incomplete/`, `wandb/`, `runs/`, and `artifacts/` are ignored.
- In `results/`, only `log/` visualizations and `metrics.csv` are intended to be tracked.
- Full prediction dumps such as `answer.json`, summaries, logs, and checkpoints should remain local.

This keeps the repository small enough for GitHub while preserving the visual evidence and quantitative metrics needed to understand the experiments.

## Setup

```bash
pip install -r requirements.txt
```

The codebase contains both lightweight smoke paths and heavier diffusion/ControlNet paths. Heavy training or full evaluation requires the corresponding datasets, pretrained checkpoints, and GPU resources.

## Example Commands

Dry-run smoke checks:

```bash
python scripts/prepare_synthetic_data.py --dry-run --count 1
python scripts/train.py --dry-run --max-steps 1
python scripts/infer.py --dry-run --mask-output output/inference/dry_run_predicted_mask.png
python -m evaluation.eval_pipeline --model fallback-local-deblur --round smoke --dataset dry-run --count 1 --mode standard --detailed true --dry-run
```

Full experiments should be run only when the required local datasets and checkpoints are available.
