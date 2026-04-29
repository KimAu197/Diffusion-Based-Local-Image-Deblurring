# Local Deblur

Minimal research scaffold for diffusion-style local image deblurring. The project represents the proposal flow: local blur masks, optional segmentation, ControlNet-style spatial conditioning, CLIP-image-guidance placeholders, progressive pretraining/fine-tuning, repaint-like post-processing, blur-mask prediction for debugging, and PSNR/SSIM evaluation.

## Smoke-First Gate

All commands are designed to run first in `--dry-run` mode with generated samples. Do not launch full experiments, dataset-scale preparation, checkpoint downloads, or substantial GPU training until the main agent asks the user to confirm those operations.

## Setup

```bash
pip install -r requirements.txt
```

Heavy packages are optional in code paths. Dry-run commands require only common Python image/numeric packages.

## Quick Smoke Commands

```bash
python scripts/prepare_synthetic_data.py --dry-run --count 1
python scripts/train.py --dry-run --max-steps 1
python scripts/infer.py --dry-run --mask-output output/inference/dry_run_predicted_mask.png
python -m evaluation.eval_pipeline --model fallback-local-deblur --round smoke --dataset dry-run --count 1 --mode standard --detailed true --dry-run
```

Outputs are written under ignored `output/` and `results/` paths.

## Layout

- `local_deblur/data/`: sample contracts, manifest datasets, dry-run data, masks, synthetic blur, crop/transforms.
- `local_deblur/models/`: conditioning, fallback deblurring, post-processing, and diffusers-compatible pipeline wrapper.
- `local_deblur/training/`: smoke trainer and restoration loss helpers.
- `local_deblur/eval/`: metrics, alignment, result output helpers, and evaluation runner utilities.
- `evaluation/eval_pipeline.py`: evaluation loop entry point only.
- `scripts/`: data preparation, training, inference, and evaluation shell runner.
- `configs/`: base, model, data, training, inference, and evaluation configs.

## Full-Data Templates

Full runs require explicit confirmation first.

```bash
python scripts/prepare_synthetic_data.py --coco-images /data/coco/images --coco-instances /data/coco/instances --output-dir output/synthetic
python scripts/train.py --phase pretrain --config configs/train_pretrain.yaml --manifest output/synthetic/manifest.json
python scripts/infer.py --image Ib.png --mask M.png --checkpoint /path/to/local/checkpoint --output output/inference/result.png --mask-output output/inference/predicted_mask.png
```

Evaluation results use `results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/` and include `logging.log`, `metrics.csv`, `summary.txt`, `answer.json`, and `log/`. The `log/` directory includes both restored predictions and predicted blur masks when the mask head is enabled.

## Current Data Status

The current processed dataset is `output/datasets/coco2017_train_grouped_localblur_5k/manifest.json`. It uses COCO instance masks, groups people with nearby carried objects, filters edge/black-border crops, and mixes motion/Gaussian/defocus blur. This existing dataset was generated with a 2% to 35% mask-area range; new synthetic generation defaults to a stricter 5% to 20% range and records blur-kernel parameters in the manifest metadata. See `docs/data_status.md` for details.
