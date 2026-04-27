# Examples

`inference_manifest.json` shows the expected fields for local deblurring samples.

Smoke commands:

```bash
python scripts/prepare_synthetic_data.py --dry-run --count 1
python scripts/infer.py --dry-run --output output/inference/example.png --mask-output output/inference/example_mask.png
python -m evaluation.eval_pipeline --model fallback-local-deblur --round smoke --dataset dry-run --count 1 --mode standard --detailed true --dry-run
```

Full-data template commands require confirmation first:

```bash
python scripts/infer.py --image /path/to/Ib.png --mask /path/to/M.png --segmentation /path/to/S.png --checkpoint /path/to/local/checkpoint --output output/inference/full.png --mask-output output/inference/full_mask.png
```
