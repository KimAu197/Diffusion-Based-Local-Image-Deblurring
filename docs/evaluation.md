# Evaluation

Run the smoke evaluation with:

```bash
python -m evaluation.eval_pipeline --model fallback-local-deblur --round smoke --dataset dry-run --count 1 --mode standard --detailed true --dry-run
```

The shell wrapper is `scripts/evaluation.sh` and uses the same module entry point.

## Output Structure

Each run creates:

`results/<ROUND>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/`

Required files:

- `logging.log`: full evaluation log.
- `metrics.csv`: per-sample metric table.
- `summary.txt`: human-readable summary and LBAG reference context.
- `answer.json`: detailed predictions and metadata without binary image payloads.
- `log/`: intermediate outputs such as prediction images and predicted blur masks.

The `_<HHMM>` suffix is added only when the date-level result directory already exists.

## Metrics

Basic PSNR/SSIM are computed over the whole image. Weighted PSNR/SSIM use mask-aware weighting to emphasize the blurred region. Aligned metrics apply a small integer-shift search before computing the same measures. Mask debug metrics compare the predicted blur mask against the provided mask with IoU, Dice, and BCE.

LBAG reference context is PSNR 34.71 / SSIM 0.967. This is documented as a comparison target, not as a claimed result for smoke runs.

Full ReLoBlur evaluation requires user confirmation before execution.
