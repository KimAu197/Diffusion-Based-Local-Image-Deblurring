# Final Results Summary

This result is a proposal-ready validation of the compact `ConditionalLocalDeblurNet` baseline, not a full diffusion/ControlNet or ReLoBlur benchmark. It demonstrates that the project scaffold can train a conditional local-deblurring model with a blur-mask head, produce quantitative metrics, and save qualitative inspection artifacts on the local synthetic dataset.

## Dataset And Checkpoint

- Dataset: `output/datasets/coco2017_train_grouped_localblur_5k/manifest.json`
- Split: deterministic validation split, `split_seed: 2026`, `val_fraction: 0.1`
- Train/validation samples: 4500 / 500
- Evaluation subset: first 100 samples from the validation split
- Image size: 256
- Model: `ConditionalLocalDeblurNet`, `base_channels: 16`
- Checkpoint: `output/training/final_baseline_task010/best.pt`

## Training Metrics

Training artifacts are under `output/training/final_baseline_task010/`.

- Steps: 30 bounded training steps on CUDA
- Final train total loss: 0.094630
- Final train restoration loss: 0.031474
- Final train mask loss: 0.631558
- Best validation total loss: 0.105776
- Final validation restoration loss: 0.042429
- Final validation mask loss: 0.633471
- Final validation PSNR: 33.9480
- Final validation mask IoU: 0.733247

The loss curve is plausible for a short trainable run: total training loss decreases from 0.179438 to 0.094630, and validation loss decreases from 0.150479 at step 5 to 0.105776 at step 30.

## Evaluation Metrics

Evaluation artifacts are under `results/task011_validation_ConditionalLocalDeblurNet-task010_synthetic5k-val_100_0427/`.

- Samples evaluated: 100
- PSNR / SSIM: 35.538233 / 0.995112
- Weighted PSNR / SSIM: 25.249158 / 0.943050
- Aligned PSNR / SSIM: 35.538233 / 0.995112
- Aligned weighted PSNR / SSIM: 25.249158 / 0.943050
- Mask BCE / IoU / Dice: 0.293885 / 0.700077 / 0.812461

The required result files are present: `logging.log`, `metrics.csv`, `summary.txt`, `answer.json`, and `log/` visual outputs. `answer.json` contains 100 prediction records with input, target, prediction, ground-truth mask, predicted mask, visual-grid paths, metrics, checkpoint metadata, and sample metadata.

## Qualitative Check

Representative grids such as `log/coco_semantic_000000_grid.png`, `log/coco_semantic_000005_grid.png`, and `log/coco_semantic_000033_grid.png` show the expected panels: blurred input, ground-truth mask, predicted mask, restored output, target, and amplified absolute error. The images are nonblank and visually aligned with the synthetic local-blur task. The predicted masks generally localize the blurred regions but are soft and sometimes include background texture, which is expected for this short baseline run.

## Limitations

- The results use a synthetic 5K COCO-derived local-blur dataset, not real ReLoBlur data.
- The model is a compact PyTorch conditional baseline, not the intended Stable Diffusion + ControlNet architecture.
- The run is intentionally bounded to 30 training steps and 100 validation samples, so it is evidence of pipeline viability rather than convergence.
- High full-image PSNR/SSIM is helped by the local nature of the blur and preserved background; weighted metrics better reflect the edited region.

## Recommended Next Step

Use this baseline as the sanity-checked reference point. A bounded SD 1.5 + Tile ControlNet COCO smoke/pretraining run now exists under `output/training/sd15_tile_controlnet_coco_pretrain/`, but it is not a converged benchmark. The next experimental step is to add real ReLoBlur data, run bounded ReLoBlur posttraining from the available COCO SD + ControlNet checkpoint, and then report ReLoBlur metrics alongside the synthetic baseline.
