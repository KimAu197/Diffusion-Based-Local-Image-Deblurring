# Project State: Diffusion-Based Local Image Deblurring

Workspace: `/root/autodl-tmp/project`

Last preserved state: Domino MemorySaver after user request `/domino 继续推进实验`.

## Current Verdict

Verifier verdict: pass-with-blockers.

The project has a working scaffold, a completed synthetic PyTorch baseline result, and a completed bounded SD 1.5 + Tile ControlNet COCO smoke/pretraining run. ReLoBlur posttraining remains blocked because real ReLoBlur data is not local. No unavailable ReLoBlur metrics should be reported as completed.

## Domino Plan

Current plan file: `.cursor/domino-plan.md`.

Project goal: diffusion-based local image deblurring with Stable Diffusion + ControlNet-style spatial conditioning, blur-mask prediction, optional segmentation/background context, and eventual ReLoBlur fine-tuning/evaluation.

Current task metadata statuses:
- `task-013.md`: completed
- `task-014.md`: blocked
- `task-015.md`: completed
- `task-016.md`: completed
- `task-017.md`: blocked
- `task-018.md`: completed
- `task-019.md`: completed

## Completed Baseline

Completed available model/result:
- Model: `ConditionalLocalDeblurNet`
- Dataset manifest: `output/datasets/coco2017_train_grouped_localblur_5k/manifest.json`
- Checkpoint: `output/training/final_baseline_task010/best.pt`
- Result directory: `results/task011_validation_ConditionalLocalDeblurNet-task010_synthetic5k-val_100_0427/`
- Evaluation subset: 100 validation samples
- PSNR / SSIM: `35.538233 / 0.995112`
- Weighted PSNR / SSIM: `25.249158 / 0.943050`
- Aligned PSNR / SSIM: `35.538233 / 0.995112`
- Aligned weighted PSNR / SSIM: `25.249158 / 0.943050`
- Mask BCE / IoU / Dice: `0.293885 / 0.700077 / 0.812461`

Important interpretation: these are synthetic COCO-derived PyTorch baseline metrics, not SD + ControlNet metrics and not ReLoBlur metrics.

## Local Data State

Found locally:
- COCO2017 at `/autodl-pub/data/COCO2017`
- COCO2014 at `/autodl-pub/data/COCO14`

Processed dataset:
- Manifest: `output/datasets/coco2017_train_grouped_localblur_5k/manifest.json`
- Samples: `5000`
- Image size: `512 x 512`
- Fields: `Ib`, `M`, `S`, `target`
- Blur variants: motion `1678`, Gaussian `1653`, defocus `1669`
- Dataset stats: `output/datasets/coco2017_train_grouped_localblur_5k/dataset_stats.json`

Not found locally:
- GoPro
- RealBlur
- ReLoBlur

ReLoBlur status:
- Official source: `https://github.com/LeiaLi/ReLoBlur`
- Official Google Drive and Baidu links are documented in `docs/data_status.md` and `docs/sd_controlnet_stage_summary.md`.
- Google Drive access failed with `Network is unreachable`.
- No local ReLoBlur files were found under `/root`, `/root/autodl-tmp`, or `/autodl-pub/data`.
- Converter is ready: `scripts/prepare_reloblur_manifest.py`.
- ReLoBlur continuation requires local extracted `dataset/` and `masks/` roots, or working Google Drive/Baidu access.

## SD + ControlNet State

Dependencies/scaffold/configs/scripts are present:
- Requirements include SD + ControlNet dependencies.
- Verified imports in task notes: `diffusers`, `transformers`, `accelerate`, `safetensors`, `huggingface_hub`, `gdown`, `ControlNetModel`, `StableDiffusionControlNetPipeline`, `CLIPVisionModel`, and `CLIPImageProcessor`.
- Integration code: `local_deblur/models/sd_controlnet.py`
- Training helpers: `local_deblur/training/sd_controlnet.py`
- Configs: `configs/sd_controlnet.yaml`, `configs/train_sd_controlnet_coco.yaml`
- COCO SD + ControlNet script: `scripts/train_sd_controlnet_coco.py`
- ReLoBlur converter: `scripts/prepare_reloblur_manifest.py`

COCO SD + ControlNet pretraining status:
- Completed as a bounded smoke/pretraining validation run.
- Base SD checkpoint: `/root/autodl-tmp/models/sd15-fp16`
- Tile ControlNet checkpoint: `/root/autodl-tmp/models/control-v11f1e-sd15-tile`
- Training output: `output/training/sd15_tile_controlnet_coco_pretrain/`
- Summary: `output/training/sd15_tile_controlnet_coco_pretrain/training_summary.json`
- Loss curve: `output/training/sd15_tile_controlnet_coco_pretrain/loss_curve.csv`
- ControlNet checkpoint: `output/training/sd15_tile_controlnet_coco_pretrain/checkpoint/controlnet/diffusion_pytorch_model.safetensors`
- Auxiliary mask-head checkpoint: `output/training/sd15_tile_controlnet_coco_pretrain/checkpoint/aux_mask_head.pt`
- Final step total/diffusion/mask loss: `0.206055 / 0.096977 / 1.090777`
- Final validation mask loss/BCE/Dice: `0.977106 / 0.625439 / 0.703332`
- The first fp16 optimizer run produced `nan` at step 2 and was replaced by a stable fp32 run at learning rate `1e-6`.
- `used_baseline_fallback` is `false`.

ReLoBlur posttraining status:
- Blocked.
- Required input is missing: local ReLoBlur train/test/mask data.
- Blocked summary: `output/training/reloblur_posttrain_blocked.json`
- `used_fallback` is `false`.

## Requested Design Check

Multi-condition ControlNet:
- Partial.
- Implemented condition packing from blurred-image context, blur mask, and segmentation/background context into an RGB ControlNet condition image.
- Training batch carries `Ib`, `M`, `S`, and `target`.
- Auxiliary mask-head contract returns `mask_logits` and `mask_prob` with shape `[B, 1, H, W]`.
- CLIP image encoder semantic guidance is import-verified/placeheld but not wired into training.
- Extra condition latent concatenation is not implemented.

COCO synthetic pipeline:
- Partial but usable for the current synthetic baseline.
- COCO instance/object-aware local blur generation, grouped motion-object masks, blur variants, and mask-centered crops are implemented and used for the 5K dataset.
- GoPro/RealBlur augmentation is not runnable because those data sources are unavailable locally.

Progressive training:
- Partially executed.
- COCO SD + ControlNet smoke/pretraining completed locally; ReLoBlur posttraining configs/scripts exist but still need real ReLoBlur manifests.
- Blurred latent initialization and repaint-style post-processing are placeholders/hooks, not completed training behavior.

## Documentation

Read and rely on these files for the current factual state:
- `.cursor/domino-plan.md`
- `.cursor/tasks/task-013.md` through `.cursor/tasks/task-019.md`
- `docs/sd_controlnet_stage_summary.md`
- `docs/final_results.md`
- `docs/data_status.md`

## Next Actionable Steps

1. Place extracted ReLoBlur locally with `dataset/` and `masks/` roots, then run `scripts/prepare_reloblur_manifest.py`.
2. After ReLoBlur manifests exist, run bounded ReLoBlur posttraining using the available COCO SD + ControlNet checkpoint.
3. Run standardized evaluation for the ReLoBlur posttrained model.
4. Only after real artifacts exist, update comparison docs with ReLoBlur metrics.

## Constraint Check

- Do not fabricate ReLoBlur data.
- Do not fabricate SD + ControlNet metrics.
- Do not relabel the `ConditionalLocalDeblurNet` baseline as SD + ControlNet.
- Do not run unbounded training.
- Preserve the completed baseline result directory and docs.
