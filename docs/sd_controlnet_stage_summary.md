# SD + ControlNet Stage Summary

## Goal

This stage attempted to move beyond the compact `ConditionalLocalDeblurNet` baseline and toward the proposal's intended Stable Diffusion + ControlNet workflow:

1. install and verify SD + ControlNet dependencies,
2. collect ReLoBlur,
3. pretrain SD + ControlNet on the COCO synthetic local-blur set,
4. posttrain/fine-tune on ReLoBlur,
5. evaluate the posttrained model.

## Completed

- Installed and verified `diffusers`, `transformers`, `accelerate`, `safetensors`, `huggingface_hub`, and `gdown`.
- Added SD + ControlNet integration code:
  - `local_deblur/models/sd_controlnet.py`
  - `local_deblur/training/sd_controlnet.py`
  - `configs/sd_controlnet.yaml`
- Added a bounded COCO SD + ControlNet training entry point:
  - `scripts/train_sd_controlnet_coco.py`
  - `configs/train_sd_controlnet_coco.yaml`
- Added ReLoBlur manifest conversion:
  - `scripts/prepare_reloblur_manifest.py`

## Check Against Requested Design

- (a) Multi-condition ControlNet architecture: partially implemented. The code builds an RGB ControlNet spatial condition from blurred-image context, blur mask, and segmentation/background context. The training batch also carries `Ib`, `M`, `S`, and `target`, and the auxiliary mask-head output contract is ready. However, CLIP image-encoder semantic guidance is only verified/import-ready, not wired into training in place of text prompts. Additional conditions are not yet concatenated with diffusion latents; the current diffusers smoke path passes them through the ControlNet condition image.
- (b) Synthetic data pipeline for pre-training: partially implemented and used for COCO. The processed 5K dataset uses COCO instance masks, grouped motion-object masks, motion/Gaussian/defocus blur variants, and mask-centered 512 x 512 crops. GoPro and RealBlur are documented as unavailable local sources, but global blur augmentation with arbitrary-shape and object-aware masks is not implemented as a runnable data path yet.
- (c) Progressive training strategy: partially executed. The intended order is synthetic/COCO pretraining followed by ReLoBlur fine-tuning. A bounded SD 1.5 + Tile ControlNet COCO smoke/pretraining run completed locally and produced a ControlNet checkpoint plus an auxiliary mask-head checkpoint. ReLoBlur fine-tuning is still blocked by missing local ReLoBlur data. Blurred-input latent initialization and repaint-style post-processing are represented as documented hooks/placeholders rather than completed SD training behavior.

## Blocked External Assets

### ReLoBlur

Official source: `https://github.com/LeiaLi/ReLoBlur`

- Train Google Drive folder: `1rAPKzhhRjztj7Utbb00BJLSVaPC-1Jua`
- Test Google Drive folder: `1nYj4e7TSXeqBsUZxLvoay_JLZ7wxdNmC`
- Mask Google Drive folder: `1-4YerKKlDydgoBeZbiV0_XR9iJLKbLXI`
- Baidu codes: train `49nb`, test `nmcy`, masks `98mw`
- License/access: academic use, CC BY-NC-SA 4.0.

The attempted Google Drive download failed in this environment with `Network is unreachable`.

### SD + ControlNet COCO Smoke

The partially downloaded SD + ControlNet assets were completed/validated locally:

- Base SD checkpoint: `/root/autodl-tmp/models/sd15-fp16`
- Tile ControlNet checkpoint: `/root/autodl-tmp/models/control-v11f1e-sd15-tile`

A bounded 2-step local smoke run completed with no baseline fallback:

- Output directory: `output/training/sd15_tile_controlnet_coco_pretrain/`
- ControlNet checkpoint: `checkpoint/controlnet/diffusion_pytorch_model.safetensors`
- Auxiliary mask-head checkpoint: `checkpoint/aux_mask_head.pt`
- Final total/diffusion/mask loss: `0.206055 / 0.096977 / 1.090777`
- Final validation mask loss/BCE/Dice: `0.977106 / 0.625439 / 0.703332`

This is a smoke/pretraining validation artifact, not a converged SD + ControlNet benchmark.

## Available Baseline Result

The completed proposal-ready baseline remains:

- Model: `ConditionalLocalDeblurNet`
- Dataset: `output/datasets/coco2017_train_grouped_localblur_5k/manifest.json`
- Checkpoint: `output/training/final_baseline_task010/best.pt`
- Result directory: `results/task011_validation_ConditionalLocalDeblurNet-task010_synthetic5k-val_100_0427/`
- PSNR / SSIM: `35.538233 / 0.995112`
- Weighted PSNR / SSIM: `25.249158 / 0.943050`
- Mask BCE / IoU / Dice: `0.293885 / 0.700077 / 0.812461`

These numbers are a synthetic PyTorch baseline, not SD + ControlNet or ReLoBlur results.

## Required To Continue

To complete ReLoBlur posttraining:

1. Place extracted ReLoBlur data locally with `dataset/` and `masks/` roots.
2. Run:

```bash
python scripts/prepare_reloblur_manifest.py \
  --dataset-root /path/to/ReLoBlur/dataset \
  --masks-root /path/to/ReLoBlur/masks \
  --output-dir output/datasets/reloblur \
  --split all \
  --validate-images
```

3. Fine-tune/posttrain on the ReLoBlur manifests using the available COCO SD + ControlNet checkpoint and evaluate with the standardized pipeline.
