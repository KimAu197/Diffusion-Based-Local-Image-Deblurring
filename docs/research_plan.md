# Research Plan

The scaffold maps the proposal into a staged local deblurring workflow.

## Model Direction

The intended full model is Stable Diffusion with ControlNet-style spatial conditioning. Inputs include the blurred image, blur mask, optional segmentation, blurred latent initialization, and image-semantic guidance. A lightweight mask head is attached to ControlNet encoder mid-level features so the model predicts which pixels are blurred before restoration. The current implementation includes those interfaces and a deterministic fallback so smoke runs do not need checkpoints.

## Training Direction

Training is progressive:

- Synthetic/global pretraining uses generated or adapted blur masks to learn local restoration.
- ReLoBlur fine-tuning uses real locally blurred pairs when available.
- Losses start with masked L1 and Charbonnier restoration terms plus a BCE blur-mask loss. Optional perceptual or SSIM extensions remain guarded by dependency availability.

## Post-Processing

The pipeline preserves background outside `M` and smooths mask boundaries. This represents the proposal's repaint-style restoration step while remaining deterministic in smoke mode.

## Experiments

Planned full experiments include qualitative comparisons, LBAG baseline context, ablations for mask/segmentation/latent initialization, mask-head supervision, and PSNR/SSIM plus mask IoU/Dice/BCE reporting. Full experiments, checkpoint downloads, and substantial GPU training must be confirmed by the user before execution.
