# Context Summary

Workspace: `/root/autodl-tmp/project`

Current user request context: `/domino 继续推进实验`.

## One-Screen Handoff

This is a diffusion-based local image deblurring project. The Domino plan is in `.cursor/domino-plan.md`. The recent continuation found partially downloaded SD + ControlNet assets, completed a bounded SD 1.5 + Tile ControlNet COCO smoke/pretraining run, and updated the task state. ReLoBlur posttraining still cannot proceed until real ReLoBlur data is available locally.

Completed baseline to preserve:
- `ConditionalLocalDeblurNet` evaluated on `output/datasets/coco2017_train_grouped_localblur_5k/manifest.json`
- Result directory: `results/task011_validation_ConditionalLocalDeblurNet-task010_synthetic5k-val_100_0427/`
- PSNR / SSIM: `35.538233 / 0.995112`
- Weighted PSNR / SSIM: `25.249158 / 0.943050`
- Mask BCE / IoU / Dice: `0.293885 / 0.700077 / 0.812461`

Blocked assets:
- ReLoBlur is not local.
- Google Drive access for ReLoBlur failed with `Network is unreachable`.
- ReLoBlur continuation requires local extracted `dataset/` and `masks/` roots, or working Google Drive/Baidu access.

Completed SD + ControlNet smoke:
- Base SD checkpoint: `/root/autodl-tmp/models/sd15-fp16`
- Tile ControlNet checkpoint: `/root/autodl-tmp/models/control-v11f1e-sd15-tile`
- Output: `output/training/sd15_tile_controlnet_coco_pretrain/`
- ControlNet checkpoint: `output/training/sd15_tile_controlnet_coco_pretrain/checkpoint/controlnet/diffusion_pytorch_model.safetensors`
- Auxiliary mask-head checkpoint: `output/training/sd15_tile_controlnet_coco_pretrain/checkpoint/aux_mask_head.pt`
- Final step total/diffusion/mask loss: `0.206055 / 0.096977 / 1.090777`
- Final validation mask loss/BCE/Dice: `0.977106 / 0.625439 / 0.703332`

## Task State

Statuses normalized in task metadata:
- `task-013.md`: completed, SD + ControlNet dependencies installed and import-smoked.
- `task-014.md`: blocked, ReLoBlur collection failed due external network/data access; converter script exists.
- `task-015.md`: completed, SD + ControlNet local-deblur integration and mask-head contract scaffolded.
- `task-016.md`: completed, bounded local SD 1.5 + Tile ControlNet COCO smoke/pretraining produced checkpoint artifacts.
- `task-017.md`: blocked, ReLoBlur posttraining lacks real ReLoBlur data.
- `task-018.md`: completed, comparison/blocker report written without fabricating unavailable metrics.
- `task-019.md`: completed, final verification summary written with design-check limitations.

## Key Files To Read First

- `.cursor/project_state.md`
- `.cursor/domino-plan.md`
- `.cursor/tasks/task-013.md`
- `.cursor/tasks/task-014.md`
- `.cursor/tasks/task-015.md`
- `.cursor/tasks/task-016.md`
- `.cursor/tasks/task-017.md`
- `.cursor/tasks/task-018.md`
- `.cursor/tasks/task-019.md`
- `docs/sd_controlnet_stage_summary.md`
- `docs/final_results.md`
- `docs/data_status.md`
- `output/training/sd15_tile_controlnet_coco_pretrain/training_summary.json`
- `output/training/sd15_tile_controlnet_coco_pretrain/loss_curve.csv`
- `output/training/reloblur_posttrain_blocked.json`
- `results/task011_validation_ConditionalLocalDeblurNet-task010_synthetic5k-val_100_0427/summary.txt`

## Design Check Summary

Multi-condition ControlNet:
- Partial only. RGB ControlNet condition image uses blurred context, mask, and segmentation/background context. The batch carries `Ib`, `M`, `S`, and `target`, and the auxiliary mask head returns `mask_logits`/`mask_prob`. CLIP image encoder guidance is not wired into training, and latent concatenation is not implemented.

COCO synthetic pipeline:
- Partial. The local 5K COCO-derived grouped local-blur dataset is real and validated for the baseline. GoPro/RealBlur augmentation is not runnable because those datasets are absent.

Progressive training:
- Partially executed. COCO SD + ControlNet smoke/pretraining completed locally and saved checkpoint artifacts. ReLoBlur fine-tuning scripts/configs exist, but no ReLoBlur data is available. Blurred latent initialization and repaint are placeholders/hooks.

## Next Actionable Experimental Steps

1. Supply extracted ReLoBlur data with local `dataset/` and `masks/` roots, or restore Google Drive/Baidu access.
2. Run `python scripts/prepare_reloblur_manifest.py --dataset-root <path>/dataset --masks-root <path>/masks --output-dir output/datasets/reloblur --split all --validate-images`.
3. After ReLoBlur manifests exist, run bounded ReLoBlur posttraining using the available COCO SD + ControlNet checkpoint and then standardized evaluation.

## Constraint Check

- No fabricated ReLoBlur data.
- No fabricated SD + ControlNet or ReLoBlur metrics.
- No fallback/baseline result mislabeled as SD + ControlNet.
- No unbounded training.
- Existing baseline artifacts remain the only completed quantitative result.
