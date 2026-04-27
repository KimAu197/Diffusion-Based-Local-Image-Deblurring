from pathlib import Path

import torch

from local_deblur.data.datasets import ManifestDeblurDataset
from local_deblur.models.sd_controlnet import (
    ControlNetAuxMaskHead,
    SDControlNetConfig,
    StableDiffusionControlNetLocalDeblurPipeline,
    build_controlnet_condition_from_sample,
    controlnet_condition_to_tensor,
)
from local_deblur.training.sd_controlnet import prepare_sd_controlnet_training_batch


def test_sd_controlnet_config_is_local_only_by_default():
    config = SDControlNetConfig(base_sd_checkpoint=None, controlnet_checkpoint=None, allow_downloads=False)

    assert config.local_files_only is True
    assert config.conditioning_channels == 3
    assert config.mask_head_enabled is True


def test_build_condition_image_from_existing_manifest_sample():
    manifest = Path("output/synthetic_smoke/manifest.json")
    sample = ManifestDeblurDataset(manifest)[0]

    condition = build_controlnet_condition_from_sample(sample, image_size=64)
    tensor = controlnet_condition_to_tensor(condition)

    assert condition.mode == "RGB"
    assert condition.size == (64, 64)
    assert tensor.shape == (1, 3, 64, 64)
    assert float(tensor[:, 1].max()) > 0.0


def test_auxiliary_mask_head_contract_shapes_without_checkpoints():
    manifest = Path("output/synthetic_smoke/manifest.json")
    sample = ManifestDeblurDataset(manifest)[0]
    config = SDControlNetConfig(image_size=64, mask_head_enabled=True)

    pipeline = StableDiffusionControlNetLocalDeblurPipeline.from_config(config, load_checkpoints=False)
    condition = pipeline.prepare_condition(sample)
    mask_output = pipeline.predict_mask_from_condition(condition)

    assert pipeline.pipeline is None
    assert mask_output is not None
    assert mask_output.mask_logits.shape == (1, 1, 64, 64)
    assert mask_output.mask_prob.shape == (1, 1, 64, 64)
    assert torch.all(mask_output.mask_prob >= 0.0)
    assert torch.all(mask_output.mask_prob <= 1.0)


def test_training_batch_preparation_without_downloads():
    manifest = Path("output/synthetic_smoke/manifest.json")
    sample = ManifestDeblurDataset(manifest)[0]

    batch = prepare_sd_controlnet_training_batch(sample, SDControlNetConfig(image_size=64))

    assert batch.Ib.shape == (1, 3, 64, 64)
    assert batch.M.shape == (1, 1, 64, 64)
    assert batch.S.shape == (1, 1, 64, 64)
    assert batch.target.shape == (1, 3, 64, 64)
    assert batch.controlnet_condition.shape == (1, 3, 64, 64)
    assert batch.mask_logits is not None
    assert batch.mask_logits.shape == (1, 1, 64, 64)


def test_aux_mask_head_direct_forward():
    head = ControlNetAuxMaskHead(in_channels=3, hidden_channels=8)
    output = head(torch.zeros(1, 3, 16, 16))

    assert output.mask_logits.shape == (1, 1, 16, 16)
    assert output.mask_prob.shape == (1, 1, 16, 16)
