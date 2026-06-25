from pathlib import Path

import train


def test_direct_yaml_config_builds_stage_command(tmp_path):
    config = tmp_path / "exp1.yaml"
    config.write_text(
        """
stage: stage1
overrides:
  - model=laser
  - data=cifar10
output_dir: outputs/exp1/stage1
train:
  max_epochs: 3
  accelerator: cpu
model:
  sparsity_level: 5
""".strip()
    )

    cmd = train.build_command(["--config", str(config)])

    assert cmd[1].endswith("train.py")
    assert cmd[2] == "stage1"
    assert "data=cifar10" in cmd
    assert "output_dir=outputs/exp1/stage1" in cmd
    assert "train.max_epochs=3" in cmd
    assert "train.accelerator=cpu" in cmd
    assert "model.sparsity_level=5" in cmd


def test_facade_yaml_config_accepts_hydra_overrides(tmp_path):
    config = tmp_path / "exp2.yaml"
    config.write_text(
        """
stage: stage2
dataset: imagenet
modality: image
conditioning: class
num_gpus: 2
token_cache_path: /tmp/tokens.pt
overrides:
  ar:
    d_model: 256
    n_layers: 4
  train_ar:
    sample_num_images: 2
""".strip()
    )

    cmd = train.build_command(["--config", str(config)])

    assert cmd[1].endswith("train.py")
    assert cmd[2] == "stage2"
    assert "token_cache_path=/tmp/tokens.pt" in cmd
    assert "train_ar.devices=2" in cmd
    assert "ar.class_conditional=true" in cmd
    assert "ar.d_model=256" in cmd
    assert "ar.n_layers=4" in cmd
    assert "train_ar.sample_num_images=2" in cmd


def _override_value(cmd, key):
    prefix = f"{key}="
    for arg in cmd:
        if arg.startswith(prefix):
            return arg.split("=", 1)[1]
    raise AssertionError(f"missing override {key!r}")


def test_stage1_imagenet_class_command_maps_to_nonpatch_d5_ddp():
    cmd = train.build_command(
        [
            "stage1",
            "--adversarial",
            "true",
            "--num_gpus",
            "8",
            "--num_nodes",
            "4",
            "--dataset",
            "imagenet",
            "--modality",
            "image",
            "--conditioning",
            "class",
        ]
    )

    assert cmd[1].endswith("train.py")
    assert cmd[2] == "stage1"
    assert "model=laser_image_nonpatch_d5" in cmd
    assert "data=imagenet" in cmd
    assert "train.num_nodes=4" in cmd
    assert "train.devices=2" in cmd
    assert "train.strategy=ddp" in cmd
    assert "model.patch_based=false" in cmd
    assert "model.sparsity_level=3" in cmd
    assert "model.adversarial_weight=0.05" in cmd
    assert "train.compute_rfid_after_fit=true" in cmd
    assert "train.rfid_max_samples=0" in cmd
    assert "train.rfid_split=val" in cmd


def test_stage1_vctk_text_command_maps_to_audio_waveform_preset():
    cmd = train.build_command(
        [
            "--stage",
            "stage1",
            "--adversarial",
            "true",
            "--num_gpus",
            "4",
            "--dataset",
            "vctk",
            "--modality",
            "audio",
            "--conditioning",
            "text",
        ]
    )

    assert "model=laser_audio_waveform_nonpatch_d5" in cmd
    assert "data=vctk_waveform" in cmd
    assert "data.audio_representation=waveform" in cmd
    assert "model.patch_based=false" in cmd
    assert "model.adversarial_weight=0.03" in cmd
    assert "train.compute_rfid_after_fit=true" not in cmd


def test_stage2_command_uses_train_py_stage2_selector():
    cmd = train.build_command(
        [
            "stage2",
            "--num_gpus",
            "2",
            "--dataset",
            "imagenet",
            "--modality",
            "image",
            "--conditioning",
            "class",
            "--token_cache_path",
            "/tmp/tokens.pt",
        ]
    )

    assert cmd[1].endswith("train.py")
    assert cmd[2] == "stage2"
    assert "token_cache_path=/tmp/tokens.pt" in cmd
    assert "ar.class_conditional=true" in cmd
    assert "train_ar.devices=2" in cmd


def _assert_imagenet_pipeline_uses_rqvae_style_backbone(config_name, *, sparsity_level):
    config = Path(__file__).resolve().parents[1] / "configs" / config_name

    commands, _, _ = train._pipeline_commands(config)
    command_by_label = {label: cmd for label, cmd in commands}
    stage1_cmd = command_by_label["stage 1"]
    stage1_adv_cmd = command_by_label["stage 1 adversarial"]

    assert _override_value(stage1_cmd, "model.backbone") == "ddpm"
    assert _override_value(stage1_cmd, "model.num_downsamples") == "5"
    assert _override_value(stage1_cmd, "model.channel_multipliers") == "[1,1,2,2,4,4]"
    assert _override_value(stage1_cmd, "model.backbone_latent_channels") == "256"
    assert _override_value(stage1_cmd, "model.attn_resolutions") == "[8]"
    assert _override_value(stage1_cmd, "model.decoder_extra_residual_layers") == "0"
    assert _override_value(stage1_cmd, "model.use_mid_attention") == "true"
    assert _override_value(stage1_cmd, "model.num_hiddens") == "128"
    assert _override_value(stage1_cmd, "model.num_residual_blocks") == "2"
    assert _override_value(stage1_cmd, "model.sparsity_level") == str(sparsity_level)
    assert _override_value(stage1_cmd, "data.batch_size") == "64"
    assert _override_value(stage1_cmd, "train.accumulate_grad_batches") == "2"
    assert _override_value(stage1_cmd, "train.learning_rate") == "4e-05"
    assert _override_value(stage1_cmd, "train.beta") == "0.5"
    assert _override_value(stage1_cmd, "train.beta2") == "0.9"
    assert _override_value(stage1_cmd, "train.warmup_steps") == "5005"
    assert _override_value(stage1_cmd, "train.min_lr_ratio") == "1.0"
    assert _override_value(stage1_cmd, "model.dict_learning_rate") == "4e-05"
    assert _override_value(stage1_cmd, "model.perceptual_weight") == "1.0"
    assert _override_value(stage1_cmd, "model.perceptual_start_step") == "0"
    assert _override_value(stage1_cmd, "model.perceptual_warmup_steps") == "0"
    assert _override_value(stage1_cmd, "model.adversarial_weight") == "0.0"
    assert _override_value(stage1_adv_cmd, "train.accumulate_grad_batches") == "2"
    assert _override_value(stage1_adv_cmd, "model.adversarial_weight") == "0.75"
    assert _override_value(stage1_adv_cmd, "model.disc_learning_rate") == "4e-05"
    assert _override_value(stage1_adv_cmd, "model.discriminator_beta1") == "0.5"
    assert _override_value(stage1_adv_cmd, "model.discriminator_beta2") == "0.9"
    assert "model.backbone=simple" not in stage1_cmd
    assert "model.backbone=simple" not in stage1_adv_cmd

    checkpoint_shape_keys = (
        "model.backbone",
        "model.num_downsamples",
        "model.channel_multipliers",
        "model.backbone_latent_channels",
        "model.attn_resolutions",
        "model.decoder_extra_residual_layers",
        "model.use_mid_attention",
        "model.dropout",
        "model.num_hiddens",
        "model.num_residual_blocks",
        "model.num_residual_hiddens",
        "model.num_embeddings",
        "model.embedding_dim",
        "model.sparsity_level",
        "model.patch_based",
        "model.patch_size",
        "model.patch_stride",
    )
    for key in checkpoint_shape_keys:
        assert _override_value(stage1_adv_cmd, key) == _override_value(stage1_cmd, key)


def test_imagenet_pipeline_adv_continuation_keeps_checkpoint_architecture():
    _assert_imagenet_pipeline_uses_rqvae_style_backbone("imagenet.yaml", sparsity_level=3)


def test_imagenet_scratch_pipeline_uses_rqvae_style_backbone():
    _assert_imagenet_pipeline_uses_rqvae_style_backbone(
        "imagenet_nonpatch_d5_k4_scratch.yaml",
        sparsity_level=4,
    )
