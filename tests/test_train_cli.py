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
