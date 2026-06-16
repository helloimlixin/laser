import train


def test_stage1_imagenet_class_command_maps_to_nonpatch_d5_ddp():
    cmd = train.build_command(
        [
            "--stage",
            "1",
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
            "1",
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
