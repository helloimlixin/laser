#!/usr/bin/env python3
"""Verify a trained checkpoint under the aligned classifier before resuming."""
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(os.environ['LASER_RQ_BASE'])
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
sys.path[:0] = [str(BASE/'source'), str(BASE/'upstream'), str(BASE/'runtime')]
import torch
from src import rq_upstream
from src.models.laser_joint_execution import AlignedJointLinear
from src.training.laser_joint_recipe import JointLaserRecipe
from src.training import k4_checkpoint_io as checkpoint_io
from src.training.rq_imagenet_runtime import write_json, TarImageNet
from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.img_datasets.transforms import create_transforms


def main():
    torch.set_num_threads(4);torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    profile = JointLaserRecipe(BASE)
    out = Path('/workspace/Projects/laser/outputs')/profile.run_id
    source = checkpoint_io._checkpoint_upload_source((out/'checkpoints/last.pt').resolve())
    saved = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    profile.verify_resume(saved)
    step = saved['global_step']
    assert step >= 626 and saved['next_epoch'] >= 1
    assert all(int(s['step']) == step for s in saved['optimizer']['state'].values())
    assert all(torch.isfinite(s['exp_avg']).all() and torch.isfinite(s['exp_avg_sq']).all()
               for s in saved['optimizer']['state'].values())
    assert len(saved['rng_state_by_rank']) == 4
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    arch = augment_arch_defaults(config.arch);arch.vocab_size = config.dataset.vocab_size
    with torch.device('meta'): model = profile.model(arch)
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    assert type(model) is rq_upstream.RQTransformer
    native_keys = list(model.state_dict())
    parameter_ids = [id(p) for p in model.parameters()]
    model.eval().to(device)
    aux, _ = profile.tokenizer(device)
    data = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
    images = torch.stack([data[i][0] for i in range(2)]).to(device)
    with torch.inference_mode():
        _, codes = profile.targets(aux, images)
        labels = torch.tensor([0,999], device=device)
        native = model(codes, aux, labels, amp=True)
        # Capture IDs after the device transfer, then check adapter identity.
        before = [id(p) for p in model.parameters()]
        profile.adapt(model, aux)
        assert before == [id(p) for p in model.parameters()]
        assert native_keys == list(model.state_dict())
        assert isinstance(model.classifier.linear, AlignedJointLinear)
        aligned = model(codes, aux, labels, amp=True)
        p, q = native.float().log_softmax(-1), aligned.float().log_softmax(-1)
        kl = float((p.exp()*(p-q)).sum(-1).max())
        assert kl < 1.e-4
        del native, aligned, p, q
        generated = profile.sample(model, aux, 128, torch.arange(128, device=device)%1000)
        assert generated.shape == (128,8,8,4)
        assert ((generated >= 0) & (generated < 65537)).all()
        for chunk in generated.split(8):
            decoded = profile.decode(aux, chunk)
            assert decoded.shape == (len(chunk),3,256,256) and torch.isfinite(decoded).all()
    write_json(out/'performance/aligned-resume-verification.json', dict(passed=True,
        checkpoint_global_step=step, all_adam_ages=step, next_epoch=saved['next_epoch'],
        next_microbatch=saved['next_microbatch'], rng_streams=4, strict_model_reload=True,
        parameter_keys_unchanged=True, parameter_objects_preserved=True,
        semantic_recipe_unchanged=True, softmax_classes=65537, internal_projection_width=65600,
        maximum_fp16_kl=kl, generated_images=128, finite_decoded_images=True,time=time.time()))
    print('Aligned classifier checkpoint and generation verified', step, kl, flush=True)


if __name__ == '__main__': main()
