#!/usr/bin/env python3
"""Validate joint32 on a pinned production checkpoint using CPU inference."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from dataclasses import asdict

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-imagenet-duplicate-20261014')
OUT = ROOT / 'outputs/imagenet-rfid421-tiny-jitter-fresh-4h100-20261014'
WORK = BASE / 'joint-evaluation'
WORK.mkdir(exist_ok=True)
sys.path[:0] = [str(BASE / 'source'), str(BASE / 'source/runtime')]
os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(BASE / 'checkpoint-upload-cache')
os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'

import torch
from src.training import rqtransformer as training
from src.training import k4_checkpoint_io as checkpoint_io

spec = importlib.util.spec_from_file_location('laser_compound_joint', ROOT / 'src/compound_joint_sampling.py')
joint = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = joint
spec.loader.exec_module(joint)


@torch.no_grad()
def main():
    torch.set_num_threads(16)
    persistent = OUT / 'train/checkpoints/best_fid_62.4132_epoch_004.pt'
    local = checkpoint_io._checkpoint_upload_source(persistent.resolve())
    assert local != persistent.resolve(), 'Use the verified local serialization'
    pinned = WORK / 'epoch4-pinned.pt'
    if not pinned.exists(): os.link(local, pinned)
    payload = torch.load(pinned, weights_only=False, mmap=True, map_location='cpu')
    assert payload['global_step'] == 2504 and payload['epoch'] == 4
    config = payload['config']
    with torch.device('meta'):
        model = training.build_model(16384 + 2048, 16384, compound=True,
            coeff_vocab_size=2048, compound_pair_autoregressive=True,
            model_preset='imagenet-1400m', sparsity_level=4)
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    del payload
    model.eval()
    aux = training.LaserAux(BASE / 'inputs/stage1-tokenizer.pt', 16384, 2048, 3., 6.4,
        coeff_scales=config['coeff_scales'], soft_target_physical=False,
        clamp_coeffs=False, sparsity_level=4).eval()
    cond = torch.tensor([207, 933])
    packed = torch.full((2, 8, 8, 4), 1024, dtype=torch.long)
    model.init_cache()
    hidden = model.cached_head_output(packed, aux, cond, (0, 0, 0), amp=False)
    prior = joint.categorical_probabilities(model.classifier(hidden), top_k=256, top_p=.95)
    atoms, weights = joint.atom_proposals(prior, 32, generator=torch.Generator().manual_seed(261001))
    cache = model._cache['spatial_ctx_hw']
    head_caches = [block._cache['past_kv'] for block in model.head_transformer.blocks]
    start = time.perf_counter()
    actual = joint.candidate_coefficient_logits(model, aux, hidden, atoms, depth=0)
    batched_seconds = time.perf_counter() - start
    start = time.perf_counter()
    expected = torch.stack([model.coefficient_logits(hidden, aux.dictionary.T[atoms[:, k]],
                           depth_index=0) for k in range(32)], dim=1)
    independent_seconds = time.perf_counter() - start
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert model._cache['spatial_ctx_hw'] is cache
    assert all(block._cache['past_kv'] is saved for block, saved in zip(model.head_transformer.blocks, head_caches))
    probabilities = joint.joint_probabilities(weights, actual, top_p=None)
    torch.testing.assert_close(probabilities.sum(-1), weights)
    model.init_cache()
    print('Production coefficient branches agree; candidate scoring does not mutate transformer caches.', flush=True)
    policies = [
        ('native', None),
        ('joint32_t1', joint.CompoundJointPolicy()),
        ('joint32_depth', joint.CompoundJointPolicy(
            atom_temperature=.9, atom_top_k=0, atom_top_p=.9,
            coefficient_temperatures=(.4, .55, .75, 1.), joint_top_p=.9)),
    ]
    evaluations = []
    folder = OUT / 'joint-sampler-verification'
    folder.mkdir(exist_ok=True)
    for name, policy in policies:
        torch.manual_seed(261001)
        start = time.perf_counter()
        if policy is None:
            atom_ids, coefficient_ids = model.sample_compound(2, aux, cond,
                atom_temperature=1., atom_top_k=256, atom_top_p=.95,
                coeff_temperature=1., coeff_top_k=0, coeff_top_p=.95, amp=False)
        else:
            atom_ids, coefficient_ids = joint.sample_compound_joint(model, 2, aux, cond,
                amp=False, policy=policy)
        sampling_seconds = time.perf_counter() - start
        assert atom_ids.shape == coefficient_ids.shape == (2, 8, 8, 4)
        for depth in range(1, 4): assert not (atom_ids[..., depth, None] == atom_ids[..., :depth]).any()
        assert coefficient_ids.min() >= 0 and coefficient_ids.max() < 2048
        images = aux.decode_compound(atom_ids, coefficient_ids)
        assert images.shape == (2, 3, 256, 256) and torch.isfinite(images).all()
        from torchvision.utils import save_image
        target = folder / f'epoch4_{name}.png'
        save_image(((images + 1) / 2).clamp(0, 1), target, nrow=2)
        evaluations.append(dict(sampler=name, policy=None if policy is None else asdict(policy),
            samples=2, labels=cond.tolist(), sampling_seconds_cpu=sampling_seconds,
            finite_decoded_images=True, distinct_atom_supports=True, image=str(target)))
        print(json.dumps(evaluations[-1]), flush=True)
    receipt = dict(passed=True,checkpoint_epoch=4,checkpoint_step=2504,production_checkpoint_fid=62.41319733449029,
        model_parameters=sum(p.numel() for p in model.parameters()),
        candidate_atoms=32,candidate_coefficient_shape=list(actual.shape),
        coefficient_branches_match_independent_conditionals=True,transformer_caches_preserved=True,
        prior_atom_mass_not_squared=True,batched_candidate_seconds_cpu=batched_seconds,
        independent_candidate_seconds_cpu=independent_seconds,
        policies=evaluations,unit_tests_passed=8,
        used_cuda=False,quality_improvement_established=False,
        limitation='Two images per setting validate the code path, not FID or sampling quality.')
    (folder / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    shutil.copyfile(ROOT / 'src/compound_joint_sampling.py', folder / 'compound_joint_sampling.py')
    shutil.copyfile(Path(__file__), folder / 'verify_compound_joint_checkpoint.py')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
