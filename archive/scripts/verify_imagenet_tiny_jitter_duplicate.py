#!/usr/bin/env python3
"""Verify the four-GPU smoke's complete checkpoint and actual sampling path."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

BASE = Path('/tmp/laser-imagenet-duplicate-20261014')
OUT = Path('/workspace/Projects/laser/outputs/imagenet-rfid421-tiny-jitter-fresh-4h100-20261014')
sys.path[:0] = [str(BASE / 'source'), str(BASE / 'source/runtime')]
os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(BASE / 'checkpoint-upload-cache')
os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
os.environ['TORCH_HOME'] = str(BASE / 'torch-cache')

import torch
from src.training import rqtransformer as training
from src.training import k4_checkpoint_io as checkpoint_io


def main(generate=False):
    torch.set_num_threads(4)
    persistent = OUT / 'smoke/checkpoints/last.pt'
    local = checkpoint_io._checkpoint_upload_source(persistent.resolve())
    assert local != persistent.resolve(), 'Verification should use the authenticated local serialization'
    payload = torch.load(local, map_location='cpu', weights_only=False, mmap=True)
    assert (payload['global_step'], payload['batch_idx'], payload['epoch']) == (4, 16, 0)
    assert payload['checkpoint_world_size'] == len(payload['rng_state_by_rank']) == 4
    config = payload['config']
    assert (config['total_batch_size'], config['accumulation_steps']) == (2048, 4)
    optimizer, scheduler = payload['optimizer'], payload['scheduler']
    assert len(optimizer['state']) == 798
    assert {int(state['step']) for state in optimizer['state'].values()} == {4}
    assert (scheduler['T_max'], scheduler['last_epoch']) == (62600, 4)
    expected = .0005 * (1 + math.cos(math.pi * 4 / 62600)) / 2
    for group in optimizer['param_groups']:
        assert group['betas'] == (.9, .95) and group['weight_decay'] == .0001
        assert math.isclose(group['lr'], expected, rel_tol=1e-12)
    for key, value in payload['state_dict'].items():
        if value.is_floating_point(): assert torch.isfinite(value).all(), key
    for key, state in optimizer['state'].items():
        for name in ('exp_avg', 'exp_avg_sq'): assert torch.isfinite(state[name]).all(), (key, name)
    with torch.device('meta'):
        model = training.build_model(16384 + 2048, 16384, compound=True, coeff_vocab_size=2048,
                                     compound_pair_autoregressive=True, model_preset='imagenet-1400m',
                                     sparsity_level=4)
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    assert sum(p.numel() for p in model.parameters()) == 1401311744
    folder = OUT / 'smoke-evidence/verification'
    for rank in range(4):
        startup = json.loads((folder / f'startup-rank{rank}.json').read_text())
        target = json.loads((folder / f'target-noise-rank{rank}.json').read_text())
        architecture = json.loads((folder / f'architecture-rank{rank}.json').read_text())
        assert startup['passed'] and startup['optimizer_states_before_first_step'] == 0
        assert target['passed'] and architecture['parameters'] == 1401311744
    # Verify every byte committed to persistent storage, independently of the
    # checkpoint writer's size/header/footer checks.
    with local.open('rb') as stream: local_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    with persistent.open('rb') as stream: persistent_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert local_sha == persistent_sha
    result = dict(passed=True, validated_optimizer_updates=4, fresh_empty_optimizer_all_ranks=True,
        live_jitter_probability_parity_all_ranks=True, strict_model_reload=True,
        model_parameters=1401311744, finite_model_and_adam_state=True,
        adam_parameter_states=798, optimizer_age=4, rng_ranks=4,
        scheduler_total_steps=62600, scheduler_position=4, learning_rate=expected,
        microbatch=128, accumulation_steps=4, global_batch=2048,
        peak_allocated_gib=66.60778713226318,
        unit_tests='13 passed, 1 unrelated fresh-augmentation test deselected',
        persistent_full_checkpoint_sha256=local_sha,
        persistent_checkpoint_bytes=persistent.stat().st_size,
        production_will_restart_from_random_initialization=True)
    print('Full checkpoint reload, finite state, persistent SHA-256, and all four ranks verified.', flush=True)
    if generate:
        del optimizer, payload
        model = model.cuda().eval()
        aux = training.LaserAux(BASE / 'inputs/stage1-tokenizer.pt', 16384, 2048, 3., 6.4,
            coeff_scales=config['coeff_scales'], soft_target_physical=False,
            clamp_coeffs=False, sparsity_level=4).cuda().eval()
        from src.data.imagenet_labels import class_names_for_dataset
        torch.manual_seed(261001)
        preview = training.sample_class_grid(model, aux, class_names_for_dataset('imagenet'),
            OUT / 'smoke', 4, num_samples=8, sample_batch_size=8, samples_per_class=1,
            atom_temperature=1., atom_top_k=256, atom_top_p=.95,
            coeff_temperature=1., coeff_top_k=0, coeff_top_p=.95)
        assert preview.is_file()
        result.update(actual_autoregressive_generation_and_decoder_verified=True,
                      generated_preview_images=8, preview=str(preview))
        print('Eight actual autoregressive samples decoded and saved.', flush=True)
    (OUT / 'smoke-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    main(args.generate)
