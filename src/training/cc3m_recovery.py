"""Explicit best-model restarts with a new optimizer and a bounded schedule."""
import hashlib
from pathlib import Path

import torch


def schedule_offset(cfg, updates):
    warm = cfg.get('warm_start', {})
    epoch, step = int(warm.get('epoch', 0)), int(warm.get('global_step', 0))
    if step != epoch*updates or not 0 <= epoch < cfg['epochs']:
        raise ValueError('Warm-start cursor must be a completed epoch within the training budget')
    return step


def load_best_weights(model, cfg, updates):
    warm = cfg['warm_start']
    if warm.get('optimizer_policy') != 'reset':
        raise ValueError('A weights restart requires an explicitly reset optimizer')
    path = Path(warm['checkpoint'])
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != warm['checkpoint_sha256']:
        raise ValueError('Best checkpoint SHA256 mismatch')
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    for key in ('arch', 'stage1_sha256', 'codebook_sha256', 'reconstruction_codebook_sha256',
                'tokenizer_backend', 'temperatures', 'global_batch', 'sampling', 'evaluation'):
        if saved['config'][key] != cfg[key]:
            raise ValueError(f'Best checkpoint changes a fixed training/evaluation setting: {key}')
    if saved['global_step'] != schedule_offset(cfg, updates) or saved['epoch'] != warm['epoch']:
        raise ValueError('Best checkpoint cursor differs from the requested source')
    model.load_state_dict(saved['model'], strict=True)
    return dict(epoch=saved['epoch'], global_step=saved['global_step'],
                best={key: float(saved['metrics'][key]) for key in ('fid', 'clip_score')})


def verify_baseline(results, cfg):
    warm = cfg.get('warm_start')
    if not warm:
        return
    if abs(results['fid']-warm['expected_fid']) > warm.get('baseline_fid_tolerance', .01):
        raise ValueError('Best-checkpoint FID did not reproduce; refusing to train')
