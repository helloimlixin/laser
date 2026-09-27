"""Resume and artifact helpers for the released-tokenizer Church control."""
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch


def capture_rng_state(device):
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng_state(state, device):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'], device)
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def restore_training_state(payload, model, optimizer, scheduler, scaler):
    model.load_state_dict(payload['state_dict'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(payload['scaler'])
    assert len(optimizer.state) == len(payload['optimizer']['state']) > 0
    assert scheduler.after_scheduler.last_epoch == payload['step']
    for actual, saved in zip(optimizer.param_groups, payload['optimizer']['param_groups']):
        assert actual['lr'] == saved['lr']


def resumed_iterator(loader, sampler, epoch, consumed, *, rng_state, device):
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(consumed):
        next(iterator)
    if rng_state is not None:
        restore_rng_state(rng_state, device)
    return iterator


def validate_control_resume(payload, *, config, cache, protocol, run_id, loader_batches):
    assert payload['run_id'] == run_id
    assert payload['tokenizer']['checkpoint_sha256'] == cache['checkpoint_sha256']
    assert payload['tokenizer']['frozen_state_sha256'] == cache['frozen_state_sha256']
    assert payload['tokenizer']['config_sha256'] == cache['config_sha256']
    for key in ('arch', 'loss', 'optimizer', 'sampling'):
        assert payload['config'][key] == config[key], key
    assert payload['config']['experiment']['total_batch_size'] == config['experiment']['total_batch_size'] == 2048
    assert payload['config']['experiment']['epochs'] == config['experiment']['epochs'] == 300
    old_protocol = payload['fid_protocol']
    for key in ('generated_samples', 'real_samples', 'sampling', 'seed', 'world_size',
                'generation_batch_per_gpu', 'decode_and_inception_batch_size'):
        assert old_protocol[key] == protocol[key], key
    # A rebuilt cache/reference is explicitly recorded, never claimed byte-identical.
    if old_protocol['reference_sha256'] != protocol['reference_sha256']:
        assert cache['amarel_rebuild']['source_reference_sha256'] == old_protocol['reference_sha256']
    old_world = len(payload['rng_states'])
    world = config['training_world_size']
    same_batches = old_world == world and payload['config']['experiment']['batch_size'] == config['experiment']['batch_size']
    if not same_batches:
        assert payload['batch_in_epoch'] == 0, 'GPU/batch migration requires an epoch-boundary checkpoint'
    assert 0 <= payload['batch_in_epoch'] <= loader_batches
    return payload['epoch'], payload['batch_in_epoch']


def ranked_candidates(ranked, *, fid, epoch, step):
    assert math.isfinite(fid)
    rows = [copy.deepcopy(r) for r in ranked if r['step'] != step]
    rows.append(dict(fid=fid, epoch=epoch, step=step, path=f'fid-epoch{epoch:03d}-step{step:07d}.pt'))
    return sorted(rows, key=lambda r: (r['fid'], r['step']))[:3]


def upload_checkpoints(run, out, ranked, *, epoch, step, protocol):
    import json
    import wandb
    out = Path(out)
    metadata = dict(epoch=epoch, optimizer_step=step, run_id=run.id, fid_protocol=protocol,
                    ranked_fid_checkpoints=ranked,
                    checkpoint_policy='latest full state plus best three FID states')
    selection = out / 'selection.json'
    selection.write_text(json.dumps(metadata, indent=2) + '\n')
    artifact = wandb.Artifact(f'model-{run.id}-selected-checkpoints', type='model', metadata=metadata)
    artifact.add_file(str(out / 'last.pt'), name='last.pt', policy='immutable', skip_cache=True)
    artifact.add_file(str(selection), name='selection.json', policy='immutable', skip_cache=True)
    for index, row in enumerate(ranked, 1):
        artifact.add_file(str(out / row['path']), name=f'best-fid-{index:02d}.pt',
                          policy='immutable', skip_cache=True)
    logged = run.log_artifact(artifact, aliases=['latest', 'last', f'step-{step}'])
    logged.wait()
    run.log({'checkpoint_upload/last_step': step, 'checkpoint_upload/best_fid_count': len(ranked),
             'checkpoint_upload/artifact': logged.qualified_name})
