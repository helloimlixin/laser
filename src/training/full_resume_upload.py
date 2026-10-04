"""Retain full optimizer-boundary checkpoints for independent metric winners.

This watcher operates on committed immutable checkpoints, so it can be attached
to a running job without interrupting training or uploading model-only winners
as recovery checkpoints. The trainer continues to upload its regular last.pt.
"""
import hashlib
import base64
import json
import math
import os
from pathlib import Path

import torch


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def recovery_metadata(payload):
    """Reject weights-only or incomplete checkpoints before cloud publication."""
    required = {'state_dict', 'optimizer', 'scheduler', 'config', 'epoch',
                'global_step', 'rng_state_by_rank', 'checkpoint_world_size'}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f'Incomplete recovery checkpoint: missing {sorted(missing)}')
    config = payload['config']
    world = int(payload['checkpoint_world_size'])
    states = payload['rng_state_by_rank']
    if world < 1 or len(states) != world:
        raise ValueError('Checkpoint must include RNG state for every training rank')
    if any(not {'torch_cpu', 'torch_cuda'} <= state.keys() for state in states):
        raise ValueError('Checkpoint is missing a CPU or CUDA RNG stream')
    optimizer = payload['optimizer']
    groups = optimizer['param_groups']
    parameters = [parameter for group in groups for parameter in group['params']]
    if not parameters or set(parameters) != set(optimizer['state']):
        raise ValueError('Incomplete Adam parameter state')
    steps = set()
    for state in optimizer['state'].values():
        if not {'step', 'exp_avg', 'exp_avg_sq'} <= state.keys():
            raise ValueError('Missing Adam counter or moments')
        steps.add(int(state['step']))
    if len(steps) != 1:
        raise ValueError('Adam counters disagree across parameters')
    schedule = config.get('lr_schedule')
    if schedule not in {'constant', 'cosine'}:
        raise ValueError('Unspecified learning-rate schedule')
    if schedule != 'constant' and payload['scheduler'] is None:
        raise ValueError('Nonconstant schedule requires scheduler state')
    accumulation = int(config['accumulation_steps'])
    batch = int(payload.get('batch_idx', 0))
    if batch < 0 or batch % accumulation:
        raise ValueError('Recovery must occur at an optimizer-step boundary')
    for key in ('epoch', 'global_step'):
        if int(payload[key]) < 0:
            raise ValueError(f'Invalid recovery cursor: {key}')
    original = payload.get('original_rqtransformer_metrics')
    # A previous evaluation copied into a mid-epoch save is not a scored model.
    if original is not None and int(original['global_step']) != int(payload['global_step']):
        raise ValueError('Metric record does not belong to the saved model')
    return dict(
        schema='laser-full-recovery-v1', epoch=int(payload['epoch']),
        next_microbatch=batch, global_step=int(payload['global_step']),
        world_size=world, optimizer='AdamW', adam_parameters=len(parameters),
        adam_step=steps.pop(), saved_learning_rates=[float(g['lr']) for g in groups],
        learning_rate_schedule=dict(kind=schedule, state=payload['scheduler'],
                                    config={k: config.get(k) for k in
                                            ('lr', 'min_lr', 'warmup_epochs',
                                             'lr_schedule_epochs', 'epochs')}),
        gradient_accumulation_steps=accumulation,
        global_batch_size=int(config['total_batch_size']),
        sampler=dict(seed=int(config['seed']), epoch=int(payload['epoch']),
                     next_microbatch=batch, training_images=int(config['training_images'])),
        rng_streams=['torch_cpu', 'torch_cuda'], rng_ranks=len(states),
        precision='bfloat16', gradient_scaler=dict(enabled=False, state=None),
        configuration=config, original_rqtransformer_metrics=original,
        fid=payload.get('fid'), inception_score=payload.get('inception_score'))


def metric_scores(metadata):
    original = metadata['original_rqtransformer_metrics']
    source = original if original is not None else metadata
    result = {}
    for kind, field in (('fid', 'fid'), ('is', 'inception_score')):
        score = source.get(field)
        if score is not None and math.isfinite(float(score)):
            result[kind] = float(score)
    return result


def cached_payload(receipt_path, cache_dir):
    """Use only a committed cache object whose immutable receipt still matches."""
    receipt = json.loads(Path(receipt_path).read_text())
    source = Path(receipt.get('payload', receipt.get('target')))
    key = hashlib.sha256(str(source).encode()).hexdigest()
    cached = Path(cache_dir) / 'objects' / (key + '.pt')
    metadata = json.loads(cached.with_suffix('.json').read_text())
    if metadata['source'] != str(source) or cached.stat().st_size != int(receipt['bytes']):
        raise ValueError('Immutable checkpoint cache identity mismatch')
    return source, cached


class FullResumeWinners:
    """Pin each complete winner; every upload snapshot contains both winners."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / 'resume-winners.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def consider(self, source, *, provenance=None):
        source = Path(source)
        # Pin before opening the mmap, since the trainer may unlink its cache.
        pin = self.directory / '.candidate.pt'
        pin.unlink(missing_ok=True)
        os.link(source, pin)
        try:
            payload = torch.load(pin, map_location='cpu', weights_only=False, mmap=True)
            metadata = recovery_metadata(payload)
            del payload
            changed = []
            digest = None
            for kind, score in metric_scores(metadata).items():
                previous = self.state.get(kind)
                if previous is not None and not (score < previous['score'] if kind == 'fid'
                                                 else score > previous['score']):
                    continue
                target = self.directory / f'best-{kind}-resume.pt'
                temporary = target.with_suffix('.new')
                temporary.unlink(missing_ok=True)
                os.link(pin, temporary)
                temporary.replace(target)
                if digest is None:
                    with pin.open('rb') as stream:
                        digest = base64.b64encode(hashlib.file_digest(stream, 'md5').digest()).decode()
                record = dict(metadata, score=score, file=target.name,
                              metric_backend='original_rqtransformer' if metadata[
                                  'original_rqtransformer_metrics'] is not None else 'torchmetrics',
                              source=str(provenance or source), bytes=target.stat().st_size,
                              md5=digest)
                self.state[kind] = record
                atomic_json(target.with_suffix('.json'), record)
                changed.append(kind)
            if changed:
                atomic_json(self.state_path, self.state)
            return changed
        finally:
            pin.unlink(missing_ok=True)

    def upload_paths(self):
        return [self.directory / name for name in sorted(
            ['resume-winners.json'] + [name for kind in self.state for name in
             (f'best-{kind}-resume.pt', f'best-{kind}-resume.json')])]

    def persist(self, destination):
        """Persist winner bytes before publishing the receipt on shared storage."""
        from .k4_checkpoint_io import _persist_serialized_checkpoint
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for kind, metadata in self.state.items():
            target = destination / metadata['file']
            receipt = target.with_suffix('.json')
            if receipt.exists() and target.is_file():
                previous = json.loads(receipt.read_text())
                if (previous.get('md5') == metadata['md5'] and
                        previous.get('bytes') == metadata['bytes'] and
                        target.stat().st_size == metadata['bytes']):
                    continue
            pin = self.directory / f'.persist-{kind}.pt'
            pin.unlink(missing_ok=True)
            os.link(self.directory / metadata['file'], pin)
            _persist_serialized_checkpoint(pin, target)
            atomic_json(receipt, metadata)
        atomic_json(destination / 'resume-winners.json', self.state)
