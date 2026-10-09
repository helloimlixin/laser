"""Frozen plain-baseline entry for a fresh ImageNet tiny-jitter run."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time

BASE = Path(os.environ['LASER_PAIR_MEMORY_BASE'])
OUT = Path(os.environ['LASER_PAIR_MEMORY_OUTPUT'])
ROOT = BASE/'source'
sys.path[:0] = [str(ROOT), str(ROOT/'runtime')]

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from src.training import rqtransformer as training
from src.training import k4_checkpoint_io as checkpoint_io
from src.training.coefficient_jitter import quantize_with_jitter, jitter_probabilities
from src.training.generation_logging import define_generation_metrics, generation_payload, log_generation_metrics
from src.models.rqtransformer.attentions import AttentionBlock

CALIBRATION = json.loads((OUT/'noise-calibration.json').read_text())
SIGMA = CALIBRATION['selected']['sigma_bins']
CAP = CALIBRATION['selected']['cap_bins']


def record(name, value):
    path = OUT/'verification'/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temp.replace(path)


native_parser = training.build_parser
def parser():
    result = native_parser()
    native_parse = result.parse_args
    def parse(*args, **kwargs):
        value = native_parse(*args, **kwargs)
        assert not value.resume and value.resume_checkpoint is None and value.init_stage2_checkpoint is None
        assert value.compound_tokens and value.compound_pair_autoregressive
        value.coefficient_target_distribution = 'bounded continuous jitter before nearest-bin quantization'
        value.coefficient_noise_sigma_bins = SIGMA
        value.coefficient_noise_cap_bins = CAP
        value.coefficient_noise_actual_rms_bins = CALIBRATION['selected']['actual_rms_bins']
        value.fresh_stage2_initialization = True
        return value
    result.parse_args = parse
    return result
training.build_parser = parser


probe_done = False
def tiny_coeff_ids(aux, coeffs, *, stochastic=True, temp=0.5, hard=False):
    global probe_done
    assert not aux.soft_target_physical
    ids, p = quantize_with_jitter(coeffs.float(), aux.coeff_bins,
        stochastic=stochastic, hard=hard, sigma_bins=SIGMA, cap_bins=CAP)
    if not probe_done:
        c = coeffs.reshape(-1, 4)[:256].float().cpu()
        bins = aux.coeff_bins.detach().float().cpu()
        actual = p.reshape(-1, 4, aux.coeff_vocab_size)[:256].float().cpu()
        nearest, expected = jitter_probabilities(c, bins, sigma_bins=SIGMA, cap_bins=CAP)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
        native_nearest = (c[..., None]-bins).abs().argmin(-1)
        assert torch.equal(nearest, native_nearest)
        flip = (1-actual.gather(-1, nearest[..., None]).squeeze(-1)).mean(0)
        record('target-noise-rank'+os.environ['RANK']+'.json', dict(passed=True,
            exact_cell_probability_parity=True, native_nearest_bin_parity=True,
            sigma_bins=SIGMA, hard_cap_bins=CAP, noise_applied_before_quantization=True,
            temperature_floor_bypassed=True, expected_flip_fraction_per_depth=flip.tolist()))
        probe_done = True
    return ids, p
training.LaserAux.compound_coeff_ids = tiny_coeff_ids


def wrap(model, backend, device, world):
    assert world == 8 and backend == 'ddp'
    assert model.config.vocab_size_cond == 1000 and model.pair_autoregressive
    assert list(model.block_size) == [8, 8, 4]
    assert len(model.body_transformer.blocks) == 42 and len(model.head_transformer.blocks) == 6
    assert not hasattr(model, 'pair_memory_queries')
    record('architecture-rank'+os.environ['RANK']+'.json', dict(passed=True,
        pid=os.getpid(), world_size=world, classes=1000, full_pair_autoregressive=True,
        coefficient_conditioned_on_corresponding_atom=True,
        parameters=sum(p.numel() for p in model.parameters()), added_parameters=0,
        device=torch.cuda.get_device_name(device)))
    for block in model.head_transformer.blocks:
        block.attn.short_attention_backend = 'compiled'
    for block in model.modules():
        if isinstance(block, AttentionBlock):
            eager = block.forward
            compiled = torch.compile(eager, fullgraph=True, dynamic=True)
            def forward(x, module=block, eager=eager, compiled=compiled):
                return compiled(x) if module.training and torch.is_grad_enabled() else eager(x)
            block.forward = forward
    return DDP(model, device_ids=[device.index], broadcast_buffers=False,
               gradient_as_bucket_view=True, bucket_cap_mb=100)
training.wrap_distributed_model = wrap
training.compound_objective = torch.compile(training.compound_objective, fullgraph=True, dynamic=True)
training.atomic_torch_save = checkpoint_io.atomic_torch_save
training.upload_selected_checkpoint_files = checkpoint_io.upload_selected_checkpoint_files


def snapshot_checkpoint(source, destination):
    local = checkpoint_io._checkpoint_upload_source(Path(source).resolve())
    assert local != Path(source).resolve(), 'Best snapshot requires local serialization'
    staging = Path(os.environ['LASER_CHECKPOINT_STAGING_DIR'])
    descriptor, name = tempfile.mkstemp(prefix='best-', suffix='.pt', dir=staging)
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    os.link(local, temporary)
    checkpoint_io._persist_serialized_checkpoint(temporary, Path(destination))
    folder = Path(destination).parent/'.checkpoint-data'
    retained = {p.resolve() for p in folder.parent.glob('*.pt') if p.is_symlink()}
    for path in folder.glob('*.pt'):
        if path not in retained:
            cached = checkpoint_io._local_checkpoint_paths(path)
            path.unlink(missing_ok=True)
            for item in cached or ():
                item.unlink(missing_ok=True)
training.snapshot_checkpoint = snapshot_checkpoint


class LoggedRun:
    def __init__(self, run):
        self.run = run
    def __getattr__(self, name):
        return getattr(self.run, name)
    def define_metric(self, name, **kwargs):
        if name in ('eval/fid', 'val/fid'):
            kwargs['summary'] = 'min,last'
        elif name in ('eval/inception_score', 'val/inception_score'):
            kwargs['summary'] = 'max,last'
        return self.run.define_metric(name, **kwargs)
    def log(self, data, *args, **kwargs):
        if 'val/fid' in data:
            assert not args and not kwargs
            payload = generation_payload(data['val/fid'], data.get('val/inception_score'),
                data.get('val/inception_score_std'), epoch=data['train/epoch'], step=data['train/global_step'])
            payload.update(data)
            log_generation_metrics(self.run, payload, OUT/'train')
            record('metric-logging.json', dict(passed=True, actual_generation_metrics=True,
                keys=list(payload), global_step=payload['train/global_step']))
        else:
            return self.run.log(data, *args, **kwargs)


import wandb
native_wandb_init = wandb.init
def wandb_init(*args, **kwargs):
    kwargs['resume'] = 'never'
    run = native_wandb_init(*args, **kwargs)
    define_generation_metrics(run)
    run.config.update({'fresh_stage2_initialization': True,
        'stage1_reconstruction_fid': 4.21, 'noise_calibration': CALIBRATION}, allow_val_change=True)
    run.summary.update({'noise/sigma_bins': SIGMA, 'noise/hard_cap_bins': CAP,
        'noise/actual_rms_bins': CALIBRATION['selected']['actual_rms_bins'],
        'noise/expected_bin_flip_fraction': CALIBRATION['independent_verification']['expected_bin_flip_fraction'],
        'evaluation/status': 'waiting_for_first_completed_evaluation',
        'evaluation/first_epoch': 1, 'evaluation/cadence_epochs': 2,
        'training/initialized_from_scratch': True})
    for name in ('noise-calibration.json', 'plan.json', 'source-manifest.json', 'preflight-verification.json'):
        run.save(str(OUT/name), base_path=str(OUT), policy='now')
    record('wandb.json', dict(url=run.url, id=run.id, online=run.settings.mode == 'online',
        metric_keys=['eval/fid', 'eval/inception_score', 'eval/inception_score_std'],
        real_metrics_pending_until_first_evaluation=True))
    return LoggedRun(run)
wandb.init = wandb_init


native_step = torch.optim.AdamW.step
steps = 0
def step(optimizer, *args, **kwargs):
    global steps
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    assert len(parameters) == 798
    if steps == 0:
        assert len(optimizer.state) == 0
        record('startup-rank'+os.environ['RANK']+'.json', dict(passed=True,
            initial_global_step=0, optimizer_states_before_first_step=0,
            fresh_stage2=True, learning_rate=optimizer.param_groups[0]['lr'], time=time.time()))
    result = native_step(optimizer, *args, **kwargs)
    steps += 1
    if steps == 1:
        assert len(optimizer.state) == 798
        assert {int(s['step']) for s in optimizer.state.values()} == {1}
    if steps in (1, 2) or steps % 10 == 0:
        record('progress-rank'+os.environ['RANK']+'.json', dict(passed=True,
            global_step=steps, adam_age=int(optimizer.state[parameters[0]]['step']),
            pid=os.getpid(), time=time.time(), learning_rate=optimizer.param_groups[0]['lr'],
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    return result
torch.optim.AdamW.step = step


if __name__ == '__main__':
    sys.argv = [str(ROOT/'train.py'), *sys.argv[1:]]
    runpy.run_path(str(ROOT/'train.py'), run_name='__main__')
