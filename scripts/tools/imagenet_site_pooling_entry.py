"""Isolated entry for matched completed-site pooling experiments."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time

BASE = Path(os.environ['LASER_SITE_POOLING_BASE'])
OUT = Path(os.environ['LASER_SITE_POOLING_OUTPUT'])
BRANCH = os.environ['LASER_BRANCH']
PHASE = os.environ['LASER_PHASE']
PLAN = json.loads((OUT / 'plan.json').read_text())
ANCHOR_STEP = int(os.environ.get('LASER_RESUME_STEP', PLAN['anchor_step']))
ROOT = BASE / 'source'
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from src.training import rqtransformer as training
from src.training import k4_checkpoint_io as checkpoint_io
from src.training.coefficient_cross_attention import extend_optimizer_for_appended_parameters
from src.training.site_pair_pooling import attach_site_pair_pooling
from src.models.rqtransformer.attentions import AttentionBlock
from src.training.fid_reference import fixed_evaluation_rng


def record(name, value):
    folder = OUT / 'verification' / BRANCH / PHASE
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(target)


native_load, native_build = torch.load, training.build_model
old_names = new_names = new_state = None
production_metrics = None


def build(*args, **kwargs):
    global old_names, new_names, new_state
    model = native_build(*args, **kwargs)
    old_names = list(dict(model.named_parameters()))
    if BRANCH != 'sum':
        attach_site_pair_pooling(model, width=PLAN['width'], heads=PLAN['heads'], mode=BRANCH)
        initial = native_load(BASE / f'initial-{BRANCH}.pt', map_location='cpu', weights_only=True)
        model.site_pooling.load_state_dict(initial, strict=True)
        if PHASE.endswith('sum-only'):
            model.site_pooling.enabled = False
        new_state = {'site_pooling.' + k:v for k,v in model.site_pooling.state_dict().items()}
    new_names = list(dict(model.named_parameters()))
    assert new_names[:len(old_names)] == old_names
    return model


def load(path, *args, **kwargs):
    global production_metrics
    payload = native_load(path, *args, **kwargs)
    if isinstance(path, (str, Path)) and Path(path).resolve() == (BASE / 'anchor.pt').resolve():
        assert payload['global_step'] == ANCHOR_STEP
        assert payload['checkpoint_world_size'] == len(payload['rng_state_by_rank']) == 8
        if BRANCH != 'sum':
            assert not set(new_state).intersection(payload['state_dict'])
            payload = dict(payload, best_fid=[], best_inception=[],
                state_dict={**payload['state_dict'], **new_state},
                optimizer=extend_optimizer_for_appended_parameters(payload['optimizer'], old_names, new_names))
        record('anchor-migration-rank'+os.environ['RANK']+'.json',dict(
            passed=True, global_step=payload['global_step'], old_parameter_tensors=len(old_names),
            appended_parameter_tensors=len(new_names)-len(old_names),
            optimizer_states=len(payload['optimizer']['state']),
            rng_ranks=len(payload['rng_state_by_rank']), scheduler_age=payload['scheduler']['last_epoch'],
            old_model_and_optimizer_tensor_objects_preserved=True))
    if (PHASE == 'production' and isinstance(payload, dict)
            and payload.get('global_step') == ANCHOR_STEP
            and payload.get('fid') is not None
            and len(payload.get('optimizer', {}).get('state', {})) == 798):
        production_metrics = {k:payload[k] for k in ('global_step','epoch','fid','inception_score','inception_score_std')}
    return payload


training.build_model = build
torch.load = load
native_parser = training.build_parser


def parser():
    result = native_parser()
    native_parse = result.parse_args
    def parse(*args, **kwargs):
        value = native_parse(*args, **kwargs)
        if value.resume_checkpoint is not None:
            persistent = value.resume_checkpoint.resolve()
            local = checkpoint_io._checkpoint_upload_source(persistent)
            assert local != persistent or str(persistent).startswith('/tmp/'), 'Resume requires local serialization'
            value.resume_checkpoint = local
        value.site_pooling_mode = BRANCH
        value.site_pooling_width = PLAN['width'] if BRANCH != 'sum' else 0
        value.site_pooling_heads = PLAN['heads'] if BRANCH != 'sum' else 0
        value.site_pooling_ablation = 'sum-only' if PHASE.endswith('sum-only') else None
        value.site_pooling_anchor_step = PLAN['anchor_step']
        return value
    result.parse_args = parse
    return result


training.build_parser = parser


def wrap(model, backend, device, world):
    assert world == 8 and backend == 'ddp'
    assert model.config.vocab_size_cond == 1000 and model.pair_autoregressive
    assert list(model.block_size) == [8, 8, 4]
    assert len(model.body_transformer.blocks) == 42 and len(model.head_transformer.blocks) == 6
    record('architecture-rank'+os.environ['RANK']+'.json',dict(
        pid=os.getpid(), branch=BRANCH, phase=PHASE, world_size=world,
        full_pair_autoregressive=True, class_conditional=True, classes=1000,
        parameters=sum(p.numel() for p in model.parameters()),
        added_parameters=sum(p.numel() for p in model.site_pooling.parameters()) if BRANCH != 'sum' else 0,
        sum_only_ablation=PHASE.endswith('sum-only'), device=torch.cuda.get_device_name(device)))
    for block in model.head_transformer.blocks:
        block.attn.short_attention_backend = 'compiled'
    for block in model.modules():
        if isinstance(block, AttentionBlock):
            eager = block.forward
            compiled = torch.compile(eager, fullgraph=True, dynamic=True)
            def forward(x, module=block, eager=eager, compiled=compiled):
                return compiled(x) if module.training and torch.is_grad_enabled() else eager(x)
            block.forward = forward
    if BRANCH != 'sum':
        pooling = model.site_pooling
        eager = pooling.forward
        compiled = torch.compile(eager, fullgraph=True, dynamic=True)
        def forward(x):
            return (compiled if pooling.training and torch.is_grad_enabled() else eager)(x)
        pooling.forward = forward
    return DDP(model, device_ids=[device.index], broadcast_buffers=False,
               gradient_as_bucket_view=True, bucket_cap_mb=100)


training.wrap_distributed_model = wrap
training.compound_objective = torch.compile(training.compound_objective, fullgraph=True, dynamic=True)
training.atomic_torch_save = checkpoint_io.atomic_torch_save
training.upload_selected_checkpoint_files = checkpoint_io.upload_selected_checkpoint_files


def snapshot_checkpoint(source, destination):
    local = checkpoint_io._checkpoint_upload_source(Path(source).resolve())
    assert local != Path(source).resolve(), 'Snapshot requires local serialization'
    staging = Path(os.environ['LASER_CHECKPOINT_STAGING_DIR'])
    staging.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='best-', suffix='.pt', dir=staging)
    os.close(descriptor)
    temporary = Path(name);temporary.unlink();os.link(local, temporary)
    checkpoint_io._persist_serialized_checkpoint(temporary, Path(destination))
    # Retain native top-k files plus the trial's common snapshot and its old
    # best-file aliases. Prune orphaned payloads only in this owned directory.
    folder = Path(destination).parent / '.checkpoint-data'
    protected = [OUT / 'anchor.pt', *folder.parent.glob('*.pt')]
    for phase in ('train', *(f'seed-{s}' for s in PLAN['fid_seeds'])):
        protected.extend((OUT/'sum'/phase/'checkpoints').glob('*.pt'))
    retained = {path.resolve() for path in protected if path.is_symlink()}
    for path in folder.glob('*.pt'):
        if path not in retained:
            cached = checkpoint_io._local_checkpoint_paths(path)
            path.unlink(missing_ok=True)
            for item in cached or ():
                item.unlink(missing_ok=True)


training.snapshot_checkpoint = snapshot_checkpoint
native_evaluate = training.evaluate_generation_metrics


def evaluate(model, *args, **kwargs):
    seed = kwargs.get('fid_seed')
    if seed is None:
        return native_evaluate(model, *args, **kwargs)
    device = next(model.parameters()).device
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(device).clone()
    with fixed_evaluation_rng(seed, device, rank=int(os.environ['RANK'])):
        protocol = dict(configured_seed=seed, effective_rank_seed=seed+int(os.environ['RANK']),
            cuda_rng_sha256=hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest(),
            explicit_seed_applied=True, class_schedule='global_sample_index % 1000')
        record('fid-sampling-rank'+os.environ['RANK']+'.json',protocol)
        result = native_evaluate(model, *args, **kwargs)
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_cuda, torch.cuda.get_rng_state(device))
    record('fid-sampling-rank'+os.environ['RANK']+'.json',dict(protocol,completed=True,training_rng_restored=True))
    return result


training.evaluate_generation_metrics = evaluate
import wandb
native_wandb_init = wandb.init


def wandb_init(*args, **kwargs):
    kwargs['resume'] = 'must' if PHASE == 'production' else 'allow'
    kwargs['config']['architecture'] = 'compound-pair-rqtransformer-imagenet-1400m' + (
        '' if BRANCH == 'sum' else '-site-pooling-'+BRANCH)
    wb = native_wandb_init(*args, **kwargs)
    wb.summary.update({'site_pooling_trial/branch':BRANCH,'site_pooling_trial/anchor_step':PLAN['anchor_step'],
        'site_pooling_trial/updates_per_arm':PLAN['updates'],'site_pooling_trial/output':str(OUT)})
    if production_metrics:
        wb.log({'train/global_step':production_metrics['global_step'],'train/epoch':production_metrics['epoch'],
            'val/fid':production_metrics['fid'],'val/inception_score':production_metrics['inception_score'],
            'val/inception_score_std':production_metrics['inception_score_std']})
    record('wandb.json',dict(url=wb.url,id=wb.id,online=wb.settings.mode=='online'))
    return wb


wandb.init = wandb_init
native_step = torch.optim.AdamW.step
steps = 0


def step(optimizer, *args, **kwargs):
    global steps
    params = optimizer.param_groups[0]['params']
    old, new = params[:len(old_names)], params[len(old_names):]
    if steps == 0:
        assert len(old) == len(optimizer.state) == 798
        assert all(int(optimizer.state[p]['step']) == ANCHOR_STEP for p in old)
        assert all(p not in optimizer.state for p in new)
        record('startup-rank'+os.environ['RANK']+'.json',dict(
            initial_global_step=ANCHOR_STEP,preserved_adam_states=len(old),
            appended_uninitialized_states=len(new),learning_rate=optimizer.param_groups[0]['lr']))
    if new and steps < 2:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new)
    result = native_step(optimizer, *args, **kwargs)
    steps += 1
    if steps == 1:
        assert all(int(optimizer.state[p]['step']) == ANCHOR_STEP+1 for p in old)
        assert all(int(optimizer.state[p]['step']) == 1 for p in new)
    if steps in (1,2) or steps % 10 == 0:
        record('progress-rank'+os.environ['RANK']+'.json',dict(
            global_step=ANCHOR_STEP+steps,additional_updates=steps,pid=os.getpid(),time=time.time(),
            old_adam_age=int(optimizer.state[old[0]]['step']),
            new_adam_age=int(optimizer.state[new[0]]['step']) if new else None,
            learning_rate=optimizer.param_groups[0]['lr'],
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    return result


torch.optim.AdamW.step = step
if __name__ == '__main__':
    sys.argv = [str(ROOT/'train.py'),*sys.argv[1:]]
    runpy.run_path(str(ROOT/'train.py'),run_name='__main__')
