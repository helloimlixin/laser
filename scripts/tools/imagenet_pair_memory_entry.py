"""Frozen ImageNet pair-memory trial entry; invoked by its supervisor."""
from contextlib import nullcontext
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
BRANCH = os.environ['LASER_BRANCH']
PHASE = os.environ.get('LASER_PHASE', 'train')
PLAN = json.loads((OUT / 'plan.json').read_text())
ANCHOR_STEP = PLAN['anchor_step']
ROOT = BASE / 'source'
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from src.training import rqtransformer as training
from src.training import k4_checkpoint_io as checkpoint_io
from src.training.coefficient_cross_attention import extend_optimizer_for_appended_parameters
from src.training.pair_memory_cross_attention import attach_pair_memory_queries
from src.models.rqtransformer.attentions import AttentionBlock
from src.training.fid_reference import fixed_evaluation_rng


def record(name, value):
    path = OUT / 'verification' / BRANCH / PHASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


native_load = torch.load
native_build = training.build_model
old_names = new_names = new_state = None


def build(*args, **kwargs):
    global old_names, new_names, new_state
    model = native_build(*args, **kwargs)
    old_names = list(dict(model.named_parameters()))
    if BRANCH != 'baseline':
        attach_pair_memory_queries(model, width=512, heads=8, memory_layers=1,
                                   query_layers=2, mode=BRANCH)
        initial = native_load(BASE / f'initial-{BRANCH}.pt', map_location='cpu', weights_only=True)
        model.pair_memory_queries.load_state_dict(initial, strict=True)
        new_state = {'pair_memory_queries.' + key: value for key, value in model.pair_memory_queries.state_dict().items()}
    new_names = list(dict(model.named_parameters()))
    assert new_names[:len(old_names)] == old_names
    return model


def load(path, *args, **kwargs):
    payload = native_load(path, *args, **kwargs)
    if isinstance(path, (str, Path)) and Path(path).resolve() == (BASE / 'anchor.pt').resolve():
        assert payload['global_step'] == ANCHOR_STEP
        assert payload['checkpoint_world_size'] == len(payload['rng_state_by_rank']) == 8
        if BRANCH != 'baseline':
            assert not set(new_state).intersection(payload['state_dict'])
            payload = dict(payload, best_fid=[], best_inception=[],
                state_dict={**payload['state_dict'], **new_state},
                optimizer=extend_optimizer_for_appended_parameters(payload['optimizer'], old_names, new_names))
        record('anchor-migration-rank' + os.environ['RANK'] + '.json', dict(
            old_parameter_tensors=len(old_names), new_parameter_tensors=len(new_names)-len(old_names),
            old_optimizer_states=len(payload['optimizer']['state']), new_optimizer_states=0,
            common_weights_and_moments_unchanged=True, global_step=payload['global_step'],
            epoch=payload['epoch'], batch_idx=payload.get('batch_idx'),
            rng_ranks=len(payload['rng_state_by_rank']), scheduler=payload['scheduler']))
    return payload


training.build_model = build
torch.load = load
native_parser = training.build_parser


def parser():
    result = native_parser()
    native_parse = result.parse_args
    def parse(*args, **kwargs):
        value = native_parse(*args, **kwargs)
        if value.fid_only and value.resume_checkpoint is not None:
            persistent = value.resume_checkpoint.resolve()
            local = checkpoint_io._checkpoint_upload_source(persistent)
            assert local != persistent, 'Evaluation must use verified local serialization'
            value.resume_checkpoint = local
        value.pair_memory_query_mode = BRANCH
        value.pair_memory_width = 512 if BRANCH != 'baseline' else 0
        value.pair_memory_encoder_layers = 1 if BRANCH != 'baseline' else 0
        value.pair_memory_query_layers = 2 if BRANCH != 'baseline' else 0
        value.pair_memory_anchor_step = ANCHOR_STEP
        return value
    result.parse_args = parse
    return result


training.build_parser = parser


def wrap(model, backend, device, world):
    assert world == 8 and backend == 'ddp'
    assert model.config.vocab_size_cond == 1000 and model.pair_autoregressive
    assert list(model.block_size) == [8, 8, 4]
    assert len(model.body_transformer.blocks) == 42 and len(model.head_transformer.blocks) == 6
    record('architecture-rank' + os.environ['RANK'] + '.json', dict(
        pid=os.getpid(), branch=BRANCH, phase=PHASE, world_size=world,
        class_conditional=True, classes=1000, full_pair_autoregressive=True,
        atom_query=BRANCH != 'baseline', coefficient_atom_query=BRANCH != 'baseline',
        parameters=sum(p.numel() for p in model.parameters()),
        added_parameters=sum(p.numel() for p in model.pair_memory_queries.parameters()) if BRANCH != 'baseline' else 0,
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
    if BRANCH != 'baseline':
        decoder = model.pair_memory_queries
        eager = decoder.forward
        compiled = torch.compile(eager, fullgraph=True, dynamic=True)
        def forward(*args):
            return (compiled if decoder.training and torch.is_grad_enabled() else eager)(*args)
        decoder.forward = forward
    return DDP(model, device_ids=[device.index], broadcast_buffers=False,
               gradient_as_bucket_view=True, bucket_cap_mb=100)


training.wrap_distributed_model = wrap
training.compound_objective = torch.compile(training.compound_objective, fullgraph=True, dynamic=True)
training.atomic_torch_save = checkpoint_io.atomic_torch_save
training.upload_selected_checkpoint_files = checkpoint_io.upload_selected_checkpoint_files


def snapshot_checkpoint(source, destination):
    """Persist best snapshots from the retained local inode, avoiding remote copies."""
    local = checkpoint_io._checkpoint_upload_source(Path(source).resolve())
    assert local != Path(source).resolve(), 'Best snapshot requires local serialization'
    staging = Path(os.environ['LASER_CHECKPOINT_STAGING_DIR'])
    descriptor, name = tempfile.mkstemp(prefix='best-', suffix='.pt', dir=staging)
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    os.link(local, temporary)
    checkpoint_io._persist_serialized_checkpoint(temporary, Path(destination))
    # Native top-k pruning unlinks the checkpoint name. Remove only orphaned
    # immutable payloads inside this owned checkpoint directory afterwards.
    folder = Path(destination).parent / '.checkpoint-data'
    retained = {path.resolve() for path in folder.parent.glob('*.pt') if path.is_symlink()}
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
    rank = int(os.environ['RANK'])
    before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    with fixed_evaluation_rng(seed, device, rank=rank):
        protocol = dict(configured_seed=seed, effective_rank_seed=int(seed)+rank,
            cuda_rng_sha256=hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest(),
            explicit_seed_applied=True, class_schedule='global_sample_index % 1000')
        record('fid-sampling-rank' + os.environ['RANK'] + '.json', protocol)
        result = native_evaluate(model, *args, **kwargs)
    assert torch.equal(before_cpu, torch.get_rng_state()) and torch.equal(before_cuda, torch.cuda.get_rng_state(device))
    record('fid-sampling-rank' + os.environ['RANK'] + '.json', dict(protocol, completed=True, training_rng_restored=True))
    return result


training.evaluate_generation_metrics = evaluate
import wandb
native_wandb_init = wandb.init


def wandb_init(*args, **kwargs):
    kwargs['resume'] = 'must' if BRANCH == 'baseline' else 'allow'
    if BRANCH != 'baseline':
        kwargs['config']['architecture'] = f'compound-pair-rqtransformer-imagenet-1400m-pair-memory-{BRANCH}'
    wb = native_wandb_init(*args, **kwargs)
    if BRANCH != 'baseline':
        wb.summary.update({'pair_memory_trial/branch': BRANCH,
            'pair_memory_trial/anchor_step': ANCHOR_STEP,
            'pair_memory_trial/updates_per_arm': PLAN['updates'],
            'pair_memory_trial/fid_samples': PLAN['fid_samples'],
            'pair_memory_trial/tests_passed': PLAN['tests_passed']})
    record('wandb.json', dict(url=wb.url, id=wb.id, online=wb.settings.mode == 'online'))
    return wb


wandb.init = wandb_init
native_step = torch.optim.AdamW.step
steps = 0


def step(optimizer, *args, **kwargs):
    global steps
    parameters = optimizer.param_groups[0]['params']
    old, new = parameters[:len(old_names)], parameters[len(old_names):]
    if steps == 0:
        assert len(optimizer.state) == len(old) == 798
        assert all(int(optimizer.state[p]['step']) == ANCHOR_STEP for p in old)
        assert all(p not in optimizer.state for p in new)
        record('startup-rank' + os.environ['RANK'] + '.json', dict(
            initial_global_step=ANCHOR_STEP, preserved_adam_states=len(old),
            appended_uninitialized_states=len(new), learning_rate=optimizer.param_groups[0]['lr'], time=time.time()))
    if new and steps < 2:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new)
    result = native_step(optimizer, *args, **kwargs)
    steps += 1
    if steps == 1:
        assert all(int(optimizer.state[p]['step']) == ANCHOR_STEP+1 for p in old)
        assert all(int(optimizer.state[p]['step']) == 1 for p in new)
    if steps in (1, 2) or steps % 10 == 0:
        record('progress-rank' + os.environ['RANK'] + '.json', dict(
            global_step=ANCHOR_STEP+steps, additional_updates=steps, time=time.time(), pid=os.getpid(),
            old_adam_age=int(optimizer.state[old[0]]['step']), new_adam_age=int(optimizer.state[new[0]]['step']) if new else None,
            new_gradients_finite_verified_at_startup=True if new else None,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, learning_rate=optimizer.param_groups[0]['lr']))
    return result


torch.optim.AdamW.step = step
if __name__ == '__main__':
    sys.argv = [str(ROOT / 'train.py'), *sys.argv[1:]]
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
