"""Execute the frozen v8-compatible K4 prior with audited GPU optimizations."""
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(os.environ.get('LASER_RUNTIME_ROOT', Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.training import rqtransformer as training
from src.models.rqtransformer.attentions import AttentionBlock

BASE = Path(os.environ['LASER_RUN_BASE'])
AUDIT = BASE / os.environ.get('LASER_PHASE', 'production') / 'verification'
AUDIT.mkdir(parents=True, exist_ok=True)
COMPILE = os.environ.get('LASER_COMPILE_BLOCKS', '1') == '1'
STOPPING = False
parsed_args = None
updates = 0
first_time = None
initial_optimizer_step = 0
model_parameters = None
profile_capture = None
profile_completed = False


def json_record(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, default=str) + '\n')
    temporary.replace(path)


def stop_requested(signum, _frame):
    global STOPPING
    STOPPING = True


for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, stop_requested)

original_parser = training.build_parser


def build_parser():
    parser = original_parser()
    parse = parser.parse_args
    def parse_args(*args, **kwargs):
        global parsed_args
        parsed_args = parse(*args, **kwargs)
        assert parsed_args.sparsity_level == 4
        assert parsed_args.total_batch_size == 2048
        assert not parsed_args.compound_tokens
        assert parsed_args.coeff_target_space == 'normalized'
        assert parsed_args.coeff_target_temperature == 0.01125
        if parsed_args.resume:
            source = parsed_args.resume_checkpoint or parsed_args.checkpoint_dir / 'last.pt'
            parsed_args.resume_checkpoint = checkpoint_io._checkpoint_upload_source(source)
        return parsed_args
    parser.parse_args = parse_args
    return parser


training.build_parser = build_parser
original_wrap = training.wrap_distributed_model


def wrap(model, backend, device, world_size):
    global model_parameters
    assert world_size == 5 and backend == 'ddp'
    model_parameters = list(model.parameters())
    for depth_block in model.head_transformer.blocks:
        depth_block.attn.short_attention_backend = 'compiled'
    if COMPILE:
        for block in model.modules():
            if isinstance(block, AttentionBlock):
                eager = block.forward
                compiled = torch.compile(eager, fullgraph=True, dynamic=True)
                def forward(x, module=block, eager=eager, compiled=compiled):
                    return compiled(x) if module.training and torch.is_grad_enabled() else eager(x)
                block.forward = forward
    original_forward = model.forward
    def forward(*args, **kwargs):
        if model.training and torch.is_grad_enabled():
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                result = original_forward(*args, **kwargs)
            assert result['atom_logits'].dtype == torch.bfloat16
            return result
        return original_forward(*args, **kwargs)
    model.forward = forward
    return DDP(model, device_ids=[device.index], broadcast_buffers=False,
               gradient_as_bucket_view=True, bucket_cap_mb=100)


training.wrap_distributed_model = wrap


def sparse_objective(atom_logits, coeff_logits, atoms, probabilities, accumulation):
    atom_log_probs = torch.nn.functional.log_softmax(atom_logits.float(), dim=-1)
    coeff_log_probs = torch.nn.functional.log_softmax(coeff_logits.float(), dim=-1)
    atom_loss = -atom_log_probs.gather(-1, atoms.long().unsqueeze(-1)).squeeze(-1)
    coeff_loss = -(probabilities * coeff_log_probs).sum(dim=-1)
    depth = atoms.shape[-1]
    return (atom_loss.sum(dim=-1) + coeff_loss.sum(dim=-1)).mean() / (2 * depth * accumulation)


training.compiled_sparse_objective = torch.compile(sparse_objective, fullgraph=True, dynamic=True) if COMPILE and os.environ.get('LASER_COMPILE_OBJECTIVE', '1') == '1' else sparse_objective

original_cosine_scheduler = training.create_cosine_lr_scheduler


def cosine_scheduler(optimizer, *, initial_lr, min_lr, total_steps, completed_steps=0, state_dict=None):
    # An explicitly requested peak-LR change preserves the current cosine
    # phase and every optimizer moment; it does not restart the schedule.
    if state_dict is not None and any(float(lr) != float(initial_lr) for lr in state_dict['base_lrs']):
        import math
        previous = list(state_dict['base_lrs'])
        assert previous == [0.0005] and initial_lr == 0.0006
        state_dict = dict(state_dict)
        current_lr = min_lr + .5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * completed_steps / total_steps))
        state_dict['base_lrs'] = [float(initial_lr)] * len(optimizer.param_groups)
        state_dict['_last_lr'] = [current_lr] * len(optimizer.param_groups)
        for group in optimizer.param_groups:
            group['lr'] = current_lr
            group['initial_lr'] = initial_lr
        if int(os.environ['RANK']) == 0:
            record = dict(previous_peak_lr=previous[0], new_peak_lr=initial_lr, current_lr=current_lr,
                          scheduler_step=completed_steps, scheduler_total_steps=total_steps,
                          optimizer_and_data_cursor_retained=True, schedule_restarted=False)
            json_record(BASE / 'production' / 'lr-adjustment.json', record)
            print('Applied requested learning-rate adjustment: ' + json.dumps(record), flush=True)
    return original_cosine_scheduler(optimizer, initial_lr=initial_lr, min_lr=min_lr,
                                     total_steps=total_steps, completed_steps=completed_steps, state_dict=state_dict)


training.create_cosine_lr_scheduler = cosine_scheduler

from src.training import k4_checkpoint_io as checkpoint_io
from src.training.background_checkpoint import BackgroundCheckpointWriter
checkpoint_writer = BackgroundCheckpointWriter()
WB = None
PHASE = os.environ.get('LASER_PHASE', 'production')
original_save = training.atomic_torch_save


def save_checkpoint(payload, target):
    if PHASE != 'production':
        return original_save(payload, target)
    saved_step = int(payload['global_step'])
    best = list(payload.get('best_fid', []))
    best_is = list(payload.get('best_inception', []))
    def committed():
        json_record(BASE / 'production' / 'durable-checkpoint.json',
                    dict(step=saved_step, target=str(target), bytes=target.stat().st_size,
                         committed_unix=time.time()))
        if WB is not None and parsed_args.upload_checkpoints:
            checkpoint_io.upload_selected_checkpoint_files(WB,
                last_checkpoint=parsed_args.checkpoint_dir / 'last.pt', best_fid=best,
                best_inception=best_is, upload_dir=parsed_args.output / 'wandb_checkpoints')
    return checkpoint_io.atomic_torch_save(payload, target, background=checkpoint_writer, on_commit=committed)


training.atomic_torch_save = save_checkpoint
training.remove_checkpoint = checkpoint_io.remove_checkpoint
original_upload = training.upload_selected_checkpoint_files


def upload_checkpoints(*args, **kwargs):
    # The durable-copy callback queues the same fixed online slots from local
    # immutable files, after shared-storage commit, without blocking training.
    if PHASE != 'production':
        return original_upload(*args, **kwargs)
    return []


training.upload_selected_checkpoint_files = upload_checkpoints
import wandb
original_wandb_init = wandb.init


def init_wandb(*args, **kwargs):
    global WB
    c = kwargs['config']
    assert c['world_size'] == 5 and c['total_batch_size'] == 2048
    c.update(stage1_rfid=4.210914134979248,
             stage1_checkpoint_sha256='dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab',
             reference_run='helloimlixin-rutgers/laser/v8dup0731113220', reference_fid=16.368215560913086,
             training_precision='bf16', compile_transformer_blocks=COMPILE,
             depth_attention_backend='compiled exact causal depth8, FP32 accumulation',
             performance_optimization='depth8_v2',
             scalar_token_shape=[8, 8, 8], sparse_pair_shape=[8, 8, 4],
             latent_noise_energy_fraction=0.006891967263072729,
             noise_rms_bins=25.6, noise_target_entropy_nats=4.661042213439941,
             evaluation_reference='ImageNet validation 50k, matching v8 training run',
             stage2_initialization='resume' if parsed_args.resume else 'scratch',
             frozen_stage1=True, asynchronous_durable_checkpoints=True)
    kwargs['allow_val_change'] = True
    WB = original_wandb_init(*args, **kwargs)
    if parsed_args.resume:
        WB.summary['continuation/target_reached'] = False
        WB.summary['continuation/training_resumed'] = True
    original_finish = WB.finish
    def finish(*args, **kwargs):
        checkpoint_writer.close()
        return original_finish(*args, **kwargs)
    WB.finish = finish
    evidence = Path(os.environ['LASER_PERSISTENT_BASE'])
    artifact = wandb.Artifact(parsed_args.wandb_id + '-launch', type='run-config')
    for name in ['recipe.yaml', 'noise-audit.json', 'preflight-summary.json', 'objective-parity.json',
                 'runtime-manifest.json', 'runtime-performance.patch', 'validation-reference.json', 'launch-provenance.json']:
        artifact.add_file(str(evidence / name), name=name)
    WB.log_artifact(artifact, aliases=['latest'])
    return WB


wandb.init = init_wandb
original_sparse_targets = training.LaserAux.sparse_targets


def profiled_sparse_targets(self, *args, **kwargs):
    global profile_capture
    if int(os.environ['RANK']) == 0 and updates == 5 and not profile_completed and profile_capture is None:
        profile_capture = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False)
        profile_capture.__enter__()
    return original_sparse_targets(self, *args, **kwargs)


training.LaserAux.sparse_targets = profiled_sparse_targets
original_step = torch.optim.AdamW.step


def step(optimizer, *args, **kwargs):
    global updates, first_time, initial_optimizer_step, profile_capture, profile_completed
    if updates == 0:
        first_time = time.monotonic()
        previous_steps = {int(state['step']) for state in optimizer.state.values()}
        assert len(previous_steps) <= 1
        initial_optimizer_step = next(iter(previous_steps), 0)
        record = dict(rank=int(os.environ['RANK']), optimizer_states=len(optimizer.state),
                      optimizer_step_before=initial_optimizer_step,
                      fresh=not parsed_args.resume, parameter_count=sum(p.numel() for p in model_parameters),
                      physical_microbatch_max=parsed_args.batch_size,
                      accumulation=int(os.environ['LASER_ACCUMULATION']), global_batch=2048,
                      precision='bf16', compiled_blocks=COMPILE, learning_rate=optimizer.param_groups[0]['lr'])
        if not parsed_args.resume:
            assert len(optimizer.state) == 0
        json_record(AUDIT / ('startup-rank' + os.environ['RANK'] + '.json'), record)
    checkpoint_writer.check()
    result = original_step(optimizer, *args, **kwargs)
    updates += 1
    if profile_capture is not None and updates == 6:
        torch.cuda.synchronize()
        profile_capture.__exit__(None, None, None)
        profile_dir = BASE / 'performance'
        profile_dir.mkdir(exist_ok=True)
        (profile_dir / 'full-training-profile.txt').write_text(profile_capture.key_averages().table(sort_by='self_cuda_time_total', row_limit=60))
        profile_capture.export_chrome_trace(str(profile_dir / 'full-training-trace.json'))
        profile_capture = None
        profile_completed = True
    if updates in (1, 20):
        tensors = list(model_parameters) + [s[k] for s in optimizer.state.values() for k in ('exp_avg', 'exp_avg_sq')]
        finite = bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all())
        assert finite
        assert {int(s['step']) for s in optimizer.state.values()} == {initial_optimizer_step + updates}
        json_record(AUDIT / ('step' + str(updates) + '-rank' + os.environ['RANK'] + '.json'),
                    dict(rank=int(os.environ['RANK']), updates=updates, finite=True,
                         optimizer_step=initial_optimizer_step + updates,
                         optimizer_states=len(optimizer.state), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                         elapsed_seconds=time.monotonic()-first_time))
    flag = torch.tensor(int(STOPPING), device=model_parameters[0].device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        parsed_args.max_optimizer_steps = updates
    return result


torch.optim.AdamW.step = step
if parsed_args is None and os.environ.get('LASER_PATCH_ONLY') != '1':
    from src.training.cli import main
    raise SystemExit(main())
