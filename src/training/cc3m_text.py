"""CC3M text-prefix training using the ImageNet physical-pair scalar prior."""
from contextlib import nullcontext
from datetime import timedelta
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import traceback
import uuid

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.seeded_bpe import SeededBPE
from src.models.physical_pair_scalar_prior import PhysicalPairScalarRQTransformer
from src.models.rqtransformer.configs import RQTransformerConfig
from src.training import rqtransformer as rq
from src.training.background_checkpoint import BackgroundCheckpointWriter
from src.training.checkpoint_upload_queue import CheckpointFile, CheckpointUploadQueue
from src.training.fid_adaptive_schedule import FidAdaptiveSchedule
from src.training.warmup_cosine_schedule import WarmupCosineSchedule
from src.training.cc3m_compound import load_cache, make_aux, prepare_reference, evaluate
from scripts.tools.build_cc3m_compound_cache import STAGE1_SHA, write_json


def build_model(options):
    released = OmegaConf.load(options['official_stage2_config'])
    config = released.arch
    config.block_size = [8, 8, 8]
    config.vocab_size = 16384 + 2048
    config.shared_cls_emb = True
    return PhysicalPairScalarRQTransformer(RQTransformerConfig.create(config), 16384)


def configure_performance(model, options):
    if options.get('compile_transformer_blocks'):
        for stack in (model.body_transformer, model.head_transformer):
            for block in stack.blocks:
                if options.get('compiled_depth_attention') and stack is model.head_transformer:
                    block.attn.short_attention_backend = 'compiled'
                block.forward = torch.compile(block.forward, dynamic=False)


def objective(model, aux, atoms, coeffs, text, temperature, accumulation=1):
    with torch.no_grad():
        tokens, (targets, probabilities) = aux.sparse_targets(
            atoms, coeffs, temp=temperature, stochastic=True, compact=True)
    outputs, text_logits = model(tokens, model_aux=aux, cond=text, amp=False)
    atom_nll = -F.log_softmax(outputs['atom_logits'].float(), -1).gather(
        -1, targets[..., None]).squeeze(-1).mean()
    coefficient_nll = -(probabilities * F.log_softmax(outputs['coeff_logits'].float(), -1)).sum(-1).mean()
    image_loss = (atom_nll + coefficient_nll) / 2
    coefficient_entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean()
    text_loss = F.cross_entropy(text_logits.flatten(0, 1).float(), text[:, 1:].reshape(-1))
    return (.9 * image_loss + .1 * text_loss) / accumulation, dict(
        image_loss=image_loss.detach(), text_loss=text_loss.detach(),
        atom_nll=atom_nll.detach(), coeff_nll=coefficient_nll.detach(),
        coeff_target_entropy=coefficient_entropy.detach(),
        coeff_kl=(coefficient_nll-coefficient_entropy).detach())


@torch.inference_mode()
def generate(model, aux, text, options):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        tokens = model.sample_sparse(len(text), aux, cond=text, amp=False,
            atom_temperature=options['atom_temperature'], atom_top_k=options['atom_top_k'],
            atom_top_p=options['atom_top_p'], coeff_temperature=options['coeff_temperature'],
            coeff_top_k=options.get('coeff_top_k', 0), coeff_top_p=options['coeff_top_p'])
        atoms = tokens[..., 0::2]
        ids = tokens[..., 1::2] - aux.num_atoms
        return ((aux.decode_compound(atoms, ids).float() + 1) / 2).clamp(0, 1)


def preview_image(pixels):
    from PIL import Image
    uint8 = pixels.detach().float().clamp(0, 1).mul(255).to(torch.uint8)
    return Image.fromarray(uint8.cpu().permute(1, 2, 0).numpy())


def log_preview(model, aux, validation, options, device, wb, step):
    import wandb
    pixels = generate(model, aux, validation['text_ids'][:8].long().to(device), options)
    wb.log({'train/global_step': step, 'samples/text_to_image': [
        wandb.Image(preview_image(pixels[i]), caption=validation['captions'][i]) for i in range(8)]})


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def is_schedule_migration(saved, options):
    migration = options.get('lr_schedule_migration') or {}
    return config_digest(saved) == migration.get('source_config_sha256')


def verify_resume_config(saved, options):
    migrating = is_schedule_migration(saved, options)
    for key in ('batch_size', 'accumulation', 'total_batch_size', 'seed',
                'coeff_scales', 'coeff_target_temperature', 'cache_sha256'):
        if saved[key] != options[key]:
            raise ValueError(f'Resume changes {key}')
    if saved['epochs'] != options['epochs'] and not (
            migrating and options['epochs'] > saved['epochs']):
        raise ValueError('Resume changes epochs without a verified schedule extension')
    for key, default in (('lr', None), ('lr_schedule', 'cosine'), ('min_lr', 0.),
                         ('fid_lr_policy', None), ('warmup_epochs', 0)):
        if saved.get(key, default) != options.get(key, default) and not migrating:
            raise ValueError(f'Resume changes {key} without a verified schedule migration')
    if saved['runtime_sha256'] != options['runtime_sha256']:
        previous = config_digest(saved['runtime_sha256'])
        expected_previous = options.get('runtime_migration_previous_manifest_sha256',
                                        options.get('preview_fix_previous_runtime_manifest_sha256'))
        if not migrating and previous != expected_previous:
            raise ValueError('Resume changes runtime_sha256')


def create_lr_scheduler(optimizer, options, updates, completed_steps=0,
                        state_dict=None, saved_config=None):
    total_steps = options['epochs'] * updates
    if options.get('lr_schedule') == 'warmup_cosine':
        return WarmupCosineSchedule(optimizer, initial_lr=options['lr'],
            min_lr=options['min_lr'], total_steps=total_steps,
            warmup_steps=round(options['warmup_epochs'] * updates),
            completed_steps=completed_steps, state_dict=state_dict)
    if options.get('lr_schedule', 'cosine') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
            T_max=total_steps, eta_min=options.get('min_lr', 0.))
        if state_dict is not None:
            scheduler.load_state_dict(state_dict)
        return scheduler
    if options['lr_schedule'] != 'fid_adaptive_cosine':
        raise ValueError('Unsupported CC3M LR schedule')
    policy = dict(initial_lr=options['lr'], min_lr=options['min_lr'],
                  total_steps=total_steps, **options['fid_lr_policy'])
    if state_dict is not None and state_dict.get('kind') == 'fid-adaptive-cosine-v1':
        source_policy = dict(initial_lr=saved_config['lr'], min_lr=saved_config['min_lr'],
            total_steps=saved_config['epochs'] * updates,
            **saved_config['fid_lr_policy']) if saved_config else policy
        if state_dict['policy'] != source_policy or state_dict['last_epoch'] != completed_steps:
            raise ValueError('Source adaptive scheduler/configuration step or policy disagree')
        source_lr = FidAdaptiveSchedule.lr_at_step(source_policy, completed_steps, state_dict['multiplier'])
        if any(abs(group['lr'] - source_lr) > 1e-12 for group in optimizer.param_groups):
            raise ValueError('Source optimizer and adaptive scheduler LR disagree')
        if source_policy != policy:
            migration = options.get('lr_schedule_migration') or {}
            if (saved_config is None or not is_schedule_migration(saved_config, options)
                    or completed_steps != migration.get('source_step')
                    or config_digest(state_dict) != migration.get('source_scheduler_sha256')):
                raise ValueError('Adaptive LR change requires the verified source adaptive checkpoint')
            target_lr = FidAdaptiveSchedule.lr_at_step(policy, completed_steps, state_dict['multiplier'])
            if (target_lr > source_lr + 1e-15 or policy['min_lr'] > source_policy['min_lr']):
                raise ValueError('Adaptive continuation migration must lower the LR')
            # Retain the completed step, best FID, observations, and reductions.
            # Only the explicitly verified learning-rate policy changes.
            state_dict = dict(state_dict, policy=policy)
    elif state_dict is not None:
        migration = options.get('lr_schedule_migration') or {}
        if (saved_config is None or not is_schedule_migration(saved_config, options)
                or completed_steps != migration.get('source_step')
                or state_dict.get('last_epoch') != completed_steps
                or state_dict.get('T_max') != total_steps
                or state_dict.get('eta_min') != saved_config.get('min_lr', 0.)
                or state_dict.get('base_lrs') != [saved_config['lr']] * len(optimizer.param_groups)):
            raise ValueError('Adaptive LR migration requires the verified source cosine checkpoint')
        source_lr = saved_config.get('min_lr', 0.) + .5 * (
            saved_config['lr'] - saved_config.get('min_lr', 0.)) * (
            1 + math.cos(math.pi * completed_steps / total_steps))
        if (len(state_dict.get('_last_lr', [])) != len(optimizer.param_groups)
                or any(abs(lr - source_lr) > 1e-12 for lr in state_dict['_last_lr'])
                or any(abs(group['lr'] - source_lr) > 1e-12 for group in optimizer.param_groups)):
            raise ValueError('Source optimizer and cosine scheduler LR disagree')
        initial = FidAdaptiveSchedule(optimizer, **policy)
        state_dict = initial.state_dict()
        state_dict.update(last_epoch=completed_steps, last_observation_step=completed_steps)
    elif state_dict is None and options.get('lr_schedule_migration'):
        raise ValueError('Continuation requires a full-state source checkpoint')
    return FidAdaptiveSchedule(optimizer, **policy, completed_steps=completed_steps,
                               state_dict=state_dict)


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def verify_resume_metrics(results, expected):
    if (results['items'] != expected['items']
            or not math.isclose(results['fid'], expected['fid'], rel_tol=0., abs_tol=1e-6)
            or not math.isclose(results['clip_score'], expected['clip_score'], rel_tol=0., abs_tol=1e-8)):
        raise ValueError('Resumed checkpoint did not reproduce its completed FID/CLIP evaluation')


def verify_checkpoint_progress(state, updates, accumulation, world):
    step, epoch, micro = state['global_step'], state['epoch'], state['next_microbatch']
    if (not state.get('resume_capable') or state['world_size'] != world
            or len(state['rng_state_by_rank']) != world
            or not 0 <= micro <= updates * accumulation or micro % accumulation
            or epoch < 0 or step != epoch * updates + micro // accumulation
            or state['scheduler']['last_epoch'] != step
            or (step and not state['optimizer']['state'])
            or any(float(value['step']) != step for value in state['optimizer']['state'].values())):
        raise ValueError('Checkpoint model boundary, optimizer, scheduler, data cursor, or rank RNG disagree')


def capture_rng(device, group=None):
    numpy_state = np.random.get_state()
    state = dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(device),
        python=random.getstate(), numpy=(numpy_state[0], torch.from_numpy(numpy_state[1].astype(np.int64)),
            numpy_state[2], numpy_state[3], numpy_state[4]))
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(state, gathered, dst=0, group=group)
    return gathered


def restore_rng(state, device):
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state(state['torch_cuda'], device)
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], value[1].numpy().astype(np.uint32), value[2], value[3], value[4]))


def serialize_checkpoint(payload, slots, options, local_slots=False):
    local = Path(options['local_checkpoints'])
    local.mkdir(parents=True, exist_ok=True)
    immutable = local / (f"step-{payload['global_step']:09d}-epoch-{payload['epoch']:03d}-"
                         f"{uuid.uuid4().hex}.pt")
    rq.atomic_torch_save(payload, immutable)
    expected_size = immutable.stat().st_size
    with immutable.open('rb') as source:
        digest = base64.b64encode(hashlib.file_digest(source, 'md5').digest()).decode()
    if local_slots:
        for slot in slots:
            rq._replace_hard_link(immutable, local / slot)
    return CheckpointFile(immutable, payload['global_step'], payload['epoch'],
                          expected_size, digest, tuple(slots))


def upload_checkpoint_file(item, slots, options, wb):
    """Publish stable upload aliases and acknowledge full file checksums."""
    local = Path(options['local_checkpoints'])
    staging = local / 'online-checkpoints' if options.get('checkpoint_async_upload') else local
    staging.mkdir(parents=True, exist_ok=True)
    persistent = Path(options['output']) / 'checkpoints'
    persistent.mkdir(parents=True, exist_ok=True)
    import wandb
    api = wandb.Api(timeout=60)
    run_path = f"{options['wandb_entity']}/{options['wandb_project']}/{wb.id}"
    run = api.run(run_path)
    pending = set(slots)
    for slot in tuple(pending):
        try:
            remote = run.file(slot)
        except ValueError:
            continue
        if remote.size == item.size and remote.md5 == item.md5:
            pending.remove(slot)
    for slot in slots:
        if slot not in pending:
            continue  # An identical full state is already durably acknowledged.
        destination = persistent / slot
        for attempt in range(3):
            temp = destination.with_suffix('.partial')
            try:
                with item.path.open('rb') as reader, temp.open('wb') as writer:
                    shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
                if temp.stat().st_size != item.size:
                    raise OSError('Checkpoint persistence truncated')
                temp.replace(destination)
                break
            except OSError:
                temp.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(5)
        rq._replace_hard_link(item.path, staging / slot)
        wb.save(str(staging / slot), base_path=str(staging), policy='now')
    deadline = time.monotonic() + 1800
    while pending:
        if time.monotonic() > deadline:
            raise TimeoutError(f'W&B did not acknowledge checkpoint uploads: {sorted(pending)}')
        run = api.run(run_path)
        for slot in tuple(pending):
            try:
                remote = run.file(slot)
            except ValueError:
                continue
            if remote.size == item.size and remote.md5 == item.md5:
                pending.remove(slot)
        if pending:
            time.sleep(10)
    receipt = dict(global_step=item.step, epoch=item.epoch,
        slots=list(slots), bytes=item.size, md5=item.md5, online_verified=True,
        includes=['model', 'optimizer', 'scheduler', 'data_cursor', 'all_rank_rng',
                  'config', 'cache_metadata', 'best_fid', 'best_clip', 'deterministic_bpe_policy'])
    write_json(Path(options['output']) / 'last-upload.json', receipt)
    summary = {'checkpoints/full_resume_state': True, 'checkpoints/online_verified': True}
    if 'last.pt' in slots:
        summary['checkpoints/last_verified_step'] = item.step
    wb.summary.update(summary)
    print(json.dumps(dict(phase='checkpoint_online_verified', **receipt)), flush=True)


def commit_checkpoint(payload, slots, options, wb):
    item = serialize_checkpoint(payload, slots, options)
    upload_checkpoint_file(item, slots, options, wb)
    item.path.unlink(missing_ok=True)


def commit_local_checkpoint(payload, slots, options, wb, uploader):
    item = serialize_checkpoint(payload, slots, options, local_slots=True)
    wb.summary['checkpoints/local_committed_step'] = item.step
    print(json.dumps(dict(phase='checkpoint_local_committed', global_step=item.step,
        epoch=item.epoch, slots=list(slots), bytes=item.size, md5=item.md5)), flush=True)
    uploader.submit(item)


def retryable_upload_error(error):
    import wandb
    return isinstance(error, (OSError, TimeoutError, ConnectionError, wandb.errors.CommError))


def abort_training(error, output, rank, step=None):
    """Expose the exception before exiting; never wait for failed peers to clean up."""
    detail = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    secret = os.environ.get('WANDB_API_KEY')
    if secret:
        detail = detail.replace(secret, '[redacted]')
    record = dict(phase='failed', timestamp=time.time(), rank=rank, global_step=step,
                  error_type=type(error).__name__, traceback=detail)
    try:
        Path(output).mkdir(parents=True, exist_ok=True)
        write_json(Path(output) / f'failure-rank-{rank:03d}.json', record)
    finally:
        print(detail, file=sys.stderr, flush=True)
        sys.stdout.flush()
        # A distributed destructor can block forever when another rank is still
        # in a collective. torchrun must see the failure and restart the ranks.
        os._exit(1)


def run(cfg):
    try:
        _run(cfg)
    except BaseException as error:
        abort_training(error, cfg.options.output, int(os.environ.get('RANK', 0)))


def _run(cfg):
    options = OmegaConf.to_container(cfg.options, resolve=True)
    rank, world, local = [int(os.environ[key]) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(hours=2))
    checkpoint_group = dist.new_group(backend='gloo', timeout=timedelta(hours=2))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(options['seed'])
    output = Path(options['output'])
    output.mkdir(parents=True, exist_ok=True)
    with Path(options['checkpoint']).open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != STAGE1_SHA:
            raise ValueError('Wrong frozen ImageNet rFID4.21 autoencoder')
    train = load_cache(options['token_cache'])
    validation = load_cache(options['validation_cache'])
    assert train['meta']['coeff_scales'] == validation['meta']['coeff_scales'] == options['coeff_scales']
    assert len(train['atoms']) == options['train_items']
    assert len(validation['atoms']) == options['validation_items']
    batch, accumulation = options['batch_size'], options['accumulation']
    assert batch * accumulation * world == options['total_batch_size']
    updates = len(train['atoms']) // options['total_batch_size']
    microbatches = updates * accumulation
    prepare_reference(options, device)
    aux = make_aux(options, options['coeff_scales'], device)
    prior = build_model(options).to(device)
    configure_performance(prior, options)
    model = DDP(prior, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(prior.parameters(), lr=options['lr'],
        betas=tuple(options.get('betas', (.9, .95))),
        weight_decay=options.get('weight_decay', 1e-4), fused=True)
    step, start_epoch, start_micro = 0, 0, 0
    best_fid, best_clip = math.inf, -math.inf
    scheduler_state = saved_config = None
    latest = Path(options.get('resume_checkpoint') or output / 'checkpoints' / 'last.pt')
    if options.get('resume_checkpoint') and not latest.is_file():
        raise FileNotFoundError(f'Requested resume checkpoint is missing: {latest}')
    if options['resume'] and latest.is_file():
        state = torch.load(latest, weights_only=True, map_location='cpu', mmap=True)
        verify_checkpoint_progress(state, updates, accumulation, world)
        verify_resume_config(state['config'], options)
        if state['world_size'] != world:
            raise ValueError('Exact resume requires the same number of ranks')
        prior.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler_state, saved_config = state['scheduler'], state['config']
        step, start_epoch, start_micro = state['global_step'], state['epoch'], state['next_microbatch']
        best_fid, best_clip = state['best_fid'], state['best_clip']
        restore_rng(state['rng_state_by_rank'][rank], device)
        if rank == 0:
            print(json.dumps(dict(phase='full_state_restored', global_step=step,
                epoch=start_epoch, next_microbatch=start_micro, rank_rng_count=world,
                optimizer_state_count=len(optimizer.state), scheduler_step=scheduler_state['last_epoch'])), flush=True)
        del state
    else:
        torch.manual_seed(options['seed'] + rank)
        random.seed(options['seed'] + rank)
        np.random.seed(options['seed'] + rank)
        if rank == 0:
            print(json.dumps(dict(phase='fresh_random_initialization', global_step=0,
                optimizer_state_count=len(optimizer.state), stage2_checkpoint_loaded=False,
                seed=options['seed'])), flush=True)
    if saved_config is not None and is_schedule_migration(saved_config, options):
        global_best = options.get('resume_global_best_metrics') or {}
        best_fid = min(best_fid, global_best.get('fid', best_fid))
        best_clip = max(best_clip, global_best.get('clip_score', best_clip))
    scheduler = create_lr_scheduler(optimizer, options, updates, step, scheduler_state, saved_config)
    if rank == 0:
        print(json.dumps(dict(phase='lr_schedule_ready', schedule=options.get('lr_schedule', 'cosine'),
            global_step=step, lr=optimizer.param_groups[0]['lr'],
            migrated=saved_config is not None and is_schedule_migration(saved_config, options))), flush=True)
    tokenizer = SeededBPE(options['bpe_dropout'])
    wb = writer = uploader = None
    if rank == 0:
        import wandb
        wb = wandb.init(entity=options['wandb_entity'], project=options['wandb_project'],
            id=options['wandb_id'], name=options['wandb_name'], mode='online', resume='allow',
            config=options, allow_val_change=True)
        wb.define_metric('train/global_step')
        wb.define_metric('train/*', step_metric='train/global_step')
        wb.define_metric('val/*', step_metric='train/global_step')
        wb.summary.update({'pipeline/phase': 'training', 'stage1/rfid': 4.210914134979248,
            'model/parameters': sum(p.numel() for p in prior.parameters()),
            'checkpoints/full_resume_state': True, 'cache/coeff_scales': options['coeff_scales'],
            'samples/pixel_encoding': 'uint8_rgb_0_255'})
        for name in ['recipe.yaml', 'launch-provenance.json', 'throughput.json', 'runtime-manifest.json',
                     'continuation-provenance.json', 'continuation.diff',
                     'lr-reduction-source-audit.json', 'lr-reduction-stop.json',
                     'preview-fix.json', 'preview-fix.diff', 'scratch-provenance.json',
                     'official-stage2.yaml', 'official-training.yaml',
                     'original-best-resume-plan.json', 'original-best-resume-cloud-source.json',
                     'original-best-upload-fix.json']:
            path = output.parent / name
            if path.is_file():
                wb.save(str(path), base_path=str(output.parent), policy='now')
        for name in ['sampler-comparison.json', 'reference-run.json']:
            path = output.parent / name
            if path.is_file():
                wb.save(str(path), base_path=str(output.parent), policy='now')
        assets = wandb.Artifact(wb.id + '-reproducibility', type='training-assets',
            metadata=dict(stage1_sha256=STAGE1_SHA, cache_sha256=options['cache_sha256'],
                dataset_revision=options['dataset_revision'], world_size=world))
        for path in [Path(options['checkpoint']), Path(options['token_cache']),
                     Path(options['validation_cache']), Path(options['fid_reference_stats']),
                     output.parent / 'runtime.tar.gz', output.parent / 'recipe.yaml',
                     output.parent / 'runtime-manifest.json', output.parent / 'environment.json',
                     Path(options['official_stage2_config'])]:
            assets.add_file(str(path), name=path.name, skip_cache=True)
        wb.log_artifact(assets, aliases=['latest', 'exact-resume-assets'])
        writer = BackgroundCheckpointWriter()
        if options.get('checkpoint_async_upload'):
            def upload_retry(item, slots, error, attempt):
                record = dict(phase='checkpoint_upload_retry', timestamp=time.time(),
                    global_step=item.step, slots=list(slots), attempt=attempt,
                    error_type=type(error).__name__, local_state_preserved=True)
                write_json(output / 'upload-retry.json', record)
                print(json.dumps(record), flush=True)
            uploader = CheckpointUploadQueue(lambda item, slots:
                upload_checkpoint_file(item, slots, options, wb),
                retry_delay=options.get('checkpoint_upload_retry_seconds', 30),
                retryable=retryable_upload_error, on_retry=upload_retry)
            if step:
                # Reload durable slot aliases after an interrupted upload. This
                # also restores an outstanding best winner when no new best
                # occurs immediately after the optimizer state is restored.
                recovered = {}
                for slot in ['best-fid.pt', 'best-clip.pt', 'last.pt']:
                    path = Path(options['local_checkpoints']) / slot
                    if not path.is_file():
                        continue
                    saved = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
                    verify_checkpoint_progress(saved, updates, accumulation, world)
                    verify_resume_config(saved['config'], options)
                    key = (saved['global_step'], saved['epoch'], saved['next_microbatch'])
                    if key not in recovered:
                        immutable = path.parent / ('recovered-' + path.name)
                        rq._replace_hard_link(path, immutable)
                        with immutable.open('rb') as source:
                            md5 = base64.b64encode(hashlib.file_digest(source, 'md5').digest()).decode()
                        recovered[key] = [immutable, saved['global_step'], saved['epoch'],
                                          immutable.stat().st_size, md5, []]
                    recovered[key][-1].append(slot)
                    del saved
                for values in sorted(recovered.values(), key=lambda value: value[1]):
                    uploader.submit(CheckpointFile(*values[:-1], tuple(values[-1])))
                print(json.dumps(dict(phase='checkpoint_uploads_recovered',
                    states=len(recovered), global_step=step)), flush=True)
    stopping = {'requested': False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.update(requested=True))

    def save(epoch, next_micro, metrics=None, synchronous=False, force_best_fid=False):
        nonlocal best_fid, best_clip
        rng = capture_rng(device, checkpoint_group)
        if rank == 0:
            writer.wait()
            slots = ['last.pt']
            if metrics is not None:
                if force_best_fid and not math.isclose(metrics['fid'], best_fid, abs_tol=1e-6, rel_tol=0.):
                    raise ValueError('Resume best-FID upload must reproduce the run-wide winner')
                if metrics['fid'] < best_fid or force_best_fid:
                    best_fid = metrics['fid']
                    slots.append('best-fid.pt')
                if metrics['clip_score'] > best_clip:
                    best_clip = metrics['clip_score']
                    slots.append('best-clip.pt')
            payload = cpu_snapshot(dict(model=prior.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), epoch=epoch, next_microbatch=next_micro,
                global_step=step, rng_state_by_rank=rng, world_size=world, config=options,
                best_fid=best_fid, best_clip=best_clip, metrics=metrics, cache_meta=train['meta'],
                resume_capable=True, checkpoint_kind='full_training_state', checkpoint_slots=slots,
                bpe_policy='merge_dropout_python_rng_seeded_by_seed_epoch_image_index_v1'))
            writer.submit(lambda: commit_local_checkpoint(payload, slots, options, wb, uploader)
                          if uploader is not None else commit_checkpoint(payload, slots, options, wb))
            if synchronous:
                writer.wait()
                if uploader is not None:
                    uploader.wait()
        dist.barrier()

    window_time, window_step = time.monotonic(), step
    migrating_source = saved_config is not None and is_schedule_migration(saved_config, options)
    pending_completion = options.get('finish_epoch_on_resume') and migrating_source
    saved_resume_boundary = False
    try:
        if step and options.get('evaluate_on_resume') and (
                not options.get('resume_expected_metrics') or migrating_source
                or step == options.get('resume_expected_step')):
            with torch.random.fork_rng(devices=[local]):
                torch.manual_seed(options['eval_seed'] + rank)
                results = evaluate(prior, aux, validation, options, device, wb, step, generator=generate)
            if options.get('resume_expected_metrics'):
                verify_resume_metrics(results, options['resume_expected_metrics'])
            if pending_completion:
                if start_micro != microbatches:
                    raise ValueError('Epoch completion requires the saved final optimizer boundary')
                decision = scheduler.observe(results['fid'])
                start_epoch += 1
                start_micro = 0
                if rank == 0 and decision is not None:
                    print(json.dumps(dict(phase='fid_lr_decision', **decision)), flush=True)
            if rank == 0:
                wb.log({'train/global_step': step, **{'val/' + key: value for key, value in results.items()}})
                wb.summary.update({'best/fid': min(results['fid'], best_fid),
                    'best/clip_score': max(results['clip_score'], best_clip)})
                write_json(output / 'evaluation/continuation-baseline.json', results)
                print(json.dumps(dict(phase='continuation_baseline', epoch=start_epoch,
                    global_step=step, **results)), flush=True)
            save(start_epoch, start_micro, results,
                 force_best_fid=bool(options.get('checkpoint_best_fid_on_resume') and migrating_source))
            saved_resume_boundary = True
        if step and options.get('preview_on_resume'):
            with torch.random.fork_rng(devices=[local]):
                torch.manual_seed(options['eval_seed'] + rank)
                prior.eval()
                if rank == 0:
                    log_preview(prior, aux, validation, options, device, wb, step)
                prior.train()
            dist.barrier()
        if step and options.get('checkpoint_on_resume') and not saved_resume_boundary:
            save(start_epoch, start_micro)
        for epoch in range(start_epoch, options['epochs']):
            order = torch.randperm(len(train['atoms']), generator=torch.Generator().manual_seed(options['seed'] + epoch))
            first = start_micro if epoch == start_epoch else 0
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for micro in range(first, microbatches):
                if writer is not None:
                    writer.check()
                if uploader is not None:
                    uploader.check()
                rows = order[(micro * world + rank) * batch:(micro * world + rank + 1) * batch]
                atoms = train['atoms'][rows].to(device, dtype=torch.long)
                coeffs = train['coeffs'][rows].to(device)
                captions = [train['captions'][i] for i in rows.tolist()]
                text = torch.tensor([item.ids for item in tokenizer.encode_batch(captions,
                    indices=rows.tolist(), epoch=epoch, seed=options['seed'])], device=device)
                sync = (micro + 1) % accumulation == 0
                with (nullcontext() if sync else model.no_sync()), torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, details = objective(model, aux, atoms, coeffs, text,
                        options['coeff_target_temperature'], accumulation)
                    loss.backward()
                if not sync:
                    continue
                grad = torch.nn.utils.clip_grad_norm_(prior.parameters(),
                    options.get('grad_clip_norm', 1.), error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                stop = torch.tensor(int(stopping['requested']), device=device)
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                if rank == 0 and (step == 1 or step % 10 == 0):
                    now = time.monotonic()
                    metrics = {'train/global_step': step, 'train/loss': float(loss.detach()) * accumulation,
                        'train/grad_norm': float(grad), 'train/lr': optimizer.param_groups[0]['lr'],
                        'train/epoch': epoch + (micro + 1) / microbatches,
                        'train/images_per_second': (step - window_step) * options['total_batch_size'] / (now - window_time),
                        **{'train/' + key: float(value) for key, value in details.items()}}
                    wb.log(metrics)
                    write_json(output / 'status.json', dict(phase='training', timestamp=time.time(), **metrics))
                    print(json.dumps(metrics), flush=True)
                    window_time, window_step = now, step
                first_metrics = None
                if step == 1 and options.get('evaluate_at_first_step') and not int(stop):
                    with torch.random.fork_rng(devices=[local]):
                        torch.manual_seed(options['eval_seed'] + rank)
                        first_metrics = evaluate(prior, aux, validation, options, device, wb,
                                                 step, generator=generate)
                    if isinstance(scheduler, FidAdaptiveSchedule):
                        scheduler.observe(first_metrics['fid'])
                    if rank == 0:
                        wb.log({'train/global_step': step,
                            **{'val/' + key: value for key, value in first_metrics.items()}})
                        wb.summary.update({'best/fid': first_metrics['fid'],
                                           'best/clip_score': first_metrics['clip_score']})
                        write_json(output / 'evaluation/initial-step-001.json', first_metrics)
                        print(json.dumps(dict(phase='generation_evaluation', epoch=0,
                            global_step=step, **first_metrics)), flush=True)
                if step == 1 or step % options['save_step_freq'] == 0 or int(stop):
                    save(epoch, micro + 1, first_metrics, synchronous=bool(int(stop)))
                if int(stop):
                    return
                if step % options['sample_grid_every'] == 0:
                    with torch.random.fork_rng(devices=[local]):
                        torch.manual_seed(options['eval_seed'] + rank)
                        prior.eval()
                        if rank == 0:
                            log_preview(prior, aux, validation, options, device, wb, step)
                        prior.train()
                    dist.barrier()
            # Commit the optimizer boundary before entering lengthy evaluation.
            save(epoch, microbatches)
            with torch.random.fork_rng(devices=[local]):
                torch.manual_seed(options['eval_seed'] + rank)
                results = evaluate(prior, aux, validation, options, device, wb, step, generator=generate)
            decision = scheduler.observe(results['fid']) if isinstance(scheduler, FidAdaptiveSchedule) else None
            if rank == 0:
                wb.log({'train/global_step': step, **{'val/' + key: value for key, value in results.items()}})
                wb.summary.update({'best/fid': min(results['fid'], best_fid),
                    'best/clip_score': max(results['clip_score'], best_clip)})
                write_json(output / f'evaluation/epoch-{epoch + 1:03d}.json', results)
                print(json.dumps(dict(phase='generation_evaluation', epoch=epoch + 1,
                    global_step=step, **results)), flush=True)
                if decision is not None:
                    wb.log({'train/global_step': step, 'train/lr': optimizer.param_groups[0]['lr'],
                            **{'lr_schedule/' + key: value for key, value in decision.items()}})
                    write_json(output / f'lr-schedule/epoch-{epoch + 1:03d}.json', decision)
                    print(json.dumps(dict(phase='fid_lr_decision', **decision)), flush=True)
            save(epoch + 1, 0, results)
            start_micro = 0
        if rank == 0:
            wb.summary['pipeline/phase'] = 'completed'
            write_json(output / 'complete.json', dict(global_step=step, epochs=options['epochs']))
    except BaseException as error:
        abort_training(error, output, rank, step)
    finally:
        try:
            if writer is not None:
                writer.close()
            if uploader is not None:
                uploader.close()
            if wb is not None:
                wb.finish(exit_code=0 if step == options['epochs'] * updates else 1)
            dist.destroy_process_group()
        except BaseException as error:
            abort_training(error, output, rank, step)
