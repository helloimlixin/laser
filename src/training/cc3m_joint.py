"""CC3M text-prefix training with the frozen ImageNet joint LASER dictionary."""
import argparse
import concurrent.futures
from contextlib import nullcontext
from datetime import timedelta
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from src import rq_upstream
from src.data.cc3m_joint import CC3MJointDataset, epoch_order, write_json
from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer
from src.scaled_atom_training import soft_cross_entropy
from src.training.rq_sdpa_backend import sdpa_attention
from src.training.rq_precision import native_bf16_autocast
from src.training.cc3m_preview import PROMPTS, SAMPLES_PER_PROMPT, conditioning_tokens, save_text_image_grid
from src.training.rq_rng import POLICY, configure_training_rng, audit_training_rng, isolated_rng
from src.training.cc3m_storage import WorkspaceMirror, local_run_root


_RUNTIME_EXECUTION = {}


def configure_execution(overrides):
    """Apply only execution controls, preserving the checkpoint training recipe."""
    allowed = {'short_sequence_math', 'cudnn_benchmark', 'cudnn_benchmark_limit', 'checkpoint_every', 'attention_batch_limit', 'encoder_batch',
               'target_chunk', 'ce_chunk_size'}
    if not isinstance(overrides, dict) or set(overrides)-allowed:
        raise ValueError('Unsupported CC3M execution override')
    for key, value in overrides.items():
        if key in {'short_sequence_math', 'cudnn_benchmark'}:
            if not isinstance(value, bool):
                raise ValueError(f'{key} must be boolean')
        elif type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    _RUNTIME_EXECUTION.clear()
    _RUNTIME_EXECUTION.update(overrides)


def execution_options(cfg):
    return (dict(encoder_batch=cfg['encoder_batch'], target_chunk=cfg['target_chunk'],
                 ce_chunk_size=1024) | cfg.get('execution', {}) | _RUNTIME_EXECUTION)


def attention_options(cfg):
    execution = execution_options(cfg)
    return dict(batch_limit=execution.get('attention_batch_limit'),
                short_sequence_math=execution.get('short_sequence_math', False))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_snapshot(v) for v in value)
    return value


def build_model(cfg, device):
    arch = OmegaConf.create(cfg['arch'])
    model = rq_upstream.RQTransformer(rq_upstream.RQTransformerConfig.create(arch))
    model.apply(model._init_weights)
    execution = cfg.get('execution', {})
    if execution.get('align_classifier'):
        from src.models.laser_joint_execution import align_joint_classifier
        align_joint_classifier(model, multiple=int(execution.get('alignment_multiple', 64)))
    return model.to(device)


def build_aux(cfg, device):
    if cfg.get('tokenizer_backend') == 'native_vector':
        from src.models.native_vector_tokenizer import NativeVectorTokenizer, file_digest, module_digest
        for key in ('stage1', 'codebook'):
            if file_digest(cfg[key]) != cfg[key+'_sha256']:
                raise ValueError(f'Native vector {key} SHA256 mismatch')
        aux = NativeVectorTokenizer(cfg['stage1'], cfg['codebook'])
        if cfg['arch']['vocab_size'] != aux.quantizer.vocab_size:
            raise ValueError('Transformer vocabulary must match the frozen vector tables')
        if module_digest(aux.backbone) != aux.spec['source_cnn_digest']:
            raise ValueError('Frozen LASER backbone differs from the tokenizer artifact')
        return aux.to(device).eval()
    if cfg.get('tokenizer_backend') == 'native_beam_pair':
        from src.models.rfid_preserved_joint_tokenizer import RFIDPreservedJointTokenizer
        return RFIDPreservedJointTokenizer(cfg['stage1'], cfg['codebook'],
            cfg['reconstruction_codebook']).to(device).eval()
    return FrozenJointLaserTokenizer(cfg['stage1'], cfg['codebook']).to(device).eval()


def optimizer_for(model, cfg):
    # Match the published optimizer, including decay on every parameter.
    return torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=tuple(cfg['betas']),
                            weight_decay=cfg['weight_decay'], fused=True)


_TEACHER_AUDITED = set()


def image_targets(aux, latent, cfg):
    """Draw prefixes and soft labels from the configured frozen teacher."""
    if cfg.get('tokenizer_backend') == 'native_vector':
        # Chunk by images: the upstream method has no chunk_size argument.
        # Prefixes and labels both come from its full-vocabulary posterior.
        parts = [aux.quantizer.get_soft_codes(chunk, temp=cfg['temperatures'], stochastic=True)
                 for chunk in latent.split(execution_options(cfg)['target_chunk'])]
        return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    if cfg.get('image_teacher') == 'beam_aligned':
        from src.training.beam_aligned_teacher import aligned_targets
        options = cfg['beam_teacher']
        if options['beam_width'] != 16 or options['max_relative_candidate_error'] != .01:
            raise ValueError('The validated beam teacher requires beam16 and the 1% candidate budget')
        pool_fn = None
        if cfg.get('execution', {}).get('exact_beam_cost', False):
            from src.training.beam_aligned_kernels import beam_pool
            pool_fn = beam_pool
        audit_key = (cfg.get('run_id'), os.environ.get('RANK', '0'))
        audit = audit_key not in _TEACHER_AUDITED
        result = aligned_targets(aux.physical_book, latent,
            float(options['sequence_temperature']), return_details=audit, pool_fn=pool_fn)
        targets, codes = result[:2]
        if audit:
            _, _, pool, weights, chosen = result
            if not torch.isfinite(targets).all():
                raise FloatingPointError('Nonfinite beam-aligned targets')
            torch.testing.assert_close(targets.sum(-1), torch.ones_like(codes, dtype=targets.dtype),
                                       rtol=2e-5, atol=2e-5)
            conditional_probability = targets.gather(-1, codes[..., None]).squeeze(-1).prod(-1).reshape(-1)
            joint_probability = weights.gather(1, chosen[:, None]).squeeze(-1)
            torch.testing.assert_close(conditional_probability, joint_probability, rtol=1e-4, atol=1e-6)
            canonical_cost = pool['costs'].gather(1, pool['canonical_index'][:, None])
            relative_error = ((pool['costs']-canonical_cost).clamp_min(0)*weights).sum(-1)/canonical_cost[:, 0].clamp_min(1e-8)
            write_json(local_run_root(cfg)/f'teacher-audit-rank{audit_key[1]}.json', dict(
                passed=True, teacher='beam16 sequence posterior with exact prefix conditionals',
                sequence_temperature=float(options['sequence_temperature']), beam_width=16,
                vocabulary=aux.physical_book.vocab_size, soft_targets=True,
                prefixes_and_targets_share_joint_distribution=True,
                ar_joint_probability_verified=True, native_greedy_teacher_used=False,
                entropy_per_depth=[float(-(targets[..., d, :]*targets[..., d, :].clamp_min(1e-30).log()).sum(-1).mean())
                                   for d in range(codes.shape[-1])],
                mean_relative_latent_error_above_canonical=float(relative_error.mean()),
                exact_beam_cost=pool_fn is not None))
            _TEACHER_AUDITED.add(audit_key)
        return targets, codes
    execution = execution_options(cfg)
    if cfg.get('tokenizer_backend') == 'native_beam_pair':
        parts = [aux.quantizer.get_soft_codes(chunk, temp=cfg['temperatures'], stochastic=True)
                 for chunk in latent.split(execution['target_chunk'])]
        return torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts])
    return aux.quantizer.get_soft_codes(latent, cfg['temperatures'],
        stochastic=True, chunk_size=execution['target_chunk'])


def objective(model, aux, images, text, cfg):
    execution = execution_options(cfg)
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        latent = torch.cat([aux.encode(x.float()) for x in images.split(execution['encoder_batch'])])
        targets, codes = image_targets(aux, latent, cfg)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits, text_logits = model(codes, model_aux=aux, cond=text, amp=True)
        if logits.dtype != torch.bfloat16 or text_logits.dtype != torch.bfloat16:
            raise RuntimeError('Native RQ forward disabled the requested BF16 autocast')
        image_loss = soft_cross_entropy(logits, targets, chunk_size=execution['ce_chunk_size'])
        text_loss = torch.nn.functional.cross_entropy(
            text_logits.flatten(0, 1).float(), text[:, 1:].reshape(-1))
        loss = cfg['image_loss_weight']*image_loss + cfg['text_loss_weight']*text_loss
    return loss, torch.stack((image_loss.detach(), text_loss.detach()))


@torch.inference_mode()
def generate(model, aux, text, cfg):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        codes = model.sample(torch.zeros(len(text), *aux.code_shape, dtype=torch.long, device=text.device),
            model_aux=aux, cond=text, amp=True, cached=True, is_tqdm=False,
            temperature=cfg['sampling']['temperature'], top_k=[cfg['sampling']['top_k']],
            top_p=[cfg['sampling']['top_p']])
    with torch.autocast('cuda', enabled=False):
        decoder_batch = int(cfg.get('execution', {}).get('decoder_batch', cfg['encoder_batch']))
        pixels = torch.cat([((aux.decode_code(chunk).float()+1)/2).clamp(0, 1)
                            for chunk in codes.split(decoder_batch)])
        if not torch.isfinite(pixels).all():
            raise FloatingPointError('Nonfinite decoded preview pixels')
        return pixels


def dataset(cfg, split, epoch=0):
    manifest = json.loads(Path(cfg['data_manifest']).read_text())
    return CC3MJointDataset(manifest[split], cfg['data_source'], cfg['data_local'],
        train=split == 'train', seed=cfg['seed'], epoch=epoch)


def loader(data, indices, batch, workers):
    generator = torch.Generator().manual_seed(
        int(getattr(data, 'seed', 0))+1009*int(getattr(data, 'epoch', 0)))
    return DataLoader(data, batch_size=batch, sampler=indices, num_workers=workers,
        pin_memory=True, drop_last=False, generator=generator,
        **(dict(prefetch_factor=2) if workers else {}))


def isolated_evaluation(function):
    @wraps(function)
    def wrapped(model, aux, cfg, device, *args, **kwargs):
        rank = dist.get_rank() if dist.is_initialized() else 0
        was_training = model.training
        try:
            with isolated_rng(cfg['seed']+1701+rank, device):
                return function(model, aux, cfg, device, *args, **kwargs)
        finally:
            model.train(was_training)
    return wrapped


def preview_image(pixels):
    from PIL import Image
    data = pixels.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8)
    return Image.fromarray(data.cpu().permute(1, 2, 0).numpy())


@isolated_evaluation
def preview(model, aux, cfg, device, wb, step, output):
    import wandb
    text = conditioning_tokens(device)
    model.eval()
    pixels = generate(model, aux, text, cfg)
    path = output/f'samples/text-grid-9x8-step-{step:07d}.png'
    path.parent.mkdir(exist_ok=True)
    save_text_image_grid(pixels, path)
    write_json(path.with_suffix('.json'), dict(global_step=step, prompts=list(PROMPTS),
        images_per_prompt=SAMPLES_PER_PROMPT, total_images=len(pixels),
        layout='one captioned row per prompt, eight images per row', pixel_encoding='uint8_rgb_0_255'))
    if wb is not None:
        wb.log({'train/global_step': step,
            'samples/text_grid_9x8': wandb.Image(str(path), caption=f'Step {step}: nine prompts, eight samples each'),
            'samples/text_to_image': [wandb.Image(preview_image(pixels[i]),
                caption=PROMPTS[i//SAMPLES_PER_PROMPT]) for i in range(len(pixels))]})
        wb.summary.update({'samples/prompts': list(PROMPTS),
            'samples/images_per_prompt': SAMPLES_PER_PROMPT, 'samples/preview_global_step': step,
            'samples/pixel_encoding': 'uint8_rgb_0_255'})
    model.train()


@torch.inference_mode()
@isolated_evaluation
def evaluate(model, aux, cfg, device, rank, world):
    from src.rqvae_metrics import OriginalRQVAEInception
    from src.training.cc3m_paper_metrics import (
        PROTOCOL, load_reference, load_clip, clip_scores, fid_from_moments)
    reference = load_reference(cfg)
    model.eval()
    net = OriginalRQVAEInception().to(device).eval()
    clip_net, clip_preprocess = load_clip(device)
    total = torch.zeros(2048, dtype=torch.float64, device=device)
    cross = torch.zeros(2048, 2048, dtype=torch.float64, device=device)
    count_clip = torch.zeros(2, dtype=torch.float64, device=device)
    data = dataset(cfg, 'validation')
    with nullcontext():
        for images, text, captions in loader(data, range(rank, len(data), world),
                cfg['eval_batch'], cfg['workers']):
            images, text = images.to(device, non_blocking=True), text.to(device, non_blocking=True)
            pixels = generate(model, aux, text, cfg)
            features, _ = net(pixels)
            features = features.double()
            total += features.sum(0)
            cross += features.T@features
            scores = clip_scores(pixels, captions, clip_net, clip_preprocess, device)
            count_clip[0] += len(text)
            count_clip[1] += scores.double().sum()
    for value in (total, cross, count_clip):
        dist.all_reduce(value)
    n = int(count_clip[0])
    assert n == cfg['validation_items']
    results = None
    if rank == 0:
        results = dict(fid=fid_from_moments(total, cross, n, reference),
                       clip_score=float(count_clip[1]/n), images=n,
                       protocol=PROTOCOL,
                       fid_reference_sha256=cfg['evaluation']['fid_reference_sha256'],
                       caption_subset='available CC3M validation captions; original-paper membership unverified',
                       clip_preprocessing='released uint8/PIL official CLIP transform; FP32 ViT-B/32')
    values = [results]
    dist.broadcast_object_list(values, src=0)
    del net, clip_net
    torch.cuda.empty_cache()
    model.train()
    return values[0]


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device))


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'], device)


class Checkpoints:
    def __init__(self, cfg, wb):
        self.cfg, self.wb = cfg, wb
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = None

    def wait(self):
        if self.future is not None:
            self.future.result()
            self.future = None

    def submit(self, payload, name='last.pt'):
        self.wait()
        frozen = cpu_snapshot(payload)
        self.future = self.pool.submit(self._write, frozen, name)

    def _write(self, payload, name):
        local = Path(self.cfg['local_checkpoints'])/name
        local.parent.mkdir(parents=True, exist_ok=True)
        artifact_upload = self.cfg.get('checkpoint_upload') == 'artifact'
        target = (local.parent/f'step-{payload["global_step"]:07d}-{time.time_ns()}.pt'
                  if artifact_upload else local)
        tmp = target.with_suffix('.tmp')
        torch.save(payload, tmp)
        tmp.replace(target)
        if artifact_upload:
            alias = local.with_suffix('.pending')
            alias.unlink(missing_ok=True)
            os.link(target, alias)
            alias.replace(local)
        with local.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        receipt = dict(global_step=payload['global_step'], bytes=local.stat().st_size,
                       sha256=digest, path=str(local), immutable_path=str(target), time=time.time())
        if self.wb is not None:
            if artifact_upload:
                import wandb
                artifact = wandb.Artifact(self.cfg['run_id']+'-checkpoint', type='model',
                    metadata=dict(global_step=payload['global_step'], epoch=payload['epoch'],
                                  sha256=digest, metrics=payload.get('metrics'), image_teacher=self.cfg.get('image_teacher')))
                artifact.add_file(str(target), name='last.pt', policy='immutable', skip_cache=True)
                aliases = [f'step-{payload["global_step"]}']
                aliases.append('latest' if name == 'last.pt' else name.removesuffix('.pt'))
                artifact = self.wb.log_artifact(artifact, aliases=aliases)
                artifact.wait()
                assert str(artifact.state) == 'COMMITTED'
                receipt.update(artifact=artifact.qualified_name, state='COMMITTED')
            else:
                self.wb.save(str(local), base_path=str(local.parent), policy='now')
        write_json(local.with_suffix('.json'), receipt)
        if artifact_upload and receipt.get('state') == 'COMMITTED':
            # Hardlinked last/best aliases preserve their payloads when cold names are pruned.
            for old in local.parent.glob('step-*.pt'):
                if old != target:
                    old.unlink()
        print(json.dumps(dict(checkpoint=str(local), global_step=payload['global_step'],
                              bytes=local.stat().st_size)), flush=True)

    def close(self):
        self.wait()
        self.pool.shutdown()


def benchmark(cfg):
    torch.backends.cudnn.benchmark = execution_options(cfg).get('cudnn_benchmark', False)
    torch.backends.cudnn.benchmark_limit = execution_options(cfg).get('cudnn_benchmark_limit', 10)
    device = torch.device('cuda', 0)
    seed_all(cfg['seed'])
    model, aux = build_model(cfg, device), build_aux(cfg, device)
    opt = optimizer_for(model, cfg)
    data = dataset(cfg, 'validation')
    rows = [data[i] for i in range(8)]
    pixels = torch.stack([r[0] for r in rows]).to(device)
    text = torch.stack([r[1] for r in rows]).to(device)
    trials = []
    execution = cfg.get('execution', {})
    candidates = execution.get('benchmark_batches', [32, 64, 128])
    memory_fraction = execution.get('benchmark_memory_fraction')
    memory_limit = (65*2**30 if memory_fraction is None else
        float(memory_fraction)*torch.cuda.get_device_properties(device).total_memory)
    if not 0 < memory_limit < torch.cuda.get_device_properties(device).total_memory:
        raise ValueError('Benchmark memory limit must leave space for DDP')
    for batch in candidates:
        if (not isinstance(batch, int) or batch < 1 or
                cfg['global_batch'] % (cfg['world_size']*batch)):
            raise ValueError('Benchmark batches must divide the global distributed update')
    with native_bf16_autocast(), sdpa_attention(model,
            **attention_options(cfg)):
        for batch in candidates:
            images = pixels.repeat(math.ceil(batch/8), 1, 1, 1)[:batch]
            ids = text.repeat(math.ceil(batch/8), 1)[:batch]
            torch.cuda.reset_peak_memory_stats()
            times = []
            try:
                for iteration in range(3):
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    loss, _ = objective(model, aux, images, ids, cfg)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    opt.step()
                    torch.cuda.synchronize()
                    if iteration:
                        times.append(time.monotonic()-start)
                row = dict(batch=batch, seconds=sum(times)/len(times),
                    peak_gib=torch.cuda.max_memory_allocated()/2**30, loss=float(loss.detach()),
                    gradient_norm=float(norm), passed=True)
            except torch.cuda.OutOfMemoryError:
                row = dict(batch=batch, passed=False, error='CUDA out of memory')
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            trials.append(row)
            print(json.dumps(row), flush=True)
            if not row['passed']:
                break
        successful = [x for x in trials if x['passed'] and x['peak_gib']*2**30 < memory_limit]
        if not successful:
            raise RuntimeError('No benchmark batch leaves room for DDP and evaluation')
        selected = max(successful, key=lambda x: x['batch']/x['seconds'])
        # Keep the already configured 2048-image global update exactly.
        cfg['microbatch'] = selected['batch']
        cfg['accumulation'] = cfg['global_batch']//(cfg['world_size']*selected['batch'])
        write_json(local_run_root(cfg)/'benchmark.json', dict(trials=trials, selected=selected,
            parameter_count=sum(p.numel() for p in model.parameters()),
            gpu=torch.cuda.get_device_name(0), memory_limit_gib=memory_limit/2**30,
            benchmark_weights_discarded=True))
    return cfg


def run(cfg, smoke=False):
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    overrides = os.environ.get('LASER_CC3M_EXECUTION_OVERRIDES')
    if overrides:
        configure_execution(json.loads(Path(overrides).read_text()))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = execution_options(cfg).get('cudnn_benchmark', False)
    torch.backends.cudnn.benchmark_limit = execution_options(cfg).get('cudnn_benchmark_limit', 10)
    from src.training.cc3m_paper_metrics import load_reference
    load_reference(cfg)  # Fail before GPU setup if the reference or protocol is wrong.
    rank, local, world = [int(os.environ[x]) for x in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=45))
    group = dist.new_group(backend='gloo', timeout=timedelta(minutes=45))
    assert world == cfg['world_size']
    assert cfg['microbatch']*cfg['accumulation']*world == cfg['global_batch']
    output = local_run_root(cfg)/('preflight' if smoke else 'train')
    output.mkdir(parents=True, exist_ok=True)
    seed_all(cfg['seed'])
    aux = build_aux(cfg, device)
    # Loading frozen assets must not select the fresh Transformer's weights.
    seed_all(cfg['seed'])
    prior = build_model(cfg, device)
    opt = optimizer_for(prior, cfg)
    updates = cfg['train_items']//cfg['global_batch']
    from src.training.cc3m_recovery import schedule_offset, load_best_weights, verify_baseline
    step_offset = 0 if smoke else schedule_offset(cfg, updates)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
        T_max=updates*cfg['epochs']-step_offset, eta_min=0.)
    step, start_epoch, cursor, best = 0, 0, 0, dict(fid=float('inf'), clip_score=-float('inf'))
    restored = None
    if not smoke:
        for path in [Path(cfg['local_checkpoints'])/'last.pt', Path(cfg['output'])/'checkpoints/last.pt']:
            if path.is_file():
                restored = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
                assert restored['config'] == cfg, 'Refusing a changed training config on resume'
                prior.load_state_dict(restored['model'], strict=True)
                opt.load_state_dict(restored['optimizer'])
                scheduler.load_state_dict(restored['scheduler'])
                step, start_epoch, cursor, best = [restored[k] for k in
                    ('global_step', 'epoch', 'next_update', 'best')]
                assert step == start_epoch*updates+cursor
                assert scheduler.last_epoch == step-step_offset
                break
        if restored is None and cfg.get('warm_start'):
            initial = load_best_weights(prior, cfg, updates)
            step, start_epoch, best = [initial[k] for k in ('global_step', 'epoch', 'best')]
    model = DDP(prior, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
    rng_options = cfg.get('training_rng', dict(policy=POLICY, base_seed=cfg['seed']))
    assert rng_options['policy'] == POLICY
    rng_metadata = configure_training_rng(rng_options['base_seed'], rank, resuming=restored is not None)
    if restored is not None:
        restore_rng(restored['rng_by_rank'][rank], device)
        del restored
    else:
        assert not opt.state and step == step_offset and scheduler.last_epoch == 0
    startup = dict(rank=rank, global_step=step, fresh_optimizer=not bool(opt.state),
        stage2_checkpoint_loaded=bool(step), execution_runtime=execution_options(cfg), **rng_metadata, **audit_training_rng(device))
    startup_by_rank = [None]*world
    dist.all_gather_object(startup_by_rank, startup, group=group)
    assert len({s['rng_state_sha256'] for s in startup_by_rank}) == world, 'Rank RNG states collide'
    assert len({s['dropout_probe_sha256'] for s in startup_by_rank}) == world, 'Rank dropout streams collide'
    write_json(output/f'startup-rank{rank}.json', startup)
    if rank == 0:
        write_json(output/'rng-verification.json', dict(passed=True, ranks=startup_by_rank,
            independent_cuda_states=True, independent_dropout_masks=True))
    training_rng = rng_state(device)
    wb, store = None, None
    mirror = None
    if rank == 0 and not smoke:
        mappings = [(local_run_root(cfg), Path(cfg['output']))]
        if os.environ.get('LASER_CC3M_CHECKPOINT_MIRROR', '1') == '1':
            mappings.append((Path(cfg['local_checkpoints']), Path(cfg['output'])/'checkpoints'))
        mirror = WorkspaceMirror(mappings).start()
        import wandb
        wb = wandb.init(entity=cfg['wandb_entity'], project='laser', id=cfg['run_id'],
            name=cfg['name'], resume='allow', config=cfg, dir=cfg['wandb_dir'],
            settings=wandb.Settings(disable_git=True))
        wb.config.update({'execution_runtime': execution_options(cfg)}, allow_val_change=True)
        if cfg.get('tokenizer_backend') == 'native_beam_pair':
            wb.use_artifact(cfg['tokenizer_recipe']['paired_tokenizer_artifact'])
        if cfg.get('tokenizer_backend') == 'native_vector':
            wb.use_artifact(cfg['tokenizer_recipe']['source_artifact'])
        if cfg.get('image_teacher') == 'beam_aligned':
            wb.use_artifact(cfg['beam_teacher']['source_artifact'])
            wb.summary.update({'teacher/kind': 'beam16 aligned conditional soft posterior',
                'teacher/beam_width': 16, 'teacher/sequence_temperature': cfg['beam_teacher']['sequence_temperature'],
                'teacher/native_greedy_teacher_used': False})
        wb.define_metric('train/global_step')
        wb.define_metric('train/*', step_metric='train/global_step')
        wb.define_metric('eval/*', step_metric='train/global_step')
        wb.summary.update(dict(parameter_count=sum(p.numel() for p in prior.parameters()),
            updates_per_epoch=updates, total_updates=updates*cfg['epochs'],
            initial_global_step=step_offset, planned_new_updates=updates*cfg['epochs']-step_offset,
            pipeline_status='training', upstream=rq_upstream.provenance(), **rng_metadata))
        for path in (local_run_root(cfg)/'reproducibility').glob('*'):
            if path.is_file():
                wb.save(str(path), base_path=str(local_run_root(cfg)), policy='now')
        write_json(local_run_root(cfg)/'wandb.json', dict(url=wb.url, run_id=wb.id))
        store = Checkpoints(cfg, wb)
        if (step and cfg.get('checkpoint_upload') != 'artifact'
                and (Path(cfg['local_checkpoints'])/'last.pt').exists()):
            wb.save(str(Path(cfg['local_checkpoints'])/'last.pt'),
                    base_path=cfg['local_checkpoints'], policy='now')
    restore_rng(training_rng, device)
    requested_stop = False

    def stop(*_):
        nonlocal requested_stop
        requested_stop = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def checkpoint(epoch, next_update, name='last.pt', metrics=None):
        states = [None]*world if rank == 0 else None
        dist.gather_object(rng_state(device), states, dst=0, group=group)
        if rank == 0:
            payload = dict(model=prior.state_dict(), optimizer=opt.state_dict(),
                scheduler=scheduler.state_dict(), global_step=step, epoch=epoch,
                next_update=next_update, config=cfg, rng_by_rank=states, best=best,
                training_rng_metadata=rng_metadata, execution_runtime=execution_options(cfg))
            if metrics is not None:
                payload['metrics'] = metrics
            store.submit(payload, name)

    try:
        with native_bf16_autocast(), sdpa_attention(prior,
                **attention_options(cfg)):
            if not smoke and step == step_offset and cfg.get('evaluate_at_start', False):
                results = evaluate(prior, aux, cfg, device, rank, world)
                verify_baseline(results, cfg)
                if cfg.get('warm_start'):
                    best.update({key: results[key] for key in ('fid', 'clip_score')})
                if rank == 0:
                    wb.log({'train/global_step': step, 'eval/epoch': start_epoch,
                            **{'eval/'+k: v for k, v in results.items()}})
                    write_json(output/f'eval-epoch-{start_epoch:03d}.json', results)
                    write_json(output/'baseline-verification.json', dict(passed=True, **results))
                if cfg.get('warm_start'):
                    checkpoint(start_epoch, 0, 'best-fid.pt', results)
                    checkpoint(start_epoch, 0, 'best-clip_score.pt', results)
                    checkpoint(start_epoch, 0)
            if step and not smoke:
                if rank == 0:
                    preview(prior, aux, cfg, device, wb, step, output)
                dist.barrier()
            for epoch in range(start_epoch, cfg['epochs']):
                data = dataset(cfg, 'train', epoch)
                order = epoch_order(data.specs, cfg['seed'], epoch, cfg['shuffle_group_shards'])
                assert len(order) == len(data)
                offset = cursor if epoch == start_epoch else 0
                indices = order[offset*cfg['global_batch']:updates*cfg['global_batch']][rank::world].tolist()
                if smoke:
                    indices = indices[:2*cfg['microbatch']*cfg['accumulation']]
                batches = iter(loader(data, indices, cfg['microbatch'], cfg['workers']))
                max_updates = min(updates, offset+2) if smoke else updates
                for update in range(offset, max_updates):
                    start = time.monotonic()
                    opt.zero_grad(set_to_none=True)
                    losses = torch.zeros(3, device=device)
                    for micro in range(cfg['accumulation']):
                        images, text, _ = next(batches)
                        images, text = images.to(device, non_blocking=True), text.to(device, non_blocking=True)
                        sync = micro+1 == cfg['accumulation']
                        with nullcontext() if sync else model.no_sync():
                            loss, components = objective(model, aux, images, text, cfg)
                            if not torch.isfinite(loss):
                                raise FloatingPointError('Nonfinite CC3M loss')
                            (loss/cfg['accumulation']).backward()
                            losses += torch.cat((loss.detach()[None], components))/cfg['accumulation']
                    norm = torch.nn.utils.clip_grad_norm_(prior.parameters(), cfg['grad_clip'], error_if_nonfinite=True)
                    opt.step()
                    scheduler.step()
                    step += 1
                    dist.all_reduce(losses)
                    losses /= world
                    stop_flag = torch.tensor(int(requested_stop), device=device)
                    dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                    metrics = {'train/global_step': step, 'train/epoch': epoch+(update+1)/updates,
                        'train/loss': float(losses[0]), 'train/image_loss': float(losses[1]),
                        'train/text_loss': float(losses[2]), 'train/gradient_norm': float(norm),
                        'train/lr': opt.param_groups[0]['lr'],
                        'train/images_per_second': cfg['global_batch']/(time.monotonic()-start),
                        'train/peak_gib': torch.cuda.max_memory_allocated()/2**30,
                        'rank': rank, 'pid': os.getpid(), 'time': time.time()}
                    write_json(output/f'progress-rank{rank}.json', metrics)
                    if rank == 0:
                        print(json.dumps(metrics), flush=True)
                        if wb is not None:
                            wb.log(metrics)
                        if store is not None and store.future is not None and store.future.done():
                            store.wait()
                    if smoke:
                        # Verify DDP parameter agreement after every complete optimizer update.
                        probe = torch.cat([p.flatten()[:4].detach().float() for p in prior.parameters()])
                        minimum, maximum = probe.clone(), probe.clone()
                        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
                        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
                        assert torch.equal(minimum, maximum), 'DDP replicas diverged'
                    else:
                        checkpoint_every = execution_options(cfg).get('checkpoint_every', cfg['save_every'])
                        if step == step_offset+1 or step % checkpoint_every == 0 or bool(stop_flag):
                            checkpoint(epoch, update+1)
                        if step == 1 or step % cfg['preview_every'] == 0:
                            if rank == 0:
                                preview(prior, aux, cfg, device, wb, step, output)
                            dist.barrier()
                        if bool(stop_flag):
                            return
                if smoke:
                    # Real model/Adam state roundtrip, with all rank RNG states.
                    state = cpu_snapshot(dict(model=prior.state_dict(), optimizer=opt.state_dict(),
                        scheduler=scheduler.state_dict(), rng=rng_state(device)))
                    path = output/f'roundtrip-rank{rank}.pt'
                    torch.save(state, path)
                    saved = torch.load(path, weights_only=False, mmap=True, map_location='cpu')
                    prior.load_state_dict(saved['model'], strict=True)
                    opt.load_state_dict(saved['optimizer'])
                    assert all(int(s['step']) == step for s in opt.state.values())
                    restore_rng(saved['rng'], device)
                    path.unlink()
                    del state, saved
                    if rank == 0:
                        before = rng_state(device)
                        preview(prior, aux, cfg, device, None, step, output)
                        after = rng_state(device)
                        assert before['python'] == after['python']
                        assert np.array_equal(before['numpy'][1], after['numpy'][1])
                        assert torch.equal(before['torch'], after['torch'])
                        assert torch.equal(before['cuda'], after['cuda'])
                        write_json(output/'passed.json', dict(passed=True, world_size=world,
                            steps=step, global_batch=cfg['global_batch'], finite_gradients=True,
                            replicas_equal=True, model_optimizer_roundtrip=True, conditioned_generation=True,
                            independent_rank_rng=True, preview_preserves_training_rng=True,
                            preflight_weights_discarded=True))
                    dist.barrier()
                    return
                # Preserve the completed epoch before an expensive generation pass.
                checkpoint(epoch, updates)
                results = evaluate(prior, aux, cfg, device, rank, world)
                if rank == 0:
                    wb.log({'train/global_step': step, 'eval/epoch': epoch+1,
                            **{'eval/'+k: v for k, v in results.items()}})
                    write_json(output/f'eval-epoch-{epoch+1:03d}.json', results)
                    improved = dict(fid=results['fid'] < best['fid'],
                                    clip_score=results['clip_score'] > best['clip_score'])
                    for metric, better in improved.items():
                        if better:
                            best[metric] = results[metric]
                # Synchronize best metrics before recording the full recovery state.
                values = [(best, improved) if rank == 0 else None]
                dist.broadcast_object_list(values, src=0, group=group)
                best, improved = values[0]
                for metric, better in improved.items():
                    if better:
                        checkpoint(epoch+1, 0, f'best-{metric}.pt', results)
                checkpoint(epoch+1, 0)
                dist.barrier()
                cursor = 0
            if rank == 0:
                wb.summary['pipeline_status'] = 'completed'
                write_json(local_run_root(cfg)/'completed.json', dict(epochs=cfg['epochs'], global_step=step))
    finally:
        if store is not None:
            store.close()
        if mirror is not None:
            mirror.close()
        # Release other ranks and unused GPU workspace before network draining.
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        if wb is not None:
            wb.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    overrides = os.environ.get('LASER_CC3M_EXECUTION_OVERRIDES')
    configure_execution(json.loads(Path(overrides).read_text()) if overrides else {})
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.benchmark:
        cfg = benchmark(cfg)
        OmegaConf.save(OmegaConf.create(cfg), args.config)
    else:
        run(cfg, args.smoke)


if __name__ == '__main__':
    main()
