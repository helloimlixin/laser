#!/usr/bin/env python3
"""CC3M reproduction using the frozen, successful FFHQ compound implementation.

The launcher supplies CC3M_BASE/source, an immutable copy of the August CC3M
extension of ffhqcmp0804205803. All model, target, and sampling operations use
that implementation; this driver adds k=4 caches and durable online selection.
"""
from __future__ import annotations

import argparse
import bisect
from contextlib import nullcontext
from datetime import timedelta
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import tarfile
import time

BASE = Path(os.environ['CC3M_BASE'])
SOURCE = BASE / 'source'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'scripts'))
import train_official_rqtransformer_laser_stage2 as ref
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from rqvae.txtimg_datasets.transforms import create_transforms
from rqvae.txtimg_datasets.tokenizers import create_tokenizer
from omegaconf import OmegaConf

TOKENIZER_SHA = 'dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'
RUN_ID = 'cc3m-imagenet421-compound-650m-100ep-20260919'
SOURCE_RUN = 'helloimlixin-rutgers/laser/cc3m-imagenet421-rq32k-650m-20260915'
DATA = Path('/scratch/xl598/Projects/data/cc3m')
STOP = False


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def stop_handler(*_):
    global STOP
    STOP = True


def stopping(device=None):
    flag = STOP or time.time() >= float(os.environ.get('CC3M_STOP_TIME', 'inf'))
    if device is not None and dist.is_initialized():
        value = torch.tensor(int(flag), device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        flag = bool(value.item())
    return flag


def setup():
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(hours=12))
    torch.set_num_threads(2)
    torch.manual_seed(1000 + rank)
    random.seed(1000 + rank)
    np.random.seed(1000 + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert 'A100' in torch.cuda.get_device_name() or 'L40S' in torch.cuda.get_device_name()
    print(json.dumps(dict(rank=rank, host=os.uname().nodename,
                         gpu=torch.cuda.get_device_name(), world=world)), flush=True)
    return rank, world, device


def barrier():
    if dist.is_initialized():
        dist.barrier()


def tokenizer(dropout=0.0):
    tok = create_tokenizer('bpe16k_huggingface', lowercase=True, dropout=dropout)
    tok.add_special_tokens(['[PAD]'])
    assert tok.token_to_id('[PAD]') == 0
    tok.enable_padding(length=32, pad_id=0)
    tok.enable_truncation(max_length=32)
    return tok


def image_transform(train):
    return create_transforms(OmegaConf.create(dict(transforms='dalle-vqvae', image_resolution=256)),
                             split='train' if train else 'val', is_eval=not train)


def auxiliary(device, scales=None):
    assert sha(BASE / 'tokenizer.pt') == TOKENIZER_SHA
    # This hash-pinned source checkpoint contains NumPy RNG/config objects in
    # addition to tensors. The legacy weights-only allowlist predates NumPy 2.
    def trusted_load(path):
        if Path(path).resolve() != (BASE / 'tokenizer.pt').resolve():
            raise ValueError('Only the verified source tokenizer may use this loader')
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    ref.load_stage1_checkpoint = trusted_load
    return ref.LaserAux(BASE / 'tokenizer.pt', 16384, 2048, 3.0,
                        coeff_scale=1.0, coeff_scales=scales,
                        soft_target_physical=False, clamp_coeffs=False,
                        attn_resolutions=(8,), sparsity_level=4).to(device).eval()


def model():
    return ref.build_model(16384 + 2048, 16384, compound=True,
                           coeff_vocab_size=2048, compound_micro_transformer_layers=2,
                           compound_depth_specific_coeff_heads=True, sparsity_level=4,
                           model_preset='cc3m-650m', num_condition_classes=16384,
                           condition_length=32)


class TarImages(Dataset):
    """Use explicit tar byte offsets; fail on corrupt data rather than adding blanks."""
    def __init__(self, path, training=False, limit=0):
        self.path = Path(path)
        grouped = {}
        with tarfile.open(path, 'r:') as archive:
            for member in archive:
                if member.isfile():
                    name = Path(member.name)
                    grouped.setdefault(str(name.with_suffix('')), {})[name.suffix.lower()] = (
                        member.offset_data, member.size)
        self.rows = []
        for key, row in sorted(grouped.items()):
            image = next((row[e] for e in ('.jpg', '.jpeg', '.png', '.webp') if e in row), None)
            if image is not None and '.txt' in row:
                self.rows.append((key, image, row['.txt']))
        if limit:
            self.rows = self.rows[:limit]
        self.training = training
        self.transform = image_transform(training)
        self.file = None

    def __len__(self):
        return len(self.rows)

    def read(self, offset):
        if self.file is None:
            self.file = self.path.open('rb')
        self.file.seek(offset[0])
        value = self.file.read(offset[1])
        if len(value) != offset[1]:
            raise IOError(f'Short read in {self.path}')
        return value

    def __getitem__(self, index):
        key, offset, text_offset = self.rows[index]
        image = Image.open(io.BytesIO(self.read(offset))).convert('RGB')
        caption = self.read(text_offset).decode('utf-8').strip()
        if self.training:
            views = []
            for view in range(2):
                seed = int(hashlib.sha256(f'{self.path.name}/{key}/{view}'.encode()).hexdigest()[:8], 16)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    views.append(self.transform(image))
            return torch.stack(views), caption
        return self.transform(image), caption


def cache_identity():
    return dict(format='cc3m_ffhq_compound_v1', tokenizer_sha256=TOKENIZER_SHA,
                image_shape=[8, 8, 4], image_views=2, text_views=100,
                bpe_dropout=0.1, transform='official_dalle_vqvae',
                coefficient_storage='physical_fp32', atom_storage='int16')


@torch.no_grad()
def build_cache(args):
    rank, world, device = setup()
    aux = auxiliary(device)
    cache = BASE / 'cache'
    cache.mkdir(exist_ok=True)
    info = json.loads((DATA / 'meta/_info.json').read_text())['splits']['train']
    shards = list(zip(info['filenames'], info['shard_lengths']))
    tok = tokenizer(0.1)
    for shard_index in range(rank, len(shards), world):
        if stopping():
            break
        filename, count = shards[shard_index]
        destination = cache / (Path(filename).stem + '.pt')
        receipt = destination.with_suffix('.json')
        if destination.is_file() and receipt.is_file():
            previous = json.loads(receipt.read_text())
            assert previous['identity'] == cache_identity() and previous['count'] == count
            assert sha(destination) == previous['sha256']
            continue
        data = TarImages(DATA / 'wds' / filename, training=True)
        assert len(data) == count, (filename, len(data), count)
        loader = DataLoader(data, batch_size=args.cache_batch, num_workers=2, pin_memory=True)
        atoms, coefficients, captions = [], [], []
        validation_error = 0.0
        for images, text in loader:
            images = images.to(device).flatten(0, 1)
            # Preserve FP32 tokenizer/OMP behavior and continuous coefficients.
            a, c = aux.encode_sparse_components(images)
            assert tuple(a.shape[1:]) == (8, 8, 4) and torch.isfinite(c).all()
            assert a.min() >= 0 and a.max() < 16384
            if not atoms:
                aa, cc = aux.encode_sparse_components(images[:2])
                assert torch.equal(aa, a[:2]), 'OMP cache support mismatch'
                validation_error = float((cc - c[:2]).abs().max())
                assert validation_error < 0.01
            atoms.append(a.cpu().short().reshape(-1, 2, 8, 8, 4))
            coefficients.append(c.cpu().float().reshape(-1, 2, 8, 8, 4))
            captions.extend(text)
        text_views = torch.empty(count, 100, 32, dtype=torch.int16)
        for view in range(100):
            text_views[:, view] = torch.tensor([x.ids for x in tok.encode_batch(captions)], dtype=torch.int16)
        payload = dict(atoms=torch.cat(atoms), coeffs=torch.cat(coefficients),
                       text_tokens=text_views, captions=captions, identity=cache_identity())
        assert payload['atoms'].shape == (count, 2, 8, 8, 4)
        assert int(text_views.min()) >= 0 and int(text_views.max()) < 16384
        ref.atomic_torch_save(payload, destination)
        write_json(receipt, dict(identity=cache_identity(), count=count, sha256=sha(destination),
                                atom_exact_fraction=1.0, max_coeff_error=validation_error))
        print(json.dumps(dict(phase='cache', rank=rank, shard=shard_index,
                              count=count, completed_shards=len(list(cache.glob('*.json'))))), flush=True)
        del payload, atoms, coefficients, text_views
    barrier()
    if rank == 0:
        receipts = [cache / (Path(name).stem + '.json') for name, _ in shards]
        if all(p.is_file() for p in receipts):
            # Fit the same normalized coefficient range policy as successful FFHQ.
            calibration = []
            for i in np.linspace(0, len(shards)-1, 16, dtype=int):
                part = torch.load(cache / (Path(shards[i][0]).stem + '.pt'), weights_only=True, mmap=True)
                calibration.append(part['coeffs'][:256].reshape(-1, 4))
            values = torch.cat(calibration).abs()
            scales = (torch.quantile(values, 0.995, dim=0) / 3.0).tolist()
            assert all(math.isfinite(x) and x > 0 for x in scales)
            write_json(cache / 'ready.json', dict(identity=cache_identity(), passed=True,
                       num_images=sum(info['shard_lengths']), shards=info['filenames'],
                       lengths=info['shard_lengths'], coeff_scales=scales,
                       scale_quantile=0.995, normalized_coefficient_max=3.0,
                       target_temperature=0.5, target_space='normalized_ffhq'))
        else:
            write_json(BASE / 'requeue.json', dict(phase='cache', reason='allocation deadline'))
    barrier()


class CachedPairs(Dataset):
    def __init__(self, root, epoch):
        self.root, self.epoch = Path(root), int(epoch)
        self.meta = json.loads((self.root / 'ready.json').read_text())
        assert self.meta['identity'] == cache_identity() and self.meta['passed']
        self.ends = np.cumsum(self.meta['lengths']).tolist()
        self.parts = {}
        self.scales = torch.tensor(self.meta['coeff_scales']).reshape(1, 1, 4)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        shard = bisect.bisect_right(self.ends, index)
        row = index - (self.ends[shard - 1] if shard else 0)
        if shard not in self.parts:
            path = self.root / (Path(self.meta['shards'][shard]).stem + '.pt')
            self.parts[shard] = torch.load(path, weights_only=True, mmap=True)
        part = self.parts[shard]
        view = (self.epoch + index) % 2
        return (part['atoms'][row, view].long(),
                (part['coeffs'][row, view].float() / self.scales).clamp(-3, 3),
                part['text_tokens'][row, self.epoch % 100].long())


def config(args, scales):
    return dict(source_run=SOURCE_RUN, compound_reference='helloimlixin-rutgers/laser/ffhqcmp0804205803',
                tokenizer_sha256=TOKENIZER_SHA, latent_shape=[8, 8, 4], epochs=100,
                architecture='official-cc3m-650m-plus-ffhq-micro2', num_atoms=16384,
                coeff_vocab_size=2048, coeff_max=3., coeff_scales=scales,
                coefficient_target_temperature=0.5, coefficient_target_space='normalized_ffhq',
                atom_loss_weight=1.5, geometry_loss_weight=0.05, geometry_start_epoch=2,
                geometry_warmup_epochs=3, geometry_top_k=4, text_weight=0.1, image_weight=0.9,
                lr=0.0005, betas=[0.9, 0.95], weight_decay=0.0001, max_grad_norm=1.0,
                lr_schedule='cosine_100_epochs_no_warmup', global_batch=2048,
                local_batch=args.batch, world_size=int(os.environ.get('WORLD_SIZE', 1)),
                text_context=32, text_vocab=16384, bpe_dropout=0.1, text_views=100,
                image_views=2, precision='BF16 training; FP32 cache, targets and losses',
                evaluation='all_13443_CC3M_validation_pairs_FID_and_CLIP_ViT_B32_cosine',
                evaluation_every_epoch=1, save_every_steps=500, checkpoint_mode='online_artifacts',
                sample_atom_top_k=16384, sample_atom_top_p=0.7,
                sample_coeff_top_k=2048, sample_coeff_top_p=0.7)


def loss_for(net, aux, batch, device, progress):
    atoms, coeffs, text = [x.to(device, non_blocking=True) for x in batch]
    with torch.no_grad():
        ids, probs = aux.compound_coeff_ids(coeffs, stochastic=True, temp=0.5)
        physical = aux.physical_contributions(atoms, coeffs)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output, text_logits = net(atoms * 2048 + ids, model_aux=aux, cond=text, amp=False)
        image_loss, detail = ref.compound_objective(
            output['atom_logits'], output['coeff_logits'], None, atoms, probs, physical,
            atom_weight=1.5, geometry_weight=ref.scheduled_geometry_weight(0.05, progress, 2, 3),
            accumulation=1, distribution_geometry=True, geometry_dictionary=aux.dictionary,
            geometry_coeff_bins=aux.coeff_bins, geometry_coeff_scales=aux.coeff_scales,
            geometry_top_k=4)
        text_loss = F.cross_entropy(text_logits.float().reshape(-1, 16384), text[:, 1:].reshape(-1))
        loss = 0.9 * image_loss + 0.1 * text_loss
    return loss, dict(image_loss=float(image_loss.detach()), text_loss=float(text_loss.detach()),
                      atom_nll=float(detail['atom_nll'].mean().detach()))


def init_wandb(cfg):
    import wandb
    wb = wandb.init(entity='helloimlixin-rutgers', project='laser', id=RUN_ID,
                    name=RUN_ID, mode='online', resume='allow', config=cfg)
    assert wb.settings.mode == 'online'
    wb.config.update(cfg, allow_val_change=True)
    return wb


def checkpoint(net, optimizer, scheduler, step, steps_per_epoch, cfg, local, wb, best, metrics=None):
    rank = dist.get_rank() if dist.is_initialized() else 0
    rng = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
               numpy=np.random.get_state(), python=random.getstate())
    all_rng = [None] * dist.get_world_size() if dist.is_initialized() else [rng]
    if dist.is_initialized():
        dist.all_gather_object(all_rng, rng)
    if rank == 0:
        model_state = net.state_dict()
        aliases = ['latest', 'last', f'step-{step}']
        improved_fid = improved_clip = False
        if metrics is not None:
            best['evaluated_step'] = step
            if metrics['fid'] < best['fid']:
                best['fid'] = metrics['fid']
                improved_fid = True
                aliases.append('best-fid')
            if metrics['clip_score'] > best['clip_score']:
                best['clip_score'] = metrics['clip_score']
                improved_clip = True
                aliases.append('best-clip')
            aliases.append(f'epoch-{step//steps_per_epoch}')
        ref.atomic_torch_save(dict(state_dict=model_state, optimizer=optimizer.state_dict(),
                                  scheduler=scheduler.state_dict(), step=step, steps_per_epoch=steps_per_epoch,
                                  epoch=step/steps_per_epoch, config=cfg, best=best, rng=all_rng,
                                  metrics=metrics), local/'last.pt')
        if improved_fid:
            ref.snapshot_checkpoint(local/'last.pt', local/'best-fid.pt')
        if improved_clip:
            ref.snapshot_checkpoint(local/'last.pt', local/'best-clip.pt')
        paths = [local/'last.pt'] + [local/name for name in ('best-fid.pt', 'best-clip.pt') if (local/name).exists()]
        ref.upload_checkpoints(wb, paths, artifact_name=RUN_ID+'-checkpoints', aliases=aliases,
                               metadata=dict(step=step, epoch=step/steps_per_epoch,
                                             best_fid=best['fid'] if math.isfinite(best['fid']) else None,
                                             best_clip=best['clip_score'] if math.isfinite(best['clip_score']) else None,
                                             tokenizer_sha256=TOKENIZER_SHA))
        # Only publish the cursor after W&B acknowledges the immutable artifact.
        write_json(BASE/'last-upload.json', dict(step=step, epoch=step/steps_per_epoch,
                                                aliases=aliases, committed=True, updated=time.time()))
    barrier()


@torch.no_grad()
def preview(net, aux, device, step, wb):
    import wandb
    prompts = ['A red car parked on a city street', 'A small house beside a lake',
               'A cat sitting on a wooden chair', 'A dog running through a green field',
               'A bowl of fruit on a kitchen table', 'A snow covered mountain',
               'A boat sailing on the ocean', 'A person riding a bicycle']
    images = []
    net.eval()
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(9011)
        for prompt in prompts:
            condition = ref.encode_cc3m_prompts([prompt]*8).to(device)
            atoms, ids = net.sample_compound(8, aux, cond=condition, atom_top_k=16384,
                                             atom_top_p=0.7, coeff_top_k=2048, coeff_top_p=0.7)
            images.append(((aux.decode_compound(atoms, ids).float().cpu()+1)*0.5).clamp(0,1))
    target = BASE/'train'/f'prompt-grid-step{step:07d}.jpg'
    target.parent.mkdir(parents=True, exist_ok=True)
    ref.save_paper_style_text_grid(torch.cat(images), prompts, target)
    wb.log({'preview/samples':wandb.Image(str(target)), 'preview/step':step})
    ref.release_cuda_memory(net)


@torch.no_grad()
def evaluate(net, aux, device, rank, world, limit=0):
    validation = ref.CC3MValidationDataset(DATA, transform=image_transform(False))
    assert len(validation) == 13443
    count = limit or len(validation)
    loader = DataLoader(validation, batch_size=8, sampler=range(rank, count, world), num_workers=2)
    torch.manual_seed(71000 + rank)
    fid, clip_score = ref.evaluate_text_generation_metrics(
        net, aux, loader, count, Path(os.environ['CC3M_LOCAL'])/'clip',
        atom_top_k=16384, atom_top_p=0.7, coeff_top_k=2048, coeff_top_p=0.7)
    assert math.isfinite(fid) and math.isfinite(clip_score)
    return dict(fid=fid, clip_score=clip_score, pairs=count)


def train(args):
    rank, world, device = setup()
    local = Path(os.environ['CC3M_LOCAL'])/'checkpoints'
    local.mkdir(parents=True, exist_ok=True)
    data = CachedPairs(Path(os.environ['CC3M_LOCAL'])/'cache', 0)
    cfg = config(args, data.meta['coeff_scales'])
    accumulation = 2048 // (world * args.batch)
    assert accumulation * world * args.batch == 2048
    aux = auxiliary(device, data.meta['coeff_scales'])
    net = model().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0005, betas=(0.9, 0.95), weight_decay=0.0001)
    steps_per_epoch = len(data) // 2048
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100*steps_per_epoch)
    step, best = 0, dict(fid=float('inf'), clip_score=-float('inf'))
    selection = json.loads((BASE/'selection.json').read_text())
    if selection['artifact']:
        saved = torch.load(local/'last.pt', weights_only=False, mmap=True, map_location='cpu')
        for key, value in cfg.items():
            if key not in ('world_size', 'local_batch') and saved['config'][key] != value:
                raise ValueError(f'Resume config changed: {key}')
        net.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        step, best = saved['step'], saved['best']
        assert saved['steps_per_epoch'] == steps_per_epoch and scheduler.last_epoch == step
        if len(saved['rng']) == world:
            rng = saved['rng'][rank]
            torch.set_rng_state(rng['torch']); torch.cuda.set_rng_state(rng['cuda'])
            np.random.set_state(rng['numpy']); random.setstate(rng['python'])
        del saved
    cfg['parameters'] = sum(p.numel() for p in net.parameters())
    wb = init_wandb(cfg) if rank == 0 else None
    if rank == 0:
        write_json(BASE/'training-config.json', cfg)
        wb.summary['training_status'] = 'running'
        wb.summary['phase'] = 'training'
        wb.summary['slurm/job_id'] = int(os.environ['SLURM_JOB_ID'])
    ddp = DDP(net, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 else net
    start = time.monotonic()
    start_step = step

    def finish_epoch():
        # A checkpoint at an epoch boundary may have been committed just before
        # a time-limit handoff. Complete its pending evaluation before training.
        estimate = json.loads((BASE/'gpu-preflight.json').read_text()).get('evaluation_budget_seconds', 7200)
        if time.time()+estimate >= float(os.environ.get('CC3M_STOP_TIME', 'inf')):
            if rank == 0:
                write_json(BASE/'requeue.json', dict(phase='evaluation', step=step, reason='allocation deadline'))
                wb.finish()
            return False
        with torch.random.fork_rng(devices=[device.index]):
            metrics = evaluate(net, aux, device, rank, world)
        if rank == 0:
            wb.log({'val/'+k:v for k,v in metrics.items()} | {'val/epoch':step//steps_per_epoch, 'val/step':step})
        checkpoint(net, optimizer, scheduler, step, steps_per_epoch, cfg, local, wb, best, metrics)
        if rank == 0:
            write_json(BASE/f'evaluation-epoch-{step//steps_per_epoch:03d}.json', metrics)
            preview(net, aux, device, step, wb)
        barrier()
        return True

    if step and step % steps_per_epoch == 0 and best.get('evaluated_step', 0) < step:
        if not finish_epoch():
            return
    for epoch in range(step//steps_per_epoch, 100):
        data.epoch = epoch
        offset = step % steps_per_epoch
        permutation = torch.randperm(len(data), generator=torch.Generator().manual_seed(epoch))[:steps_per_epoch*2048]
        indices = permutation.reshape(-1, accumulation, world, args.batch)[offset:, :, rank].reshape(-1).tolist()
        loader = DataLoader(data, batch_size=args.batch, sampler=indices, num_workers=2,
                            pin_memory=True, persistent_workers=False)
        ddp.train()
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(loader):
            sync = (micro+1) % accumulation == 0
            with ddp.no_sync() if isinstance(ddp, DDP) and not sync else nullcontext():
                loss, detail = loss_for(ddp, aux, batch, device, step/steps_per_epoch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                (loss/accumulation).backward()
            if not sync:
                continue
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and (step % 10 == 0 or step <= start_step+3):
                payload = {f'train/{k}':v for k,v in detail.items()}
                payload.update({'train/loss':float(loss.detach()), 'train/step':step,
                                'train/epoch':step/steps_per_epoch, 'train/lr':scheduler.get_last_lr()[0],
                                'train/grad_norm':float(norm),
                                'train/images_per_second':2048*(step-start_step)/max(time.monotonic()-start,1)})
                wb.log(payload)
                write_json(BASE/'status.json', dict(phase='training', step=step, epoch=step/steps_per_epoch,
                                                   loss=float(loss.detach()), updated=time.time()))
                print(json.dumps(payload), flush=True)
            stop = stopping(device) or (args.max_steps > 0 and step-start_step >= args.max_steps)
            if stop or step % 500 == 0:
                checkpoint(net, optimizer, scheduler, step, steps_per_epoch, cfg, local, wb, best)
            if stop:
                if rank == 0:
                    write_json(BASE/'requeue.json', dict(phase='train', step=step, reason='allocation deadline'))
                    wb.finish()
                return
        # Same epoch checkpoint is saved before and after evaluation, so a failed
        # evaluator never loses an epoch of completed optimizer work.
        checkpoint(net, optimizer, scheduler, step, steps_per_epoch, cfg, local, wb, best)
        if not finish_epoch():
            return
    if rank == 0:
        write_json(BASE/'complete.json', dict(epochs=100, step=step, best=best))
        wb.summary['training_status'] = 'completed_100_epochs'
        wb.finish()


def preflight(args):
    rank, world, device = setup()
    aux = auxiliary(device)
    data = TarImages(DATA/'wds/cc3m-train-0000.tar', training=True, limit=16)
    images, captions = next(iter(DataLoader(data, batch_size=16)))
    with torch.no_grad():
        atoms, coefficients = aux.encode_sparse_components(images[:,0].to(device))
        scales = (torch.quantile(coefficients.abs().reshape(-1,4), 0.995, dim=0)/3).clamp_min(1e-5)
        aux.coeff_scales.copy_(scales)
    net = model().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0005, betas=(0.9,0.95), weight_decay=0.0001)
    text = ref.encode_cc3m_prompts(captions)
    batch = (atoms, (coefficients/scales).clamp(-3,3), text)
    ddp = DDP(net, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True) if world>1 else net
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, detail = loss_for(ddp, aux, batch, device, 0.0)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        assert torch.isfinite(loss)
        print(json.dumps(dict(phase='preflight', rank=rank, step=step, loss=float(loss), grad_norm=float(norm))), flush=True)
    del optimizer, ddp
    gc.collect(); torch.cuda.empty_cache()
    evaluation_start = time.monotonic()
    metrics = evaluate(net, aux, device, rank, world, limit=max(16, world*2))
    if rank == 0:
        report = dict(passed=True, tokenizer_sha256=TOKENIZER_SHA, latent_shape=list(atoms.shape[1:]),
                      parameters=sum(p.numel() for p in net.parameters()), world_size=world,
                      local_batch=16, loss=float(loss), smoke_metrics=metrics,
                      evaluation_budget_seconds=max(3600, (time.monotonic()-evaluation_start)*13443/metrics['pairs']*1.3),
                      peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
        write_json(BASE/'gpu-preflight.json', report)
        wb = init_wandb(config(args, scales.tolist()))
        import wandb
        artifact = wandb.Artifact(RUN_ID+'-inputs', type='dataset', metadata=report)
        for path in [BASE/'tokenizer.pt', BASE/'source.tar.gz', BASE/'gpu-preflight.json', BASE/'runtime-sha256.txt']:
            artifact.add_file(str(path), name=path.name, policy='immutable', skip_cache=True)
        wb.log_artifact(artifact, aliases=['latest']).wait()
        wb.summary['preflight/passed'] = True
        wb.summary['preflight/gpu_passed'] = True
        wb.summary['training_status'] = 'building_token_cache'
        wb.summary['phase'] = 'building_token_cache'
        wb.finish()


def resolve():
    import wandb
    try:
        artifact = wandb.Api(timeout=120).artifact('helloimlixin-rutgers/laser/'+RUN_ID+'-checkpoints:latest')
    except wandb.errors.CommError as error:
        missing = any(s in str(error).lower() for s in ('not found', 'does not exist', 'unable to fetch artifact'))
        if not missing or (BASE/'last-upload.json').exists():
            raise
        write_json(BASE/'selection.json', dict(artifact=None))
        return
    write_json(BASE/'selection.json', dict(artifact=artifact.qualified_name, digest=artifact.digest))


def stage():
    selection = json.loads((BASE/'selection.json').read_text())
    if selection['artifact']:
        import wandb
        artifact = wandb.Api(timeout=120).artifact(selection['artifact'])
        assert artifact.digest == selection['digest']
        artifact.download(root=str(Path(os.environ['CC3M_LOCAL'])/'checkpoints'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['cache', 'train', 'preflight', 'resolve', 'stage'])
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--cache-batch', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=0)
    args = p.parse_args()
    if args.phase in ('resolve', 'stage'):
        globals()[args.phase]()
    else:
        dict(cache=build_cache, train=train, preflight=preflight)[args.phase](args)
