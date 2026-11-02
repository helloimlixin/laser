"""Train the depth-major LASER DCTransformer on a verified frozen token cache."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

from src.models.compound_dc_transformer import CompoundDCTransformer
from src.models.rqvae.modules import Decoder
from src.training.exact_global_batch import ExactGlobalBatchSampler
from src.training.rq_imagenet_runtime import cpu_snapshot, sha256, write_json


class FrozenImageNetAux(nn.Module):
    """Only the original dictionary, physical coefficient grid and image decoder."""
    def __init__(self, checkpoint, coeff_scales, coeff_vocab_size=2048):
        super().__init__()
        payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        state = payload['state_dict']
        self.register_buffer('dictionary', F.normalize(state['quantizer.dictionary'].float(), dim=0))
        self.register_buffer('coeff_bins', torch.linspace(-3., 3., coeff_vocab_size))
        self.register_buffer('coeff_scales', torch.tensor(coeff_scales, dtype=torch.float32))
        self.decoder = Decoder(double_z=False, z_channels=256, resolution=256,
            in_channels=3, out_ch=3, ch=128, ch_mult=[1, 1, 2, 2, 4, 4],
            num_res_blocks=2, attn_resolutions=[8], dropout=0.)
        self.post_quant_conv = nn.Conv2d(256, 256, 1)
        for name in ('decoder', 'post_quant_conv'):
            prefix = name + '.'
            getattr(self, name).load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                                if k.startswith(prefix)}, strict=True)
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def decode_compound(self, atoms, coefficient_ids):
        physical = self.coeff_bins[coefficient_ids.long()] * self.coeff_scales
        latent = (self.dictionary.T[atoms.long()] * physical[..., None]).sum(-2)
        return self.decoder(self.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous())).clamp(-1., 1.)


def learning_rate(step, *, lr, warmup, total):
    if step < warmup:
        return lr * (step + 1) / warmup
    return lr * .5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))


def loss_components(output, atoms, coefficients, num_atoms, coeff_vocab_size):
    atom = F.cross_entropy(output['atom_logits'].float().reshape(-1, num_atoms), atoms.reshape(-1))
    coefficient = F.cross_entropy(output['coeff_logits'].float().reshape(-1, coeff_vocab_size), coefficients.reshape(-1))
    return .5 * (atom + coefficient), atom, coefficient


def rng_state(device):
    return dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state(device),
                python=random.getstate(), numpy=np.random.get_state())


def restore_rng(state, device):
    torch.set_rng_state(state['torch_cpu']); torch.cuda.set_rng_state(state['torch_cuda'], device)
    random.setstate(state['python']); np.random.set_state(state['numpy'])


@torch.no_grad()
def preview(model, aux, cfg, step, rank, world, wb):
    model.eval()
    labels = torch.tensor(cfg.preview_classes, device='cuda').repeat_interleave(cfg.preview_per_class)
    local = torch.arange(rank, len(labels), world, device='cuda')
    images = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(cfg.seed + 100000 + rank)
        for indices in local.split(cfg.sample_batch_size):
            atoms, coefficients = model.sample_compound(len(indices), aux, cond=labels[indices], amp=True)
            images.append(((aux.decode_compound(atoms, coefficients).float() + 1) / 2).cpu())
    gathered = [None] * world
    dist.all_gather_object(gathered, (local.cpu(), torch.cat(images)))
    if rank == 0:
        arranged = torch.empty(len(labels), 3, 256, 256)
        for indices, values in gathered:
            arranged[indices] = values
        path = Path(cfg.output) / 'samples' / f'step-{step:07d}.png'
        path.parent.mkdir(exist_ok=True)
        save_image(arranged, path, nrow=cfg.preview_per_class)
        if wb:
            import wandb
            caption = 'Rows: ' + ', '.join(str(c) for c in cfg.preview_classes)
            wb.log({'train/global_step': step, 'samples/class_conditional': wandb.Image(str(path), caption=caption)})
    model.train()


@torch.no_grad()
def evaluate(model, aux, cfg, rank, world):
    from src.rqvae_metrics import DistributedOriginalRQVAEMetrics
    metric = DistributedOriginalRQVAEMetrics('cuda', compute_inception_score=True,
                                            reference_stats_path=cfg.fid_reference)
    count = cfg.fid_num_samples // world + (rank < cfg.fid_num_samples % world)
    model.eval()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(cfg.seed + 200000 + rank)
        for first in range(0, count, cfg.fid_batch_size):
            labels = (torch.arange(first, min(first + cfg.fid_batch_size, count), device='cuda') * world + rank) % cfg.model.num_classes
            atoms, coefficients = model.sample_compound(len(labels), aux, cond=labels, amp=True)
            for offset in range(0, len(labels), cfg.decode_batch_size):
                images = aux.decode_compound(atoms[offset:offset + cfg.decode_batch_size], coefficients[offset:offset + cfg.decode_batch_size])
                metric.update(((images.float() + 1) / 2).clamp(0, 1), real=False)
            if rank == 0:
                print(json.dumps(dict(phase='fid',generated_per_rank=min(first + cfg.fid_batch_size, count), total_per_rank=count)), flush=True)
        fid, inception, std = metric.compute(shuffle_for_inception=True)
    model.train()
    return dict(fid=fid, inception_score=inception, inception_score_std=std)


def run(config):
    cfg = config.options
    rank, world, local = [int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    torch.set_num_threads(cfg.cpu_threads)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=60), device_id=device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert cfg.global_batch % (world * cfg.microbatch) == 0
    accumulation = cfg.global_batch // (world * cfg.microbatch)
    out = Path(cfg.output); out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
    if rank == 0:
        assert sha256(cfg.checkpoint) == cfg.stage1_sha256, 'Stage 1 checkpoint hash mismatch'
    dist.barrier()
    data = torch.load(cfg.token_cache, map_location='cpu', mmap=True, weights_only=False)
    assert data['meta']['stage1_checkpoint_sha256'] == cfg.stage1_sha256
    assert list(data['meta']['coeff_scales']) == list(cfg.coeff_scales)
    assert data['atoms'].shape == (cfg.training_images, *cfg.model.block_size)
    dataset = TensorDataset(data['atoms'], data['coefficient_ids'], data['labels'])
    sampler = ExactGlobalBatchSampler(dataset, cfg.global_batch, world, rank, accumulation, cfg.seed)
    total_steps = sampler.steps_per_epoch * cfg.epochs
    assert 0 <= cfg.warmup_steps < total_steps
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    raw = CompoundDCTransformer(**OmegaConf.to_container(cfg.model)).to(device)
    aux = FrozenImageNetAux(cfg.checkpoint, cfg.coeff_scales, cfg.model.coeff_vocab_size).to(device)
    optimizer = torch.optim.Adam(raw.parameters(), lr=cfg.lr, betas=(.9, .999), eps=1e-8, fused=True)
    step = epoch = cursor = 0; best_fid = None
    state = None
    if cfg.resume_checkpoint:
        state = torch.load(cfg.resume_checkpoint, map_location='cpu', mmap=True, weights_only=False)
        assert state['config']['model'] == OmegaConf.to_container(cfg.model)
        assert state['config']['stage1_sha256'] == cfg.stage1_sha256
        assert state['config']['cache_sha256'] == cfg.cache_sha256
        assert state['world_size'] == world and state['accumulation'] == accumulation
        assert state['total_steps'] == total_steps
        raw.load_state_dict(state['state_dict'], strict=True); optimizer.load_state_dict(state['optimizer'])
        step, epoch, cursor = state['global_step'], state['epoch'], state['batch_idx']
        best_fid = state['best_fid']
        assert step == epoch * sampler.steps_per_epoch + cursor
    model = DDP(raw, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True,
                        generator=torch.Generator().manual_seed(cfg.seed + rank))
    torch.manual_seed(cfg.seed + rank); torch.cuda.manual_seed(cfg.seed + rank)
    if state:
        restore_rng(state['rng_state_by_rank'][rank], device)
        del state
    parameter_count = sum(p.numel() for p in raw.parameters())
    launch = dict(world_size=world, accumulation=accumulation, parameter_count=parameter_count,
                  stage1_frozen=not any(p.requires_grad for p in aux.parameters()),
                  training_from_scratch=not bool(cfg.resume_checkpoint), global_step=step,
                  training_images=len(dataset), classes=cfg.model.num_classes, total_steps=total_steps)
    write_json(out / f'start-rank{rank}.json', launch)
    wb = None; publisher = None; jobs = []
    if rank == 0:
        write_json(out / 'launch-config.json', dict(OmegaConf.to_container(cfg), **launch))
        if cfg.wandb_mode != 'disabled':
            import wandb
            wb = wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, id=cfg.run_id,
                name=cfg.wandb_name, resume='must' if cfg.resume_checkpoint else 'never',
                config=dict(OmegaConf.to_container(cfg), **launch), dir=str(out))
            wb.define_metric('train/global_step')
            for pattern in ['train/*', 'eval/*', 'samples/*']:
                wb.define_metric(pattern, step_metric='train/global_step')
            write_json(out / 'wandb.json', dict(id=wb.id, url=wb.url))
            publisher = ThreadPoolExecutor(max_workers=1)
    dist.barrier()

    def upload(path, saved_step):
        import wandb
        artifact = wandb.Artifact(cfg.run_id + '-checkpoint', type='model', metadata={'global_step': saved_step})
        artifact.add_file(str(path), name='last.pt')
        wb.log_artifact(artifact, aliases=['latest', f'step-{saved_step}']).wait()
        write_json(out / 'checkpoint-upload.json', dict(global_step=saved_step, checkpoint=str(path), uploaded=True))

    def snapshot(metrics=None):
        streams = [None] * world
        dist.all_gather_object(streams, rng_state(device))
        if rank == 0:
            payload = dict(state_dict=cpu_snapshot(raw.state_dict()), optimizer=cpu_snapshot(optimizer.state_dict()),
                global_step=step, epoch=epoch, batch_idx=cursor, rng_state_by_rank=streams,
                config=OmegaConf.to_container(cfg), world_size=world, accumulation=accumulation,
                total_steps=total_steps, scheduler=dict(completed_steps=step,
                    next_lr=learning_rate(step, lr=cfg.lr, warmup=cfg.warmup_steps, total=total_steps)),
                best_fid=best_fid, evaluation=metrics)
            local_path = Path(cfg.local_checkpoint_dir) / f'step-{step:07d}.pt'
            local_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, local_path); del payload
            path = checkpoints / local_path.name
            temporary = path.with_suffix('.partial'); shutil.copyfile(local_path, temporary); temporary.replace(path)
            # A small receipt points only to a completely written full checkpoint.
            write_json(checkpoints / 'latest.json', dict(checkpoint=str(path), global_step=step))
            if metrics and metrics['fid'] == best_fid:
                write_json(checkpoints / 'best-fid.json', dict(checkpoint=str(path), global_step=step, **metrics))
            if wb and cfg.upload_checkpoints:
                jobs.append(publisher.submit(upload, path, step))
            for job in jobs:
                if job.done(): job.result()
            # Keep recent recovery states, retained best, and any upload in progress.
            if all(job.done() for job in jobs):
                best_receipt = checkpoints / 'best-fid.json'
                best_path = json.loads(best_receipt.read_text())['checkpoint'] if best_receipt.exists() else None
                for stale in sorted(checkpoints.glob('step-*.pt'))[:-2]:
                    if str(stale) != best_path:
                        stale.unlink()
            for stale in local_path.parent.glob('step-*.pt'):
                if stale != local_path: stale.unlink()
            print(json.dumps(dict(phase='checkpoint', global_step=step, path=str(path))), flush=True)
        dist.barrier()

    optimizer.zero_grad(set_to_none=True)
    started = last = time.monotonic()
    stop = min(total_steps, step + cfg.max_steps) if cfg.max_steps else total_steps
    torch.cuda.reset_peak_memory_stats(device)
    try:
        while step < stop:
            if cursor == sampler.steps_per_epoch:
                epoch += 1; cursor = 0
            sampler.set_epoch(epoch); sampler.set_start_batch(cursor * accumulation)
            metrics = torch.zeros(3, device=device)
            for offset, (atoms, coefficients, labels) in enumerate(loader, start=cursor * accumulation):
                atoms, coefficients, labels = [x.to(device, dtype=torch.long, non_blocking=True)
                                               for x in (atoms, coefficients, labels)]
                final_micro = (offset + 1) % accumulation == 0
                with nullcontext() if final_micro else model.no_sync():
                    output = model(atoms * cfg.model.coeff_vocab_size + coefficients, model_aux=aux, cond=labels, amp=True)
                    loss, atom_nll, coefficient_nll = loss_components(output, atoms, coefficients,
                        cfg.model.num_atoms, cfg.model.coeff_vocab_size)
                    scale = world * len(atoms) / sampler.global_count(offset)
                    (loss * scale).backward()
                metrics += torch.stack([loss.detach(), atom_nll.detach(), coefficient_nll.detach()]) * (len(atoms) / sampler.global_count(offset))
                del output, loss, atom_nll, coefficient_nll, atoms, coefficients, labels
                if not final_micro: continue
                assert all(p.grad is not None for p in raw.parameters()), 'Missing model gradients'
                norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
                lr = learning_rate(step, lr=cfg.lr, warmup=cfg.warmup_steps, total=total_steps)
                for group in optimizer.param_groups: group['lr'] = lr
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                step += 1; cursor += 1
                dist.all_reduce(metrics); torch.cuda.synchronize(device)
                now = time.monotonic()
                if rank == 0:
                    values = dict(global_step=step, epoch=epoch + cursor / sampler.steps_per_epoch,
                        loss=float(metrics[0]), atom_nll=float(metrics[1]), coefficient_nll=float(metrics[2]),
                        lr=lr, gradient_norm=float(norm), seconds_per_update=now-last,
                        images_per_second=sampler.global_count(offset)/(now-last),
                        peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30,
                        elapsed_seconds=now-started)
                    print(json.dumps(values), flush=True)
                    write_json(out / 'progress.json', values)
                    with (out / 'metrics.jsonl').open('a') as handle: handle.write(json.dumps(values) + '\n')
                    if wb: wb.log({'train/' + key: value for key, value in values.items()})
                metrics.zero_()
                boundary = cursor == sampler.steps_per_epoch
                if cfg.sample_every and (step % cfg.sample_every == 0 or step == stop):
                    preview(raw, aux, cfg, step, rank, world, wb)
                evaluated = None
                if boundary and cfg.fid_every_epochs and ((epoch + 1) % cfg.fid_every_epochs == 0 or epoch == 0):
                    # Save before long evaluations so an interrupted FID loses no training.
                    snapshot()
                    evaluated = evaluate(raw, aux, cfg, rank, world)
                    best_fid = min(best_fid if best_fid is not None else math.inf, evaluated['fid'])
                    if wb: wb.log({'train/global_step': step, **{'eval/' + k: v for k, v in evaluated.items()}})
                if cfg.save_every and (step == 1 or step % cfg.save_every == 0 or boundary or step == stop):
                    snapshot(evaluated)
                last = time.monotonic()
                if step >= stop: break
    finally:
        if publisher: publisher.shutdown(wait=True)
        if wb: wb.finish()
        if dist.is_initialized(): dist.destroy_process_group()
