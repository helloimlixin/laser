"""Official CC3M text-prefix training with the audited FFHQ compound objective."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from scripts.tools.build_cc3m_compound_cache import STAGE1_SHA, ShardBatches, text_tokenizer, write_json
from src.models.rqtransformer.configs import RQTransformerConfig
from src.training import rqtransformer as rq
from src.training.background_checkpoint import BackgroundCheckpointWriter


def build_model(options):
    """Use the released body/head and the official 32-token text prefix."""
    recipe = ('ffhq/stage2/ffhq256-rqtransformer-8x8x4-350M.yaml'
              if options.get('model_preset') == 'ffhq-350m-text'
              else 'cc3m/cc3m-rqtransformer-8x8x4-650M.yaml')
    official = OmegaConf.load(rq.ROOT/'third_party/rq-vae-transformer/configs'/recipe)
    official.arch.vocab_size = official.dataset.vocab_size
    official.arch.vocab_size_cond = 16384
    official.arch.block_size_cond = 32
    config = RQTransformerConfig.create(official.arch)
    return rq.CompoundLaserRQTransformer(config, 16384, 2048,
        micro_transformer_layers=2, depth_specific_coeff_heads=True,
        pair_autoregressive=True, mask_seen_atoms_training=False)


def load_cache(path, expected_stage1_sha=STAGE1_SHA):
    data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    meta = data['meta']
    if meta['stage1_sha256'] != expected_stage1_sha or meta['shape'] != [8,8,4] or meta['clip_coefficients']:
        raise ValueError('Wrong stage-1 source, latent shape, or clipped cache')
    if data['atoms'].shape != data['coeffs'].shape or data['coeffs'].dtype != torch.float32:
        raise ValueError('Invalid compound cache tensors')
    if len(data['captions']) != len(data['atoms']) or data['text_ids'].shape != (len(data['atoms']),32):
        raise ValueError('Image/caption alignment mismatch')
    # Materialize the compact local cache once; no page faults or source-image
    # reads are left on the optimizer's hot path.
    for name in ('atoms','coeffs','text_ids'):
        data[name] = data[name].clone()
    return data


def make_aux(options, scales, device):
    return rq.LaserAux(Path(options['checkpoint']), 16384, 2048, 3.,
        coeff_scales=scales, attn_resolutions=tuple(options.get('stage1_attn_resolutions', [8])), sparsity_level=4,
        soft_target_physical=False, clamp_coeffs=False).to(device).eval()


def objective(model, aux, atoms, coeffs, text, progress, accumulation=1):
    with torch.no_grad():
        coeff_ids, probabilities = aux.compound_coeff_ids(coeffs, temp=.5, stochastic=True, hard=False)
        packed = atoms.long() * 2048 + coeff_ids
        physical = aux.physical_contributions(atoms, coeffs)
    outputs, text_logits = model(packed, model_aux=aux, cond=text, amp=False)
    weight = rq.scheduled_geometry_weight(.05, progress, 2., 3.)
    image_loss, details = rq.compound_objective(outputs['atom_logits'], outputs['coeff_logits'],
        None, atoms, probabilities, physical, atom_weight=1.5, geometry_weight=weight,
        accumulation=1, distribution_geometry=True, geometry_dictionary=aux.dictionary,
        geometry_coeff_bins=aux.coeff_bins, geometry_coeff_scales=aux.coeff_scales, geometry_top_k=4)
    text_loss = F.cross_entropy(text_logits.reshape(-1, text_logits.shape[-1]).float(), text[:,1:].reshape(-1))
    loss = (.9 * image_loss + .1 * text_loss) / accumulation
    return loss, dict(image_loss=image_loss.detach(), text_loss=text_loss.detach(),
        atom_nll=details['atom_nll'].mean().detach(), coeff_nll=details['coeff_cross_entropy'].mean().detach(),
        geometry=details['geometry'].detach())


def evaluation_indices(count, rank, world):
    """Use each held-out caption exactly once, without sampler padding."""
    return list(range(rank, count, world))


@torch.inference_mode()
def generate(model, aux, text, options):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        atoms, coeff_ids = model.sample_compound(len(text), aux, cond=text, amp=False,
            atom_temperature=1., atom_top_k=options.get('atom_top_k',16384), atom_top_p=options.get('atom_top_p',.7),
            coeff_temperature=1., coeff_top_k=0, coeff_top_p=options.get('coeff_top_p',.85))
        return ((aux.decode_compound(atoms, coeff_ids).float()+1)*.5).clamp(0,1)


@torch.inference_mode()
def prepare_reference(options, device):
    """Cache real-image statistics for the actual verified held-out subset."""
    reference = Path(options['fid_reference_stats'])
    if reference.is_file():
        return
    if options.get('dataset') == 'coco2014':
        from scripts.tools.build_coco_compound_cache import prepare_coco_reference
        return prepare_coco_reference(options, device)
    from src.rqvae_metrics import DistributedOriginalRQVAEMetrics, _mean_covariance
    world, rank = dist.get_world_size(), dist.get_rank()
    shards = sorted((Path(options['data'])/'shards').glob('cc3m-validation-*.tar'))[rank::world]
    loader = DataLoader(ShardBatches(shards, 32), batch_size=None, num_workers=2, pin_memory=True)
    metric = DistributedOriginalRQVAEMetrics(device)
    for batch in loader:
        if not batch['done']:
            metric.update((batch['images'].to(device)+1)*.5, real=True)
    for value in (metric.real_sum, metric.real_cross, metric.real_count):
        dist.all_reduce(value)
    assert int(metric.real_count) == options['validation_items']
    if rank == 0:
        mu, sigma = _mean_covariance(metric.real_sum, metric.real_cross, int(metric.real_count))
        reference.parent.mkdir(parents=True, exist_ok=True)
        np.savez(reference, mu=mu, sigma=sigma)
        write_json(reference.with_suffix('.json'), dict(items=int(metric.real_count),
            split='validation', dataset_revision=options['dataset_revision'],
            transform='resize256_center_crop256', metric='original-rqvae-inception'))
    dist.barrier()
    del metric
    torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate(model, aux, validation, options, device, wb, step, generator=generate):
    from src.rqvae_metrics import DistributedOriginalRQVAEMetrics
    import clip
    from PIL import Image
    rank, world = dist.get_rank(), dist.get_world_size()
    metric = DistributedOriginalRQVAEMetrics(device, reference_stats_path=options['fid_reference_stats'])
    clip_model, preprocess = clip.load('ViT-B/32', device=device)
    clip_model.eval()
    values = torch.zeros(2, dtype=torch.float64, device=device)
    indices = evaluation_indices(len(validation['captions']), rank, world)
    model.eval()
    started = time.monotonic()
    for offset in range(0, len(indices), options['eval_batch_size']):
        rows = indices[offset:offset+options['eval_batch_size']]
        text = validation['text_ids'][rows].long().to(device)
        pixels = generator(model, aux, text, options)
        # The official CLIP implementation truncates pixels to uint8 PIL images.
        uint8 = (pixels * 255).to(torch.uint8).cpu().permute(0,2,3,1).numpy()
        metric.update(torch.from_numpy(uint8).permute(0,3,1,2).float().to(device)/255, real=False)
        clip_images = torch.stack([preprocess(Image.fromarray(x)) for x in uint8]).to(device)
        captions = [validation['captions'][i] for i in rows]
        clip_text = clip.tokenize(captions, truncate=True).to(device)
        scores = F.cosine_similarity(clip_model.encode_image(clip_images).float(), clip_model.encode_text(clip_text).float())
        values[0] += scores.double().sum()
        values[1] += len(rows)
        if rank == 0 and offset == 0 and wb is not None:
            import wandb
            wb.log({'val/examples': [wandb.Image(Image.fromarray(x), caption=t) for x,t in zip(uint8[:8],captions[:8])], 'train/step':step})
        if rank == 0 and offset % (10*options['eval_batch_size']) == 0:
            print(json.dumps(dict(phase='evaluation', local_images=offset+len(rows), local_total=len(indices))), flush=True)
    dist.all_reduce(values)
    assert int(values[1]) == options['validation_items']
    fid, _, _ = metric.compute()
    result = dict(fid=float(fid), clip_score=float(values[0]/values[1]), items=int(values[1]), seconds=time.monotonic()-started)
    del metric, clip_model
    torch.cuda.empty_cache()
    model.train()
    return result


def run(cfg):
    options = OmegaConf.to_container(cfg.options, resolve=True)
    rank, world, local = [int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK')]
    torch.cuda.set_device(local)
    device = torch.device('cuda',local)
    dist.init_process_group('nccl', timeout=timedelta(hours=4))
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(options['seed'])
    output = Path(options['output']); output.mkdir(parents=True, exist_ok=True)
    expected_sha = options.get('stage1_sha256', STAGE1_SHA)
    import hashlib
    with Path(options['checkpoint']).open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != expected_sha:
            raise ValueError('Stage-1 checkpoint hash does not match this run')
    train = load_cache(options['token_cache'], expected_sha)
    validation = load_cache(options['validation_cache'], expected_sha)
    if train['meta']['coeff_scales'] != validation['meta']['coeff_scales']:
        raise ValueError('Validation must use training coefficient scales')
    assert len(train['atoms']) == options['train_items'] and len(validation['atoms']) == options['validation_items']
    batch, accumulation = options['batch_size'], options['accumulation']
    total_batch = batch * world * accumulation
    assert total_batch == options['total_batch_size']
    updates_per_epoch = len(train['atoms']) // total_batch
    microbatches = updates_per_epoch * accumulation
    prepare_reference(options, device)
    aux = make_aux(options, train['meta']['coeff_scales'], device)
    prior = build_model(options).to(device)
    model = DDP(prior, device_ids=[local], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'], betas=(.9,.95), weight_decay=1e-4, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['epochs']*updates_per_epoch, eta_min=0.)
    latest = output/'checkpoints/last.pt'
    step, start_epoch, start_micro, best_fid, best_clip = 0, 0, 0, [], []
    state = None
    if options['resume'] and latest.is_file():
        state = torch.load(rq._checkpoint_upload_source(latest), weights_only=True, map_location='cpu')
        for key in ('total_batch_size','batch_size','accumulation','epochs','lr','seed'):
            if state['config'][key] != options[key]:
                raise ValueError(f'Resume changes {key}')
        prior.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        step, start_epoch, start_micro = state['global_step'], state['epoch'], state['next_microbatch']
        best_fid, best_clip = state['best_fid'], state['best_clip']
        print(f'Rank {rank} resumed update {step}, epoch {start_epoch}, microbatch {start_micro}', flush=True)
    rq.restore_rank_rng_state(state, device, options['seed'])
    del state
    tokenizer = text_tokenizer(.1)
    wb = None
    if rank == 0:
        import wandb
        wb = wandb.init(entity=options['wandb_entity'], project=options['wandb_project'],
            id=options['wandb_id'], name=options['wandb_name'], mode='online', resume='allow',
            config=options, allow_val_change=True)
        wb.define_metric('train/step')
        wb.define_metric('train/*', step_metric='train/step')
        wb.define_metric('val/*', step_metric='train/step')
        wb.summary.update({'pipeline/phase':'training', 'model/parameters':sum(p.numel() for p in prior.parameters()),
            'cache/coeff_scales':train['meta']['coeff_scales'],
            'checkpoints/latest_online_file':'last.pt', 'checkpoints/best_fid_online_file':'best-fid-01.pt',
            'checkpoints/best_clip_online_file':'best-clip-01.pt'})
    writer = BackgroundCheckpointWriter() if rank == 0 else None
    stopping = {'requested':False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.update(requested=True))

    def upload():
        rq.upload_selected_checkpoint_files(wb, last_checkpoint=latest, best_fid=best_fid,
            best_clip=best_clip, upload_dir=output/'wandb_checkpoints')

    def save(epoch, next_micro, metrics=None, synchronous=False):
        nonlocal best_fid, best_clip
        rng = rq.gather_rank_rng_states(device)
        if rank == 0:
            writer.wait()
            if metrics:
                candidate = dict(model=prior.state_dict(), config=options, epoch=epoch,
                    global_step=step, metrics=metrics, cache_meta=train['meta'])
                if not best_fid or metrics['fid'] < best_fid[0][0]:
                    path = output/'checkpoints/best-fid-01.pt'
                    rq.atomic_torch_save(candidate,path)
                    best_fid = [(metrics['fid'],str(path))]
                if not best_clip or metrics['clip_score'] > best_clip[0][0]:
                    path = output/'checkpoints/best-clip-01.pt'
                    rq.atomic_torch_save(candidate,path)
                    best_clip = [(metrics['clip_score'],str(path))]
            snapshot = dict(model=prior.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), config=options, epoch=epoch,
                next_microbatch=next_micro, global_step=step, rng_state_by_rank=rng,
                best_fid=best_fid, best_clip=best_clip, cache_meta=train['meta'],
                text_rng_resume='BPE dropout uses an independent tokenizer RNG; draws restart on resume')
            rq.atomic_torch_save(snapshot, latest, background=None if synchronous else writer, on_commit=upload)
        dist.barrier()

    if rank == 0 and latest.is_file():
        upload()
    started, window_start, window_step = time.monotonic(), time.monotonic(), step
    try:
        for epoch in range(start_epoch, options['epochs']):
            # Each global update is partitioned into disjoint rank microbatches.
            order = torch.randperm(len(train['atoms']), generator=torch.Generator().manual_seed(options['seed']+epoch))
            model.train(); optimizer.zero_grad(set_to_none=True)
            first = start_micro if epoch == start_epoch else 0
            for micro in range(first, microbatches):
                if writer is not None:
                    writer.check()
                begin = (micro * world + rank) * batch
                rows = order[begin:begin+batch]
                atoms = train['atoms'][rows].to(device, dtype=torch.long)
                coeffs = train['coeffs'][rows].to(device)
                captions = [train['captions'][i] for i in rows.tolist()]
                text = torch.tensor([r.ids for r in tokenizer.encode_batch(captions)], dtype=torch.long, device=device)
                sync = (micro+1)%accumulation == 0
                with (nullcontext() if sync else model.no_sync()), torch.autocast('cuda',dtype=torch.bfloat16):
                    loss, details = objective(model, aux, atoms, coeffs, text,
                        epoch+micro/microbatches, accumulation)
                    loss.backward()
                if not sync:
                    continue
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                step += 1
                stop = torch.tensor(int(stopping['requested']),device=device)
                dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if rank == 0 and (step == 1 or step % 10 == 0):
                    now = time.monotonic()
                    metrics = {'train/step':step, 'train/loss':float(loss.detach())*accumulation,
                        'train/grad_norm':float(grad), 'train/lr':optimizer.param_groups[0]['lr'],
                        'train/epoch':epoch+(micro+1)/microbatches,
                        'train/images_per_second':(step-window_step)*total_batch/(now-window_start),
                        **{'train/'+k:float(v) for k,v in details.items()}}
                    wb.log(metrics)
                    print(json.dumps(metrics),flush=True)
                    write_json(output/'status.json', dict(phase='training', timestamp=time.time(), **metrics))
                    window_start, window_step = now, step
                if step % options['save_step_freq'] == 0 or step == 1 or int(stop):
                    save(epoch,micro+1,synchronous=bool(int(stop)))
                if int(stop):
                    return
                if step % options['sample_grid_every'] == 0:
                    # Fork Torch RNG so previews do not change training stochastic contexts.
                    with torch.random.fork_rng(devices=[local]):
                        torch.manual_seed(1701)
                        prior.eval()
                        if rank == 0:
                            rows_preview = list(range(8))
                            pixels = generate(prior,aux,validation['text_ids'][rows_preview].long().to(device),options)
                            wb.log({'train/step':step,'samples/text_to_image':[
                                wandb.Image(pixels[i].cpu(),caption=validation['captions'][i]) for i in rows_preview]})
                        prior.train()
                    dist.barrier()
            results = None
            if (epoch+1)%options['eval_every'] == 0:
                with torch.random.fork_rng(devices=[local]):
                    torch.manual_seed(1701+rank)
                    results = evaluate(prior,aux,validation,options,device,wb,step)
                if rank == 0:
                    wb.log({'train/step':step, **{'val/'+k:v for k,v in results.items()}})
                    write_json(output/f'evaluation/epoch-{epoch+1:03d}.json',results)
                    wb.summary['best/fid'] = min(results['fid'], best_fid[0][0] if best_fid else math.inf)
                    wb.summary['best/clip_score'] = max(results['clip_score'], best_clip[0][0] if best_clip else -math.inf)
            save(epoch+1,0,results,synchronous=True)
            start_micro = 0
        if rank == 0:
            wb.summary['pipeline/phase'] = 'completed'
    finally:
        if writer is not None:
            writer.close()
        if wb is not None:
            wb.finish()
        dist.destroy_process_group()


if __name__ == '__main__':
    from src.training.cli import main
    main()
