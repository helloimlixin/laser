#!/usr/bin/env python3
"""Verify the transferred compound model and its actual finite tokenizer."""
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(os.environ['LASER_RQ_BASE'])
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
os.environ['TORCH_HOME'] = str(BASE/'torch-cache')
sys.path[:0] = [str(BASE/'source'), str(BASE/'upstream'), str(BASE/'runtime'), str(BASE)]

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf
from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from src.training import rq_imagenet_shared as shared
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config


def main():
    rank, world, local = (int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    assert world == 4
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', device_id=device)
    profile = CompoundReplayRecipe(BASE)
    out = Path('/workspace/Projects/laser/outputs')/profile.run_id
    saved = torch.load(BASE/'converted-epoch002.pt', map_location='cpu', mmap=True, weights_only=False)
    profile.verify_resume(saved)
    assert saved['global_step'] == 1252 and saved['next_epoch'] == 2 and saved['next_microbatch'] == 0
    assert len(saved['optimizer']['state']) == 837
    assert all(int(v['step']) == 1252 for v in saved['optimizer']['state'].values())
    assert saved['scheduler']['after']['T_max'] == 62600
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(saved['config']['arch']))
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    del saved
    model.requires_grad_(False).eval().to(device)
    aux, _ = profile.tokenizer(device)
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    data = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
    images = torch.stack([data[rank*2+i][0] for i in range(2)]).to(device)
    labels = torch.tensor([data[rank*2+i][1] for i in range(2)], device=device)
    with torch.inference_mode():
        targets, codes = profile.targets(aux, images)
        target_proof = profile.verify_targets(aux, targets, codes)
        expected = model(codes, aux, labels, amp=True)
        prefix = torch.zeros_like(codes)
        prefix[..., 1::2] = aux.num_atoms+aux.coeff_vocab_size//2
        model.init_cache()
        maximum = torch.zeros((), device=device)
        for h in range(8):
            for w in range(8):
                for event in range(8):
                    logits = model.cached_forward(prefix[:, :h+1], aux, labels,
                        amp=True, sample_loc=(h,w,event))
                    key = 'atom_logits' if event%2 == 0 else 'coeff_logits'
                    logits = logits[:, :aux.num_atoms] if event%2 == 0 else logits[:, aux.num_atoms:]
                    p = expected[key][:,h,w,event//2].float().log_softmax(-1)
                    q = logits.float().log_softmax(-1)
                    kl = torch.where(torch.isfinite(p), p.exp()*(p-q), 0.).sum(-1).max()
                    maximum = torch.maximum(maximum, kl)
                    prefix[:,h,w,event] = codes[:,h,w,event]
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        assert float(maximum) < 2.e-4
        del targets, codes, expected, prefix, images, labels, logits, p, q
        model.init_cache()
        torch.cuda.empty_cache()
        shared.BASE, shared.OUT, shared.PROFILE, shared.RUN = BASE, out/'baseline', profile, profile.run_id
        shared.original_metrics.INCEPTION_PATH = BASE/'upstream/rqvae/metrics/inception.py'
        before = profile.digest(aux)
        started = time.time()
        baseline = shared.generation(model, aux, device, rank, world, 2, None, batch=512, seed=261001)
        assert profile.digest(aux) == before
        if rank == 0:
            baseline.update(generated_images=50000, real_images=1281167, real_split='train',
                source_step=1252, source_recorded_fid=float(profile.recipe.continuation.source_fid),
                generation_seed=261001, generation_batch=512, training_updates_added=0,
                elapsed_seconds=time.time()-started)
            write_json(out/'baseline/fid50k.json', baseline)
        metric = shared.original_metrics.DistributedOriginalRQVAEMetrics(device)
        loader = DataLoader(Subset(data, range(rank,len(data),world)), batch_size=8,
                            num_workers=4, pin_memory=True)
        for index,(images,_) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            codes = profile.hard_codes(aux, images)
            decoded = profile.decode(aux, codes)
            assert torch.isfinite(decoded).all()
            metric.update(images.mul(.5).add(.5).clamp(0,1), real=True)
            metric.update(decoded.mul(.5).add(.5).clamp(0,1), real=False)
            if rank == 0 and index%50 == 0:
                write_json(out/'baseline/reconstruction-progress.json', dict(batches_on_rank=index+1,
                    reconstructed_on_rank=min((index+1)*8,12500), target_per_rank=12500,time=time.time()))
        assert int(metric.real_count) == int(metric.fake_count) == 12500
        rfid, _, _ = metric.compute()
        assert profile.digest(aux) == before
    if rank == 0:
        rfid_limit = float(profile.recipe.verification.maximum_reconstruction_rfid)
        fid_limit = float(profile.recipe.continuation.source_fid)+float(profile.recipe.verification.maximum_baseline_fid_regression)
        passed = rfid <= rfid_limit and baseline['fid'] <= fid_limit
        proof = dict(passed=passed, strict_model_reload=True, checkpoint_step=1252,
            optimizer_states=837, all_adam_ages=1252, cosine_updates=62600,
            full_cache_events_per_rank=512, ranks=world, maximum_fp16_cache_kl=float(maximum),
            targets=target_proof, full_generation_fid=baseline['fid'], maximum_baseline_fid=fid_limit,
            actual_compound_tokenizer_rfid=rfid, maximum_tokenizer_rfid=rfid_limit,
            reconstruction_real_images=50000, reconstruction_generated_images=50000,
            generation_images=50000, generation_reference_training_images=1281167,
            original_dictionary_encoder_decoder_frozen=True, training_updates_added=0)
        write_json(out/'baseline/runtime-verification.json', proof)
        print(json.dumps(proof),flush=True)
        assert passed, 'Compound baseline or finite-tokenizer reconstruction gate failed'
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
