"""Evaluate coefficient support using native original FID50k and its IS."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--distribution', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--policy', choices=['clean-q995', 'observed-max', 'baseline'], required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.base/'source'), str(args.base/'source/runtime')]
    import torch
    import torch.distributed as dist
    from src.training import rqtransformer as training
    from src.training.imagenet_ffhq_adapter import ImageNetFFHQCompound, build_model
    from src.training.k4_checkpoint_io import _checkpoint_upload_source
    training.CompoundLaserRQTransformer = ImageNetFFHQCompound
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', timeout=timedelta(minutes=40))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(args.cache_dir)
    os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
    payload = torch.load(_checkpoint_upload_source(args.checkpoint.resolve()),
                         map_location='cpu', mmap=True, weights_only=False)
    config = payload['config']
    model = build_model(18432, 16384, compound=True, coeff_vocab_size=2048,
        sparsity_level=4, compound_micro_transformer_layers=2,
        compound_depth_specific_coeff_heads=True, compound_pair_autoregressive=True,
        physical_pair_context=False, model_preset='imagenet-1400m')
    model.load_state_dict(payload['state_dict'], strict=True)
    step, epoch = payload['global_step'], payload['epoch']
    del payload
    for block in model.head_transformer.blocks:
        block.attn.short_attention_backend = 'compiled'
    distribution = json.loads(args.distribution.read_text())
    if args.policy != 'baseline':
        key = '0.995' if args.policy == 'clean-q995' else '1.0'
        model.coefficient_sampling_limits = [d['normalized_absolute_quantiles'][key]
                                             for d in distribution['depths']]
    model.to(device).eval()
    aux = training.LaserAux(args.base/'inputs/stage1-tokenizer.pt', 16384, 2048, 3.,
        coeff_scales=config['coeff_scales'], soft_target_physical=False,
        clamp_coeffs=False, sparsity_level=4).to(device).eval()
    reference = args.base/'inputs/imagenet_256_train.npz'
    started = time.time()
    fid, score, std = training.evaluate_generation_metrics(model, aux, None, 50000, 128,
        num_condition_classes=1000, atom_temperature=.9, atom_top_k=0, atom_top_p=.9,
        coeff_temperature=1., coeff_top_k=0, coeff_top_p=.85,
        compute_inception_score=True, metric_backend='original-rqvae',
        fid_reference_stats=reference, fid_seed=261001)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        result = dict(policy=args.policy, checkpoint=str(args.checkpoint.resolve()),
            global_step=step, epoch=epoch, coefficient_target_sigma_bins=config['coefficient_noise_sigma_bins'],
            sampling_limits=model.coefficient_sampling_limits,
            generated_images=50000, real_split='full_train', metric_backend='original-rqvae',
            reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
            generation_batch=128, decoder_batch=64, world_size=dist.get_world_size(),
            seed=261001, fid_original_train50k=float(fid), inception_score=float(score),
            inception_score_std=float(std), elapsed_seconds=time.time()-started,
            wandb_logged=False, optimizer_loaded=False, training_modified=False)
        (args.output/(args.policy+'-evaluation.json')).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
