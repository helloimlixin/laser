#!/usr/bin/env python3
"""Fit the Church-style integer codebook on the frozen ImageNet 4.21 tokenizer."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get('LASER_PROJECT_ROOT', Path(__file__).resolve().parents[2]))
UPSTREAM = ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(UPSTREAM)]

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from rqvae.img_datasets.transforms import create_transforms
from src.original_rq_training import atomic_json, file_sha256, load_tokenizer, state_sha256
from src.scaled_atom_rq import (FrozenSparseBackbone, continuous_matching_pursuit,
                                orthogonal_matching_pursuit, fit_signed_levels)
from src.scaled_atom_training import FrozenScaledTokenizer, TrainingScaledAtomRQ
from src.imagenet_scaled_stage2 import image_manifest, ManifestImages, load_imagenet_config
from calibrate_scaled_atom_temperature import measure


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('/workspace/Projects/data/imagenet2012'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fit-images', type=int, default=2048)
    parser.add_argument('--temperature-images', type=int, default=128)
    parser.add_argument('--levels',type=int,default=8)
    parser.add_argument('--levels-file',type=Path)
    parser.add_argument('--levels-variant')
    parser.add_argument('--manifest-source',type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    started = time.time()
    def status(phase, **values):
        record = dict(phase=phase,elapsed_seconds=time.time()-started,updated_unix=time.time(),**values)
        atomic_json(out/'preparation-status.json',record)
        print(json.dumps(record),flush=True)
    manifests = {}
    for split, expected in [('train',1281167),('val',50000)]:
        target = out/f'{split}-manifest.json'
        if not target.exists() and args.manifest_source:
            target.symlink_to((args.manifest_source/f'{split}-manifest.json').resolve())
        if not target.exists():
            status('indexing_images',split=split)
            atomic_json(target,image_manifest(args.data/split))
        manifests[split] = json.loads(target.read_text())
        assert manifests[split]['images'] == expected
    assert manifests['train']['class_to_idx'] == manifests['val']['class_to_idx']
    checkpoint_hash = file_sha256(args.checkpoint)
    provenance = json.loads((args.checkpoint.parent/'artifact-provenance.json').read_text())
    best = [x for x in provenance['metadata']['best'] if x['slot']==1][0]
    assert abs(best['rfid']-4.210914134979248) < 1e-8 and best['epoch']==10
    backbone = FrozenSparseBackbone(args.checkpoint).to(args.device).eval()
    before = state_sha256(backbone)
    config = load_imagenet_config(UPSTREAM)
    transform = create_transforms(config.dataset,split='train')
    order = np.random.default_rng(421).permutation(manifests['train']['images'])
    fit_ids = order[:args.fit_images].tolist()
    temp_ids = order[args.fit_images:args.fit_images+args.temperature_images].tolist()
    def loader(split, indices, train=True):
        return DataLoader(ManifestImages(args.data/split,manifests[split],
            transform if train else create_transforms(config.dataset,split='val'),
            seed=421,indices=indices),batch_size=16,num_workers=8,pin_memory=True)
    if args.levels_file:
        fitted=torch.load(args.levels_file,weights_only=True,map_location='cpu')
        assert fitted['checkpoint_sha256']==checkpoint_hash
        assert fitted['fit_indices']==fit_ids
        torch.testing.assert_close(fitted['dictionary'],backbone.dictionary.cpu(),rtol=0,atol=0)
        if args.levels_variant:
            levels=fitted['candidates'][args.levels_variant].to(args.device)
            assert len(levels)==args.levels
        else:
            levels=fitted['levels']['mp'][str(args.levels)].to(args.device)
    else:
        coefficients = []
        for images,_,_ in loader('train',fit_ids):
            z = backbone.encode(images.to(args.device,non_blocking=True))
            coefficients.append(continuous_matching_pursuit(z,backbone.dictionary)['coefficients'].cpu())
            if sum(len(x) for x in coefficients)%256 == 0:
                status('fitting_imagenet_coefficient_levels',images=sum(len(x) for x in coefficients))
        levels = fit_signed_levels(torch.cat(coefficients).to(args.device),args.levels)
    codebook = out/'scaled-atom-codebook.pt'
    torch.save(dict(dictionary=backbone.dictionary.cpu(),levels={str(args.levels):levels.cpu()},
                    checkpoint_sha256=checkpoint_hash,fit_indices=fit_ids),codebook)
    backbone.to(args.device)
    quantizer = TrainingScaledAtomRQ(backbone.dictionary,levels).to(args.device)
    reference_root = Path('/workspace/tmp/original-rqvae-473/published-stage1')
    reference,_ = load_tokenizer(reference_root/'model.pt',reference_root/'config.yaml',args.device)
    reference_z, sparse_z = [], []
    for images,_,_ in loader('train',temp_ids):
        images = images.to(args.device,non_blocking=True)
        reference_z.append(reference.encode(images))
        sparse_z.append(backbone.encode(images))
    original = measure(reference.quantizer,torch.cat(reference_z),.5,99200)
    limit = original['sampled_to_hard_residual_mse_ratio']+.02
    status('temperature_control',**original)
    sweep = []
    for temperature in [.5,.25,.125,.0625,.03125,.015625,.0078125]:
        result = measure(quantizer,torch.cat(sparse_z),temperature,99200)
        sweep.append(result)
        status('temperature_sweep',**result)
        if result['sampled_to_hard_residual_mse_ratio'] <= limit:
            break
    assert sweep[-1]['sampled_to_hard_residual_mse_ratio'] <= limit
    calibration = dict(selected_temperature=sweep[-1]['temperature'],original_rq_control=original,
        scaled_rq_sweep=sweep,allowed_mse_ratio=limit,calibration_indices=temp_ids,
        fit_indices=fit_ids,levels=levels.cpu().tolist(),level_count=args.levels,source_checkpoint_sha256=checkpoint_hash,
        codebook_sha256=file_sha256(codebook),control_checkpoint_sha256=file_sha256(reference_root/'model.pt'),
        selection_rule='Largest tested temperature <=0.5 with sampled/hard latent MSE ratio <= original ImageNet RQ at 0.5 plus 0.02',
        control_dataset='same disjoint ImageNet training images',source_rfid=best['rfid'])
    atomic_json(out/'temperature-calibration.json',calibration)
    del reference,reference_z,sparse_z
    gram = backbone.dictionary.T @ backbone.dictionary
    stats = torch.zeros(4,device=args.device,dtype=torch.float64)
    validation_ids = np.random.default_rng(422).choice(50000,256,replace=False).tolist()
    for batch,(images,labels,indices) in enumerate(loader('val',validation_ids,False)):
        images = images.to(args.device)
        z = backbone.encode(images)
        omp = orthogonal_matching_pursuit(z,backbone.dictionary,gram)
        scaled = quantizer.quantize(z)
        torch.testing.assert_close(quantizer.embed(scaled['codes']).sum(-2),scaled['quantized'])
        decoded = backbone.decode(scaled['quantized'])
        stats[0] += (omp['quantized']-z).square().sum()
        stats[1] += (scaled['quantized']-z).square().sum()
        stats[2] += (decoded.clamp(-1,1)-images).square().sum()/4
        stats[3] += len(images)
        if batch==0:
            save_image(torch.cat([images[:8],decoded[:8]]).mul(.5).add(.5).clamp(0,1),
                       out/'tokenizer-reconstructions.png',nrow=8)
            # Tiny real-image cache exercises exactly the production loader/trainer.
            smoke = out/'smoke-cache'
            smoke.mkdir(exist_ok=True)
            for split in ('train','val'):
                np.save(smoke/f'{split}-view0-latents.npy',z.cpu().numpy())
                np.save(smoke/f'{split}-view0-codes.npy',scaled['codes'].cpu().numpy().astype(np.uint32))
                np.save(smoke/f'{split}-labels.npy',labels.numpy().astype(np.int16))
            atomic_json(smoke/'complete.json',dict(train_views=1,train_images=len(images),val_images=len(images),
                checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=checkpoint_hash,
                codebook=str(codebook),codebook_sha256=file_sha256(codebook),frozen_state_sha256=None,
                smoke_only=True))
    assert before==state_sha256(backbone)
    quality = dict(validation_images=int(stats[3]),omp_latent_mse=float(stats[0]/(stats[3]*64*256)),
        scaled_latent_mse=float(stats[1]/(stats[3]*64*256)),scaled_to_omp_latent_mse_ratio=float(stats[1]/stats[0]),
        scaled_reconstruction_psnr=float(-10*torch.log10(stats[2]/(stats[3]*3*256*256))),
        source_rfid=best['rfid'],expanded_tokenizer_rfid='pending full validation cache',
        token_formula=f'0 = zero; 1 + atom_id * {args.levels} + coefficient_bin',vocab_size=quantizer.vocab_size,
        checkpoint_sha256=checkpoint_hash,codebook_sha256=file_sha256(codebook),frozen_backbone_unchanged=True)
    atomic_json(out/'tokenizer-verification.json',quality)
    status('complete',**quality)


if __name__=='__main__':
    main()
