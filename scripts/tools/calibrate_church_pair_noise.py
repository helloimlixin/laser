#!/usr/bin/env python3
"""Measure sparse noise entropy and reconstruction distortion on Church pixels."""
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path('/mnt/laser-church/dropout-experiment')
OUT = ROOT / 'noise-calibration'
sys.path.insert(0, str(ROOT/'runtime'))
import lmdb
import numpy as np
from PIL import Image
import torch
from src.training.rqtransformer import LaserAux, val_image_transform
from src.stochastic_compound import stochastic_omp


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2026092431)
    device = torch.device('cuda:0')
    bank = torch.load(ROOT/'assets/compound-cache.pt', weights_only=True, mmap=True)
    meta = bank['meta']
    aux = LaserAux(Path('/mnt/laser-church/assets/tokenizer.pt'), 16384, 2048,
        meta['coeff_max'], 1., coeff_scales=[1.]*4, coeff_bin_centers=meta['coeff_bin_centers'],
        soft_target_physical=True, clamp_coeffs=False, sparsity_level=4, attn_resolutions=(8,)).to(device).eval()
    transform = val_image_transform()
    gram = aux.dictionary.T @ aux.dictionary
    splits = {}
    for split in ['train', 'val']:
        path = Path('/workspace/Projects/data/lsun/lmdb')/f'church_outdoor_{split}_lmdb'
        env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
        if (path/'keys.npy').exists():
            keys = np.load(path/'keys.npy', allow_pickle=False)
        else:
            with env.begin() as txn:
                keys = sorted(k for k, _ in txn.cursor())
        indices = np.random.default_rng(2026092431).choice(len(keys), 64, replace=False)
        pixels, latents = [], []
        for start in range(0, 64, 4):
            with env.begin() as txn:
                images = torch.stack([transform(Image.open(io.BytesIO(txn.get(bytes(keys[i])))).convert('RGB')) for i in indices[start:start+4]]).to(device)
            z = aux.quant_conv(aux.encoder(images)).permute(0, 2, 3, 1).contiguous().float()
            pixels.append(images.cpu());latents.append(z.cpu())
        env.close()
        splits[split] = dict(images=torch.cat(pixels), latents=torch.cat(latents), indices=indices.tolist())
    torch.save(splits, OUT/'encoded-probe.pt')
    started = time.monotonic()
    records = {}
    for split, data in splits.items():
        z = data['latents'].to(device)
        def omp(temp, seed):
            torch.manual_seed(seed)
            return stochastic_omp(z, aux.dictionary, depth=4, temperature=temp, gram=gram)
        hard = omp(0., 0)
        hard_error = (z-hard['quantized']).square().mean().item()
        atoms = []
        for temp in [0., .03125, .0625, .125, .25, .5]:
            trials = []
            for seed in range(3):
                r = omp(temp, 2026092440+seed)
                trials.append(dict(latent_mse=(z-r['quantized']).square().mean().item(),
                    changed_ordered_support=(r['atoms']!=hard['atoms']).any(-1).float().mean().item(),
                    changed_atom_by_depth=(r['atoms']!=hard['atoms']).float().mean((0,1,2)).tolist(),
                    entropy_by_depth=r['selection_entropy'].mean((0,1,2)).tolist()))
            row=dict(temperature=temp, latent_mse=float(np.mean([r['latent_mse'] for r in trials])),
                latent_mse_ratio=float(np.mean([r['latent_mse'] for r in trials]))/hard_error,
                changed_ordered_support=float(np.mean([r['changed_ordered_support'] for r in trials])),
                changed_atom_by_depth=np.mean([r['changed_atom_by_depth'] for r in trials],axis=0).tolist(),
                entropy_by_depth=np.mean([r['entropy_by_depth'] for r in trials],axis=0).tolist())
            atoms.append(row)
            print(json.dumps(dict(split=split,kind='atom',**row)),flush=True)
        stochastic = omp(.0625, 2026092440)
        a, c = stochastic['atoms'], stochastic['coefficients']
        vectors = aux.dictionary.T[a].reshape(-1,4,256)
        values = c.reshape(-1,4)
        latent = z.reshape(-1,256)
        centers = aux.coeff_bins
        base_error = (z-stochastic['quantized']).square().mean().item()
        coefficient_rows = []
        for temp in [0., .03125, .125, .25, .5, 1., 2.]:
            total_error, variance_sum, bias_sum, entropies, stds, flips, ids = [], [], [], [], [], [], []
            for start in range(0,len(values),256):
                target=values[start:start+256];v=vectors[start:start+256]
                nearest=(target[...,None]-centers).abs().argmin(-1)
                if temp:
                    q=(-(target[...,None]-centers).square()/temp).softmax(-1)
                else:
                    q=torch.nn.functional.one_hot(nearest,2048).float()
                mean=(q*centers).sum(-1)
                variance=(q*(centers-mean[...,None]).square()).sum(-1)
                expectation=(v*mean[...,None]).sum(-2)
                # Independent coefficient draws: exact expected latent squared error.
                noise_variance=(variance*v.square().sum(-1)).sum(-1)
                total_error.append(((latent[start:start+256]-expectation).square().sum(-1)+noise_variance)/256)
                variance_sum.append(noise_variance/256)
                bias_sum.append((mean-target).square())
                entropies.append(-(q*q.clamp_min(1e-30).log()).sum(-1))
                stds.append(variance)
                flips.append(1-q.gather(-1,nearest[...,None]).squeeze(-1))
                ids.append(torch.multinomial(q.reshape(-1,2048),1).reshape_as(target))
            sampled=centers[torch.cat(ids)].reshape_as(c)
            quantized=(aux.dictionary.T[a]*sampled[...,None]).sum(-2)
            pixel_error=[]; versus_hard=[]
            # A fixed 16-image decoded diagnostic supplements the exact latent expectation.
            for start in range(0,16,4):
                decoded=aux.decoder(aux.post_quant_conv(quantized[start:start+4].permute(0,3,1,2))).clamp(-1,1)
                reference=aux.decoder(aux.post_quant_conv(hard['quantized'][start:start+4].permute(0,3,1,2))).clamp(-1,1)
                pixels=data['images'][start:start+4].to(device)
                pixel_error.append((decoded-pixels).square().flatten(1).mean(-1)/4)
                versus_hard.append((decoded-reference).square().flatten(1).mean(-1)/4)
            expected_error=torch.cat(total_error).mean().item()
            row=dict(temperature=temp,expected_latent_mse=expected_error,
                ratio_to_greedy=expected_error/hard_error,ratio_to_same_support_continuous=expected_error/base_error,
                noise_variance_mse=torch.cat(variance_sum).mean().item(),
                entropy_by_depth=torch.cat(entropies).mean(0).tolist(),
                physical_std_rms_by_depth=torch.cat(stds).mean(0).sqrt().tolist(),
                mean_shift_rms_by_depth=torch.cat(bias_sum).mean(0).sqrt().tolist(),
                change_from_nearest_probability_by_depth=torch.cat(flips).mean(0).tolist(),
                decoded_16_image_mean_psnr=(-10*torch.cat(pixel_error).log10()).mean().item(),
                decoded_mse_vs_greedy=torch.cat(versus_hard).mean().item())
            coefficient_rows.append(row)
            print(json.dumps(dict(split=split,kind='coefficient',**row)),flush=True)
        records[split]=dict(images=64,sites=len(values),greedy_latent_mse=hard_error,
            coefficient_rms_by_depth=values.square().mean(0).sqrt().tolist(),
            atoms=atoms,coefficients=coefficient_rows,indices=data['indices'])
        write(OUT/'calibration.json',dict(results=records,elapsed_seconds=time.monotonic()-started))
    write(OUT/'calibration.json',dict(passed=True,seed=2026092431,
        stage1_sha256=meta['stage1_checkpoint_sha256'],training_policy_changed=False,
        note='Diagnostic only, not a generated FID experiment. Atom rows average three draws; coefficient rows use exact conditional moments on one fixed support draw; pixels use 16 images and one coefficient draw.',
        results=records,elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
