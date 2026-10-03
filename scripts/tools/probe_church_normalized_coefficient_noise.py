#!/usr/bin/env python3
"""Compare deterministic OMP with FFHQ-style normalized coefficient targets."""
import json
from pathlib import Path
import sys

ROOT = Path('/mnt/laser-church/dropout-experiment')
OUT = ROOT/'normalized-coefficient-probe'
sys.path.insert(0, str(ROOT/'runtime'))
import torch
from src.training.rqtransformer import LaserAux
from src.stochastic_compound import stochastic_omp


@torch.inference_mode()
def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_device(3)
    torch.cuda.set_per_process_memory_fraction(.05,3)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:3')
    cache=torch.load('/mnt/laser-church/assets/compound-cache.pt',weights_only=True,mmap=True)
    scales=torch.tensor(cache['meta']['coeff_scales'],device=device)
    aux=LaserAux(Path('/mnt/laser-church/assets/tokenizer.pt'),16384,2048,3.,1.,
        coeff_scales=scales.cpu().tolist(),soft_target_physical=False,clamp_coeffs=False,
        sparsity_level=4,attn_resolutions=(8,)).to(device).eval()
    data=torch.load(ROOT/'noise-calibration/encoded-probe.pt',weights_only=True)
    bins=aux.coeff_bins
    records={}
    for split,d in data.items():
        z=d['latents'].to(device)
        if split=='train':
            indices=d['indices']
            atoms=cache['atoms'][indices].to(device).long()
            clean_coefficients=cache['coeffs'][indices].to(device)*scales
            clean=(aux.dictionary.T[atoms]*clean_coefficients[...,None]).sum(-2)
        else:
            # Batch sites to cap scratch memory while the active trainer runs.
            gram=aux.dictionary.T@aux.dictionary
            parts=[stochastic_omp(chunk,aux.dictionary,depth=4,temperature=0.,gram=gram)
                   for chunk in z.split(8)]
            atoms=torch.cat([p['atoms'] for p in parts])
            clean_coefficients=torch.cat([p['coefficients'] for p in parts])
            clean=torch.cat([p['quantized'] for p in parts])
            del gram,parts
        normalized=clean_coefficients/scales
        vectors=aux.dictionary.T[atoms].reshape(-1,4,256)
        u=normalized.reshape(-1,4)
        target=clean.reshape(-1,256)
        latent=z.reshape(-1,256)
        rows=[]
        for space,temp in [('normalized',0.),('physical',.125),('normalized',.125),('normalized',.25),('normalized',.5)]:
            torch.manual_seed(2026092461)
            variances=[];means=[];entropies=[];errors=[];reconstruction=[];ids=[];flips=[]
            for start in range(0,len(u),256):
                part=u[start:start+256]
                aux.soft_target_physical=space=='physical'
                selected,q=aux.compound_coeff_ids(part,stochastic=temp>0,temp=max(temp,1e-6),hard=temp==0)
                ids.append(selected)
                centers=bins[None,None,:]*scales[None,:,None]
                mean=(q*centers).sum(-1)
                variance=(q*(centers-mean[...,None]).square()).sum(-1)
                v=vectors[start:start+256]
                mean_latent=(v*mean[...,None]).sum(-2)
                total_variance=(variance*v.square().sum(-1)).sum(-1)
                errors.append(((mean_latent-target[start:start+256]).square().sum(-1)+total_variance)/256)
                reconstruction.append(((mean_latent-latent[start:start+256]).square().sum(-1)+total_variance)/256)
                means.append(mean-part*scales)
                variances.append(variance)
                entropies.append(-(q*q.clamp_min(1e-30).log()).sum(-1))
                flips.append((q*(centers*(part*scales)[...,None]<0)).sum(-1))
                if space=='normalized' and temp:
                    expected=(-(part[...,None]-bins).square()/temp).softmax(-1)
                    torch.testing.assert_close(q,expected,rtol=0,atol=0)
            sampled=bins[torch.cat(ids)].reshape_as(normalized)*scales
            noisy=(aux.dictionary.T[atoms]*sampled[...,None]).sum(-2)
            psnr=[];clean_psnr=[];baseline_psnr=[]
            for start in range(0,16,2):
                def decode(quantized):
                    return aux.decoder(aux.post_quant_conv(quantized[start:start+2].permute(0,3,1,2))).clamp(-1,1)
                decoded=decode(noisy)
                reference=decode(clean)
                pixels=d['images'][start:start+2].to(device)
                psnr.append(-10*((decoded-pixels).square().flatten(1).mean(-1)/4).log10())
                clean_psnr.append(-10*((decoded-reference).square().flatten(1).mean(-1)/4).clamp_min(1e-20).log10())
                baseline_psnr.append(-10*((reference-pixels).square().flatten(1).mean(-1)/4).log10())
            variance=torch.cat(variances)
            row=dict(target_space=space,temperature=temp,
                physical_sigma_rms_by_depth=variance.mean(0).sqrt().tolist(),
                normalized_sigma_rms_by_depth=(variance.mean(0).sqrt()/scales).tolist(),
                physical_bias_rms_by_depth=torch.cat(means).square().mean(0).sqrt().tolist(),
                entropy_by_depth=torch.cat(entropies).mean(0).tolist(),
                coefficient_sign_flip_probability=torch.cat(flips).mean(0).tolist(),
                expected_noise_mse_relative_to_clean_energy=torch.cat(errors).mean().item()/target.square().mean().item(),
                expected_reconstruction_mse_ratio=torch.cat(reconstruction).mean().item()/(latent-target).square().mean().item(),
                pixel_psnr_to_original_16_images=torch.cat(psnr).mean().item(),
                pixel_psnr_to_clean_reconstruction_16_images=torch.cat(clean_psnr).mean().item(),
                clean_pixel_psnr_to_original_16_images=torch.cat(baseline_psnr).mean().item())
            rows.append(row);print(json.dumps(dict(split=split,**row)),flush=True)
        records[split]=dict(images=len(z),indices=d['indices'],coeff_rms=clean_coefficients.square().mean((0,1,2)).sqrt().tolist(),
            scales=scales.cpu().tolist(),normalized_outside_bins_fraction=(normalized.abs()>3).float().mean().item(),rows=rows)
        (OUT/'report.json').write_text(json.dumps(dict(results=records),indent=2)+'\n')
    report=dict(passed=True,stage1_sha256=cache['meta']['stage1_checkpoint_sha256'],
        normalization='Church full-training maximum absolute coefficient / 3, per depth; this is not RMS normalization',
        training_changed=False,atoms='deterministic OMP',results=records,
        note='Same normalized kernel as saved FFHQ trainer, extended to K4. Exact moments; one seeded draw for 16-image pixel diagnostics. No generated FID or optimal-noise claim.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
