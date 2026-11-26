#!/usr/bin/env python3
"""Audit whether fixed-support labels match residuals of emitted prefixes."""
import json
import time

from audit_compound512_gap import BASE, ROOT, OUT, CompoundReplayRecipe, torch
from src.training.laser_hard_ar import sparse_encoding, nearest_coefficient_ids
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from rqvae.img_datasets.transforms import create_transforms
from rqvae.utils.config import load_config


@torch.inference_mode()
def main():
    torch.set_num_threads(4); torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(5*1024**3/torch.cuda.get_device_properties(device).total_memory, device)
    profile = CompoundReplayRecipe(BASE); aux,_ = profile.tokenizer(device)
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    dataset = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
    indices = json.loads((ROOT/'outputs/imagenet-laser-joint-predictability-20261016/original_rq-final.json').read_text())['splits']['heldout_val']['indices']
    gram = aux.dictionary.T@aux.dictionary
    cases = ['ideal_prefix_refit', 'actual_final_coefficients', 'actual_hard512_context', 'actual_training_noise_context']
    counters = {case:torch.zeros(3, 4, dtype=torch.float64) for case in cases}
    for first in range(0,128,4):
        images = torch.stack([dataset[i][0] for i in indices[first:first+4]]).to(device)
        z = aux.quant_conv(aux.encoder(images)).permute(0,2,3,1).contiguous()
        atoms, values = sparse_encoding(z, aux.dictionary, 4, gram=gram)
        normalized = (values/aux.coeff_scales).clamp(-3,3)
        hard = aux.coeff_bins[nearest_coefficient_ids(normalized, aux.coeff_bins)]*aux.coeff_scales
        p = (-(normalized[...,None]-aux.coeff_bins).square()/.03125).softmax(-1)
        torch.manual_seed(261016449+first)
        sampled = torch.multinomial(p.reshape(-1,512),1).reshape_as(atoms)
        noisy = aux.coeff_bins[sampled]*aux.coeff_scales
        correlations = z@aux.dictionary
        selected = correlations.gather(-1, atoms)
        selected_gram = gram[atoms[..., :, None],atoms[..., None, :]]
        for depth in [1,2,3]:
            prefix = torch.linalg.solve(selected_gram[..., :depth,:depth],selected[..., :depth,None])[...,0]
            for case, coefficients in [('ideal_prefix_refit',prefix),('actual_final_coefficients',values[...,:depth]),
                ('actual_hard512_context',hard[...,:depth]),('actual_training_noise_context',noisy[...,:depth])]:
                residual = correlations-(coefficients[...,None]*gram[atoms[...,:depth]]).sum(-2)
                score = residual.abs(); score.scatter_(-1,atoms[...,:depth],-torch.inf)
                actual = atoms[...,depth]
                actual_score = score.gather(-1,actual[...,None])
                rank = (score > actual_score).sum(-1)+1
                measurements = torch.stack([(rank==1).double().sum(),(rank<=10).double().sum(),
                    (rank<=256).double().sum(),rank.double().sum()]).cpu()
                counters[case][depth-1] += measurements
        write_json(OUT/'prefix-consistency-progress.json',dict(images=first+4,time=time.time()))
    report = dict(validation_images=128,sites_per_depth=8192,next_atom_depths=[2,3,4],
        cases={case:dict(top1_agreement=(values[:,0]/8192).tolist(),top10=(values[:,1]/8192).tolist(),
            top256=(values[:,2]/8192).tolist(),mean_rank=(values[:,3]/8192).tolist()) for case,values in counters.items()},
        source_supports_unchanged=True,training_updates_added=0,
        limitations=['Greedy-prefix consistency is an encoding property, not a requirement for valid autoregressive likelihood.',
            'This does not establish that changing coefficient coordinates improves generated-image FID.'])
    assert min(report['cases']['ideal_prefix_refit']['top1_agreement'])>.999
    write_json(OUT/'prefix-consistency.json',report); print(json.dumps(report),flush=True)


if __name__ == '__main__': main()
