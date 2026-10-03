"""Audit fresh training views against the preserved K4 coefficient codec."""
import json
from pathlib import Path
import sys

import torch

BASE = Path('/tmp/laser-imagenet-epoch6-aug-stage2')
DATA = Path('/tmp/laser-imagenet-stage2/imagenet')
EVIDENCE = Path('/workspace/Projects/laser/outputs/imagenet-rfid421-epoch6-augcosine-5h200-20261002')
sys.path[:0] = [str(BASE / 'runtime'), str(BASE / 'runtime/runtime')]
from src.training.fresh_images import EpochImageFolder
from src.training.rqtransformer import LaserAux, image_transform


@torch.inference_mode()
def main():
    ready = json.loads((DATA / 'training-ready.json').read_text())
    assert ready['training_images'] == 1281167 and ready['classes'] == 1000
    torch.manual_seed(261004)
    torch.set_num_threads(8)
    device = torch.device('cuda:0')
    dataset = EpochImageFolder(DATA / 'train', transform=image_transform(), augmentation_seed=261001)
    assert len(dataset) == 1281167 and len(dataset.classes) == 1000
    indices = torch.randperm(len(dataset))[:512].tolist()
    dataset.set_epoch(6)
    first = [dataset[i][0] for i in indices[:16]]
    repeated = [dataset[i][0] for i in indices[:16]]
    assert all(torch.equal(a,b) for a,b in zip(first,repeated))
    dataset.set_epoch(7)
    changed = sum(not torch.equal(a,dataset[i][0]) for a,i in zip(first,indices[:16]))
    assert changed > 0
    dataset.set_epoch(6)
    aux = LaserAux(Path('/tmp/laser-imagenet-stage2/best_rfid_slot1_model.pt'),
        16384, 2048, 3., coeff_scale=6.4,
        coeff_scales=[8.203365325927734,4.265638828277588,3.0662174224853516,1.8273425102233887],
        sparsity_level=4, soft_target_physical=False, clamp_coeffs=False).to(device)
    assert not aux.training and not any(p.requires_grad for p in aux.parameters())
    atoms_out, coeffs_out = [], []
    for offset in range(0,len(indices),16):
        images = torch.stack([dataset[i][0] for i in indices[offset:offset+16]]).to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            atoms,coeffs = aux.encode_sparse_components(images)
        if offset == 0:
            # Recomputing the identical frozen Gram matrix must produce the
            # same discrete support and FP32 coefficient values.
            del aux._frozen_dictionary_gram
            with torch.autocast('cuda',dtype=torch.bfloat16):
                checked_atoms,checked_coeffs = aux.encode_sparse_components(images)
            assert torch.equal(atoms,checked_atoms)
            torch.testing.assert_close(coeffs,checked_coeffs,rtol=0,atol=0)
        assert torch.isfinite(coeffs).all()
        assert not (atoms.sort(-1).values[...,1:] == atoms.sort(-1).values[...,:-1]).any()
        atoms_out.append(atoms.cpu());coeffs_out.append(coeffs.cpu())
    atoms = torch.cat(atoms_out).to(device)
    coeffs = torch.cat(coeffs_out).to(device)
    bin_step = float(aux.coeff_bins[1] - aux.coeff_bins[0])
    continuous_energy = hard_error_energy = soft_error_energy = 0.
    entropy_sum = torch.zeros(4,device=device)
    noise_squared = torch.zeros(4,device=device)
    boundary_sum = torch.zeros(4,device=device)
    for offset in range(0,len(atoms),16):
        a,c = atoms[offset:offset+16],coeffs[offset:offset+16]
        tokens,(_,probabilities) = aux.sparse_targets(a,c,temp=.01125,stochastic=True,compact=True)
        soft_coeffs = aux.coeff_bins[tokens[...,1::2]-16384]
        hard_coeffs = aux.coeff_bins[(c[...,None]-aux.coeff_bins).abs().argmin(-1)]
        support = aux.dictionary.t()[a]
        physical = c*aux.coeff_scales
        continuous = (support*physical[...,None]).sum(-2)
        hard_noise = (support*((hard_coeffs-c)*aux.coeff_scales)[...,None]).sum(-2)
        soft_noise = (support*((soft_coeffs-c)*aux.coeff_scales)[...,None]).sum(-2)
        continuous_energy += float(continuous.square().sum())
        hard_error_energy += float(hard_noise.square().sum())
        soft_error_energy += float(soft_noise.square().sum())
        entropy_sum += (-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1)).sum((0,1,2))
        noise_squared += ((soft_coeffs-c)*aux.coeff_scales).square().sum((0,1,2))
        boundary_sum += torch.isin(tokens[...,1::2]-16384,torch.tensor([0,2047],device=device)).sum((0,1,2))
    sites = len(atoms)*64
    outside = (coeffs.abs()>3).float().mean((0,1,2))
    report = dict(samples=len(atoms),seed=261004,epoch=6,training_ready=ready,
        train_transform='Resize256 RandomCrop256 RandomHorizontalFlip(p=0.5)',
        views_repeat_within_epoch=True,changed_views_next_epoch_of_16=changed,
        frozen_encoder=True,encoder_precision='BF16',omp_precision='FP32',
        gram_reuse_exact_parity=True,coeff_max=3.,coeff_vocab_size=2048,coeff_target_temperature=.01125,
        coefficient_scales=aux.coeff_scales.tolist(), normalized_bin_step=bin_step,
        finite=True,unique_atom_supports=True,
        max_abs_normalized_coefficient_per_depth=coeffs.abs().amax((0,1,2)).tolist(),
        coefficients_outside_grid_fraction_per_depth=outside.tolist(),
        target_entropy_nats_per_depth=(entropy_sum/sites).tolist(),
        sampled_physical_noise_rms_per_depth=(noise_squared/sites).sqrt().tolist(),
        sampled_noise_rms_bins_per_depth=((noise_squared/sites).sqrt()/aux.coeff_scales/bin_step).tolist(),
        sampled_boundary_fraction_per_depth=(boundary_sum/sites).tolist(),
        hard_quantization_latent_energy_fraction=hard_error_energy/continuous_energy,
        stochastic_coefficient_noise_latent_energy_fraction=soft_error_energy/continuous_energy,
        prior_cached_view_noise_energy_fraction=.006891967263072729)
    report['acceptable'] = (bool(outside.max()<.001)
        and report['hard_quantization_latent_energy_fraction']<.0001
        and report['stochastic_coefficient_noise_latent_energy_fraction']<.01)
    for path in (BASE/'fresh-augmentation-audit.json',EVIDENCE/'fresh-augmentation-audit.json'):
        path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    assert report['acceptable'], 'Fresh-view coefficient range/noise audit requires review'


if __name__ == '__main__':
    main()
