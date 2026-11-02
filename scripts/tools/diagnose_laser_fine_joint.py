#!/usr/bin/env python3
"""Read-only epoch-one diagnosis: conditional likelihood and sampler interventions."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

BASE = Path('/tmp/laser-rq-fine-joint512-imagenet-20261015')
ROOT = Path('/workspace/Projects/laser')
OUT = ROOT/'outputs/imagenet-laser-fine-joint512-scratch6-4h100-20261015/diagnosis-epoch001'
os.environ.update(json.loads((BASE/'environment.json').read_text()))
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
sys.path[:0] = [str(BASE/'source'), str(BASE/'runtime'), str(BASE/'upstream'), str(BASE)]
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from src import rq_upstream
from src.models.laser_fine_joint_adapter import exclude_previous_atoms
from src.orthogonal_sparse_codec import ordered_support_basis
from src.training.laser_fine_joint_recipe import FineJointLaserRecipe
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from src.training import k4_checkpoint_io as checkpoint_io
from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.img_datasets.transforms import create_transforms


def per_depth(value):
    return value.reshape(-1, 4).double().mean(0).cpu().tolist()


def latent_statistics(aux, codes):
    atoms = codes//aux.coeff_vocab_size
    gamma = aux.coeff_bins[codes%aux.coeff_vocab_size]*aux.coeff_scales
    basis, lower = ordered_support_basis(aux.dictionary.T[atoms])
    z = (basis*gamma[..., None]).sum(-2)
    normalized = F.normalize(z, dim=-1)
    cosine = torch.cat(((normalized[:, 1:]*normalized[:, :-1]).sum(-1).flatten(),
                         (normalized[:, :, 1:]*normalized[:, :, :-1]).sum(-1).flatten()))
    return dict(gamma_abs=per_depth(gamma.abs()), gamma_squared=per_depth(gamma.square()),
        gamma_positive=per_depth((gamma>0).float()),
        basis_innovation=per_depth(lower.diagonal(dim1=-2, dim2=-1)),
        latent_squared_mean=float(z.square().mean()), latent_mean=float(z.mean()),
        neighbor_cosine=float(cosine.mean()),
        latent_channel_mean=z.mean((0, 1, 2)).cpu().tolist(),
        latent_channel_std=z.std((0, 1, 2)).cpu().tolist())


def pixel_statistics(images):
    return dict(mean=float(images.mean()), std=float(images.std()),
        saturation_fraction=float((images.abs()>.98).float().mean()),
        horizontal_variation=float((images[:, :, :, 1:]-images[:, :, :, :-1]).abs().mean()),
        vertical_variation=float((images[:, :, 1:]-images[:, :, :-1]).abs().mean()))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.20, 0)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    profile = FineJointLaserRecipe(BASE)
    checkpoint = ROOT/'outputs'/profile.run_id/'checkpoints/last.pt'
    path = checkpoint_io._checkpoint_upload_source(checkpoint.resolve())
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert saved['global_step'] == 626
    profile.verify_resume(saved)
    config = load_config(BASE/'inputs/stage2-training-config.yaml')
    arch = augment_arch_defaults(config.arch); arch.vocab_size = config.dataset.vocab_size
    with torch.device('meta'): model = profile.model(arch)
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    model.eval().requires_grad_(False).to(device)
    del saved
    aux, _ = profile.tokenizer(device)
    head = model.classifier.linear
    dataset = TarImageNet(BASE/'data', 'val', create_transforms(config.dataset, split='val'))
    indices = np.random.default_rng(261015).choice(len(dataset), 128, replace=False).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=4, num_workers=4, pin_memory=True)
    torch.manual_seed(261015); torch.cuda.manual_seed(261015)
    sums = {}; count = 0
    hard_codes = []; image_batches = []; hard_recon = []; predicted_recon = []; oracle_sign_recon = []
    def accumulate(name, value):
        sums[name] = sums.get(name, torch.zeros(4, dtype=torch.float64)) + value.reshape(-1, 4).double().sum(0).cpu()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        x = aux.encode(images)
        codes, gamma = aux.latent_codes(x)
        atoms = codes//aux.coeff_vocab_size
        pvalues = aux.coeff_bins[None, :]*aux.coeff_scales[:, None]
        targets = (-(gamma[..., None]-pvalues).square()/(2*.075**2)).softmax(-1)
        features = model(codes, aux, labels, amp=True)
        atom_logp = exclude_previous_atoms(features[..., :aux.num_atoms], atoms).float().log_softmax(-1)
        coefficient_logits = head.coefficient_logits(features, atoms).float()
        logp = coefficient_logits.log_softmax(-1); p = logp.exp()
        shuffled_atoms = atoms.flatten().roll(773).reshape_as(atoms)
        shuffled_logp = head.coefficient_logits(features, shuffled_atoms).float().log_softmax(-1)
        uniform_context = features.clone()
        uniform_context[..., aux.num_atoms:] = features[..., aux.num_atoms:].mean((0, 1, 2), keepdim=True)
        contextless_logp = head.coefficient_logits(uniform_context, atoms).float().log_softmax(-1)
        accumulate('atom_ce', -atom_logp.gather(-1, atoms[..., None]).squeeze(-1))
        accumulate('atom_accuracy', (atom_logp.argmax(-1)==atoms).float())
        accumulate('atom_entropy', -(atom_logp.exp()*atom_logp.nan_to_num()).sum(-1))
        accumulate('coefficient_soft_ce', -(targets*logp).sum(-1))
        accumulate('coefficient_hard_ce', -logp.gather(-1, (codes%aux.coeff_vocab_size)[..., None]).squeeze(-1))
        accumulate('coefficient_target_entropy', -(targets*targets.clamp_min(1e-30).log()).sum(-1))
        accumulate('coefficient_predicted_entropy', -(p*logp).sum(-1))
        accumulate('coefficient_shuffled_atom_soft_ce', -(targets*shuffled_logp).sum(-1))
        accumulate('coefficient_mean_context_soft_ce', -(targets*contextless_logp).sum(-1))
        accumulate('coefficient_correct_vs_shuffled_atom_kl', (p*(logp-shuffled_logp)).sum(-1))
        pred_mean = (p*pvalues).sum(-1)
        pred_variance = (p*(pvalues-pred_mean[..., None]).square()).sum(-1)
        accumulate('coefficient_expected_squared_error', pred_variance+(pred_mean-gamma).square())
        accumulate('coefficient_mean_squared_error', (pred_mean-gamma).square())
        accumulate('coefficient_target_squared', gamma.square())
        sign_probability = (p*((pvalues>0)==(gamma[..., None]>0))).sum(-1)
        accumulate('coefficient_correct_sign_probability', sign_probability)
        correct_sign = ((pvalues>0)==(gamma[..., None]>0))
        # Flip the sign of every wrong-sign sample, preserving its magnitude
        # distribution exactly. Conditioning on the correct half would also
        # change magnitudes and would not isolate sign alone.
        sign_oracle_p = (p+p.flip(-1))*correct_sign
        sign_oracle_p = sign_oracle_p/sign_oracle_p.sum(-1, keepdim=True).clamp_min(1e-30)
        torch.testing.assert_close((sign_oracle_p*pvalues.abs()).sum(-1),
                                   (p*pvalues.abs()).sum(-1), rtol=2e-5, atol=1e-5)
        accumulate('oracle_sign_coefficient_expected_squared_error',
                   (sign_oracle_p*(pvalues-gamma[..., None]).square()).sum(-1))
        accumulate('oracle_magnitude_coefficient_expected_squared_error',
                   (p*(pvalues.sign()*gamma.abs()[..., None]-gamma[..., None]).square()).sum(-1))
        accumulate('coefficient_predicted_abs', (p*pvalues.abs()).sum(-1))
        context = features[..., aux.num_atoms:].float()
        accumulate('context_rms', context.square().mean(-1).sqrt())
        accumulate('atom_embedding_rms', head.atom_context(atoms).square().mean(-1).sqrt())
        sample_bins = torch.multinomial(p.reshape(-1, aux.coeff_vocab_size), 1).reshape_as(codes)
        predicted_codes = atoms*aux.coeff_vocab_size+sample_bins
        basis, _ = ordered_support_basis(aux.dictionary.T[atoms])
        zhard = aux.get_code_emb_with_depth(codes)[0].sum(-2)
        zmean = (basis*pred_mean[..., None]).sum(-2)
        # Expected latent MSE with ground-truth atoms / prefixes isolates coefficients.
        sums['hard_latent_error'] = sums.get('hard_latent_error', 0.) + float((x-zhard).square().sum())
        sums['predicted_coefficient_latent_error'] = sums.get('predicted_coefficient_latent_error', 0.) + float((x-zmean).square().sum()+pred_variance.sum())
        sign_mean = (sign_oracle_p*pvalues).sum(-1)
        sign_variance = (sign_oracle_p*(pvalues-sign_mean[..., None]).square()).sum(-1)
        sign_latent = (basis*sign_mean[..., None]).sum(-2)
        sums['oracle_sign_latent_error'] = sums.get('oracle_sign_latent_error', 0.) + float(
            (x-sign_latent).square().sum()+sign_variance.sum())
        magnitude_values = pvalues.sign()*gamma.abs()[..., None]
        magnitude_mean = (p*magnitude_values).sum(-1)
        magnitude_variance = (p*(magnitude_values-magnitude_mean[..., None]).square()).sum(-1)
        magnitude_latent = (basis*magnitude_mean[..., None]).sum(-2)
        sums['oracle_magnitude_latent_error'] = sums.get('oracle_magnitude_latent_error', 0.) + float(
            (x-magnitude_latent).square().sum()+magnitude_variance.sum())
        hard_codes.append(codes.cpu()); image_batches.append(images.cpu())
        hard_recon.append(aux.decode_code(codes).cpu())
        predicted_recon.append(aux.decode_code(predicted_codes).cpu())
        sign_bins = torch.multinomial(sign_oracle_p.reshape(-1, aux.coeff_vocab_size), 1).reshape_as(codes)
        oracle_sign_recon.append(aux.decode_code(atoms*aux.coeff_vocab_size+sign_bins).cpu())
        count += len(images)*64
        write_json(OUT/'progress.json', dict(phase='teacher_forcing', images=count//64, time=time.time()))
    result = {k:(v/count).tolist() if torch.is_tensor(v) else v/(count*256) for k, v in sums.items()}
    result.update(images=128, checkpoint_global_step=626, diagnostic_only=True,
        validation_indices=indices,
        source=latent_statistics(aux, torch.cat(hard_codes).to(device)),
        real_images=pixel_statistics(torch.cat(image_batches)),
        hard_reconstruction=pixel_statistics(torch.cat(hard_recon)),
        true_atoms_predicted_coefficients=pixel_statistics(torch.cat(predicted_recon)))
    result['oracle_sign_expected_latent_mse'] = result.pop('oracle_sign_latent_error')
    result['oracle_magnitude_expected_latent_mse'] = result.pop('oracle_magnitude_latent_error')
    save_image(torch.cat(image_batches)[:16].mul(.5).add(.5).clamp(0, 1), OUT/'real.png', nrow=4)
    save_image(torch.cat(hard_recon)[:16].mul(.5).add(.5).clamp(0, 1), OUT/'hard-reconstruction.png', nrow=4)
    save_image(torch.cat(predicted_recon)[:16].mul(.5).add(.5).clamp(0, 1), OUT/'true-atoms-predicted-coefficients.png', nrow=4)
    save_image(torch.cat(oracle_sign_recon)[:16].mul(.5).add(.5).clamp(0, 1), OUT/'true-atoms-oracle-sign.png', nrow=4)
    write_json(OUT/'teacher-forcing.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source','validation_indices']}), flush=True)
    if args.teacher_only:
        return
    del features, atom_logp, coefficient_logits, p, logp, targets, uniform_context
    torch.cuda.empty_cache()
    original = dict(profile.recipe.sampling)
    variants = dict(current=dict(atom_top_k=2048, atom_top_p=1., coefficient_temperature=1.),
        rq_atom_truncation=dict(atom_top_k=256, atom_top_p=.95, coefficient_temperature=1.),
        rq_atom_truncation_coefficient_t08=dict(atom_top_k=256, atom_top_p=.95, coefficient_temperature=.8))
    for name, options in variants.items():
        profile.recipe.sampling.update(original); profile.recipe.sampling.update(options)
        torch.manual_seed(261016); torch.cuda.manual_seed(261016)
        codes = []; decoded = []
        for first in range(0, 128, 32):
            labels = torch.arange(first, first+32, device=device)*7%1000
            c = profile.sample(model, aux, 32, labels)
            codes.append(c.cpu())
            for chunk in c.split(4): decoded.append(profile.decode(aux, chunk).cpu())
            write_json(OUT/'progress.json', dict(phase='sampling', variant=name, images=first+32, time=time.time()))
        codes = torch.cat(codes); decoded = torch.cat(decoded)
        report = dict(diagnostic_only=True, generated_images=128, sampling=options,
            latent=latent_statistics(aux, codes.to(device)), pixels=pixel_statistics(decoded))
        write_json(OUT/(name+'.json'), report)
        torch.save(codes, OUT/(name+'-codes.pt'))
        save_image(decoded[:16].mul(.5).add(.5).clamp(0, 1), OUT/(name+'.png'), nrow=4)
        print(json.dumps(dict(variant=name, pixels=report['pixels'],
            latent={k:v for k,v in report['latent'].items() if not k.startswith('latent_channel')})), flush=True)
    write_json(OUT/'completed.json', dict(completed=True,checkpoint_global_step=626,time=time.time(),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3, diagnostic_only=True,
        production_training_unchanged=True, generated_images_per_variant=128, fid_not_recomputed=True))


if __name__ == '__main__': main()
