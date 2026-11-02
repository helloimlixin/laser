#!/usr/bin/env python3
"""Compare epoch-one coefficient sign conditioning on identical held-out images."""
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
FINE = Path('/tmp/laser-rq-fine-joint512-imagenet-20261015')
OUT = ROOT/'outputs/imagenet-laser-fine-joint512-scratch6-4h100-20261015/diagnosis-epoch001'
os.environ['RQ_UPSTREAM_ROOT'] = str(FINE/'upstream')
sys.path[:0] = [str(FINE/'source'), str(FINE/'runtime'), str(FINE/'upstream'), str(FINE)]
import torch
from torch.utils.data import DataLoader, Subset
from src import rq_upstream
from src.training.laser_joint_recipe import JointLaserRecipe
from src.training.rq_imagenet_recipes import LaserRecipe
from src.training.rq_imagenet_runtime import TarImageNet, write_json
from rqvae.utils.config import load_config, augment_arch_defaults
from rqvae.img_datasets.transforms import create_transforms


@torch.inference_mode()
def main():
    torch.set_num_threads(4); torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.20, 0)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    indices = json.loads((OUT/'teacher-forcing.json').read_text())['validation_indices']
    reports = {}
    for kind, base, cls in [
        ('coarse_joint', Path('/tmp/laser-rq-native-joint-imagenet-20261015'), JointLaserRecipe),
        ('scalar', Path('/tmp/laser-rq-clean-scratch-imagenet-20261015'), LaserRecipe)]:
        saved = None; checkpoint = None
        for receipt in (base/'checkpoint-cache/objects').glob('*.json'):
            metadata = json.loads(receipt.read_text())
            if '/smoke/' in metadata['source']: continue
            candidate = torch.load(receipt.with_suffix('.pt'), map_location='cpu', weights_only=False, mmap=True)
            if candidate['global_step'] == 626:
                saved, checkpoint = candidate, receipt.with_suffix('.pt'); break
            del candidate
        assert saved is not None
        profile = cls(base); profile.verify_resume(saved)
        config = load_config(base/'inputs/stage2-training-config.yaml')
        arch = augment_arch_defaults(config.arch); arch.vocab_size = config.dataset.vocab_size
        with torch.device('meta'): model = profile.model(arch)
        model.load_state_dict(saved['state_dict'], strict=True, assign=True)
        model.eval().requires_grad_(False).to(device)
        aux, _ = profile.tokenizer(device); profile.adapt(model, aux)
        del saved
        dataset = TarImageNet(base/'data', 'val', create_transforms(config.dataset, split='val'))
        loader = DataLoader(Subset(dataset, indices), batch_size=4, num_workers=4, pin_memory=True)
        sign_sum = torch.zeros(4, dtype=torch.float64); atom_ce = torch.zeros(4, dtype=torch.float64)
        coefficient_entropy = torch.zeros(4, dtype=torch.float64)
        denominator = 0; first_fine_sign = 0.
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if kind == 'coarse_joint':
                q = aux.quantizer
                x = aux.encode(images)
                codes = q.quantize(x, chunk_size=128)['codes']
                output = model(codes, aux, labels, amp=True).float()
                joint_p = output.softmax(-1)
                for d in range(4):
                    atoms = q.atom_ids[d, codes[..., d]]
                    candidate_ids = 1+atoms[..., None]*4+torch.arange(4, device=device)
                    assert torch.equal(q.atom_ids[d, candidate_ids], atoms[..., None].expand_as(candidate_ids))
                    p = output[..., d, :].gather(-1, candidate_ids).softmax(-1)
                    values = q.coefficients[d, candidate_ids]
                    correct = (values>0)==(q.coefficients[d, codes[..., d]][..., None]>0)
                    sign_sum[d] += (p*correct).sum(-1).double().sum().cpu()
                    atom_mass = joint_p[..., d, :].gather(-1, candidate_ids).sum(-1)
                    atom_ce[d] += -atom_mass.clamp_min(1e-30).log().double().sum().cpu()
                    coefficient_entropy[d] += -(p*p.clamp_min(1e-30).log()).sum(-1).double().sum().cpu()
                    if d == 0:
                        correlations = x@q.dictionary
                        source_atoms = correlations.abs().argmax(-1)
                        source_sign = correlations.gather(-1, source_atoms[..., None])>0
                        ids = 1+source_atoms[..., None]*4+torch.arange(4, device=device)
                        source_p = output[..., 0, :].gather(-1, ids).softmax(-1)
                        first_fine_sign += float((source_p*((q.coefficients[0, ids]>0)==source_sign)).sum())
            else:
                targets, codes = profile.targets(aux, images)
                output = model(codes, aux, labels, amp=True)
                p = output['coeff_logits'].float().softmax(-1)
                ids = codes[..., 1::2]-aux.num_atoms
                values = aux.coeff_bins[None, :]*aux.coeff_scales[:, None]
                true = aux.coeff_bins[ids]*aux.coeff_scales
                correct = (values>0)==(true[..., None]>0)
                sign_sum += (p*correct).sum(-1).reshape(-1, 4).double().sum(0).cpu()
                coefficient_entropy += -(p*p.clamp_min(1e-30).log()).sum(-1).reshape(-1, 4).double().sum(0).cpu()
                logp = output['atom_logits'].float().log_softmax(-1)
                atom_ce += -logp.gather(-1, codes[..., 0::2, None]).squeeze(-1).reshape(-1, 4).double().sum(0).cpu()
            denominator += len(images)*64
            write_json(OUT/'sign-reference-progress.json', dict(reference=kind, images=denominator//64,time=time.time()))
        reports[kind] = dict(checkpoint_global_step=626, checkpoint=str(checkpoint), validation_images=128,
            correct_sign_probability=(sign_sum/denominator).tolist(),
            atom_ce=(atom_ce/denominator).tolist(), coefficient_entropy=(coefficient_entropy/denominator).tolist(),
            different_tokenizations=kind=='coarse_joint', same_validation_images=True)
        if kind == 'coarse_joint': reports[kind]['first_depth_sign_conditioned_on_fine_source_atom'] = first_fine_sign/denominator
        write_json(OUT/'earlier-head-sign-comparison.json', reports)
        print(json.dumps({kind:reports[kind]}), flush=True)
        del model, aux, profile, loader, output, p
        gc.collect(); torch.cuda.empty_cache()
    reports['fine512'] = dict(correct_sign_probability=json.loads((OUT/'teacher-forcing.json').read_text())['coefficient_correct_sign_probability'],checkpoint_global_step=626)
    write_json(OUT/'earlier-head-sign-comparison.json', reports)


if __name__ == '__main__': main()
