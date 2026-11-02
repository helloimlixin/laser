"""Measure atom/coefficient gradients at the shared decoder representation."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--ordered-paths', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    sys.path[:0] = [str(args.base/'source'), str(args.base/'source/runtime')]
    import torch
    from torch.nn import functional as F
    from PIL import Image
    from torchvision import transforms
    from src.training.rqtransformer import LaserAux
    from src.training.imagenet_ffhq_adapter import build_model

    started = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(261011)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(
        24*2**30/torch.cuda.get_device_properties(device).total_memory, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    payload = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    config = payload['config']
    step, epoch = int(payload['global_step']), int(payload['epoch'])
    with torch.device('meta'):
        model = build_model(18432, 16384, compound=True, coeff_vocab_size=2048,
            sparsity_level=4, compound_micro_transformer_layers=2,
            compound_depth_specific_coeff_heads=True, compound_pair_autoregressive=True,
            physical_pair_context=False, model_preset='imagenet-1400m')
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    del payload
    model.to(device).requires_grad_(False).train()
    aux = LaserAux(args.base/'inputs/stage1-tokenizer.pt', 16384, 2048, 3.,
        coeff_scales=config['coeff_scales'], soft_target_physical=False,
        clamp_coeffs=True, sparsity_level=4).to(device).eval().requires_grad_(False)
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    gram = aux.dictionary.float().t() @ aux.dictionary.float()

    # Reservoir sampling avoids a second million-file directory scan.
    classes = set(round(i*999/31) for i in range(32))
    choices = {c: [] for c in classes}
    counts = {c: 0 for c in classes}
    rng = random.Random(261011)
    with gzip.open(args.ordered_paths, 'rt') as stream:
        for index, line in enumerate(stream):
            relative, label = line.rstrip('\n').rsplit('\t', 1)
            label = int(label)
            if label not in classes:
                continue
            counts[label] += 1
            row = (relative, label, index)
            if len(choices[label]) < 2:
                choices[label].append(row)
            else:
                replacement = rng.randrange(counts[label])
                if replacement < 2:
                    choices[label][replacement] = row
    samples = [row for label in sorted(classes) for row in choices[label]]
    assert len(samples) == 64
    rng.shuffle(samples)
    transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize([.5]*3, [.5]*3)])
    capture = {}
    original = model.classify_head_outputs

    def head_outputs(self, hidden):
        # The backbone remains frozen; both heads differentiate the same h.
        # All classifier/refiner parameters keep their checkpoint values.
        with torch.enable_grad():
            shared = hidden.detach().requires_grad_(True)
            logits = original(shared)
        capture.update(shared=shared, logits=logits)
        return logits

    model.classify_head_outputs = types.MethodType(head_outputs, model)
    rows = []
    for offset in range(0, len(samples), 4):
        chunk = samples[offset:offset+4]
        image_rows = []
        for relative, label, index in chunk:
            seed = (261001 + 1000003*epoch + 1000000007*index) % (2**63-1)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                with Image.open(args.base/'imagenet/train'/relative) as image:
                    image_rows.append(transform(image.convert('RGB')))
        images = torch.stack(image_rows).to(device)
        labels = torch.tensor([r[1] for r in chunk], device=device)
        with torch.no_grad():
            atoms, clean = aux.encode_sparse_components(images, dictionary_gram=gram)
            ids, targets = aux.compound_coeff_ids(clean, stochastic=True,
                temp=config['coeff_target_temperature'])
            packed = atoms*2048 + ids
            with torch.autocast('cuda', dtype=torch.bfloat16):
                model(packed, model_aux=aux, cond=labels, amp=False)
        logits = capture['logits']
        atom_ce = F.cross_entropy(logits['atom_logits'].float().reshape(-1, 16384), atoms.reshape(-1))
        coeff_ce = -(targets*F.log_softmax(logits['coeff_logits'].float(), dim=-1)).sum(-1).mean()
        entropy = -(targets*targets.clamp_min(1e-30).log()).sum(-1).mean()
        atom_grad = torch.autograd.grad(atom_ce, capture['shared'], retain_graph=True)[0].float()
        coeff_grad = torch.autograd.grad(coeff_ce, capture['shared'])[0].float()
        atom_norm, coeff_norm = float(atom_grad.norm()), float(coeff_grad.norm())
        ratio = coeff_norm/atom_norm
        row = dict(images=4, atom_ce=float(atom_ce.detach()), coefficient_ce=float(coeff_ce.detach()),
            coefficient_target_entropy=float(entropy), coefficient_kl=float(coeff_ce.detach()-entropy),
            atom_shared_gradient_norm=atom_norm, coefficient_shared_gradient_norm=coeff_norm,
            coefficient_to_atom_gradient_ratio=ratio,
            old_weighted_atom_to_coefficient_gradient_ratio=1.5/ratio,
            gradient_cosine=float((atom_grad*coeff_grad).sum()/(atom_grad.norm()*coeff_grad.norm())))
        assert all(math.isfinite(v) for v in row.values()) and atom_norm > 0 and coeff_norm > 0
        rows.append(row)
        print(json.dumps(dict(phase='loss_balance_calibration', images_completed=offset+4, **row)), flush=True)
        capture.clear()
        del logits, atom_grad, coeff_grad, images, targets, packed, atom_ce, coeff_ce

    ratio = statistics.median(r['coefficient_to_atom_gradient_ratio'] for r in rows)
    # Bound the objective shift to a fourfold coefficient/atom ratio.
    atom_weight = max(.25, min(1., ratio))
    report = dict(passed=True, checkpoint=str(args.checkpoint), global_step=step, epoch=epoch,
        images=64, classes=32, batches=16, seed=261011, rows=rows,
        balancing_representation='shared head outputs before atom classifier and atom-conditioned coefficient refiner',
        full_backbone_parameter_gradients_measured=False,
        median_coefficient_to_atom_gradient_ratio=ratio,
        recommended_atom_weight=atom_weight, recommended_coefficient_weight=1.,
        recommended_geometry_weight=0., coefficient_to_atom_weight_cap=4.,
        old_atom_weight=1.5, old_coefficient_weight=1., old_geometry_weight=.05,
        new_weighted_atom_to_coefficient_gradient_ratio=atom_weight/ratio,
        checkpoint_parameters_modified=False, production_modified=False, wandb_logged=False,
        ordered_paths_sha256=hashlib.sha256(args.ordered_paths.read_bytes()).hexdigest(),
        elapsed_seconds=time.time()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}), flush=True)


if __name__ == '__main__':
    main()
