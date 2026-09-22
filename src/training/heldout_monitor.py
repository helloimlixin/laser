"""Fixed-history, distributed diagnostics for compound sparse-code priors."""
from contextlib import nullcontext
import math

import torch
import torch.distributed as dist


@torch.inference_mode()
def evaluate_heldout(model, aux, datasets, *, batch_size=4, temperature=.125):
    """Measure per-image losses without changing training modes or RNG streams.

    Every rank must call this function with the same ordered datasets. Strided
    shards visit each image exactly once, including non-divisible split sizes.
    Coefficient histories are fixed in the supplied probe; target probabilities
    use the active objective. Neither training nor validation images are fitted.
    """
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    device = next(model.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    flags = [(module, module.training) for module in model.modules()]
    devices = [device.index] if device.type == 'cuda' else []
    result = {}
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            for split, data in datasets.items():
                count = len(data['atoms'])
                depth = data['atoms'].shape[-1]
                if count < 2 or any(len(data[key]) != count for key in
                                    ['coefficients', 'input_coefficient_ids']):
                    raise ValueError('probe fields must contain matching image counts >= 2')
                # Rows: atom NLL, coefficient CE, coefficient entropy, coefficient KL.
                # Final column also accumulates each image's depth-averaged loss.
                sums = torch.zeros(4, depth+1, device=device, dtype=torch.float64)
                squares = torch.zeros_like(sums)
                visits = torch.zeros((), device=device, dtype=torch.float64)
                indices = list(range(rank, count, world))
                for offset in range(0, len(indices), batch_size):
                    ids = indices[offset:offset+batch_size]
                    atoms = data['atoms'][ids].to(device=device, dtype=torch.long)
                    coeffs = data['coefficients'][ids].to(device=device, dtype=torch.float32)
                    history = data['input_coefficient_ids'][ids].to(device=device, dtype=torch.long)
                    _, targets = aux.compound_coeff_ids(coeffs, stochastic=False, temp=temperature)
                    context = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
                    with context:
                        logits = model(atoms * aux.coeff_vocab_size + history, model_aux=aux, amp=False)
                    atom = -logits['atom_logits'].float().log_softmax(-1).gather(-1, atoms[..., None]).squeeze(-1)
                    ce = -(targets * logits['coeff_logits'].float().log_softmax(-1)).sum(-1)
                    entropy = -(targets * targets.clamp_min(1e-30).log()).sum(-1)
                    values = torch.stack([atom, ce, entropy, ce-entropy]).mean((2, 3)).double()
                    if not torch.isfinite(values).all():
                        raise ValueError('non-finite held-out loss')
                    values = torch.cat([values, values.mean(-1, keepdim=True)], dim=-1)
                    sums += values.sum(1)
                    squares += values.square().sum(1)
                    visits += len(ids)
                if world > 1:
                    for tensor in (sums, squares, visits):
                        dist.all_reduce(tensor)
                if int(visits.item()) != count:
                    raise RuntimeError('held-out sharding did not visit every image once')
                means = sums / count
                variance = ((squares - sums.square()/count) / (count-1)).clamp_min(0)
                errors = (variance/count).sqrt()
                result[split] = dict(images=count)
                for row, metric in enumerate(['atom_nll', 'coeff_cross_entropy', 'coeff_target_entropy', 'coeff_kl']):
                    result[split][metric] = dict(mean=float(means[row, -1]), se=float(errors[row, -1]),
                        mean_by_depth=means[row, :-1].tolist(), se_by_depth=errors[row, :-1].tolist())
    finally:
        for module, flag in flags:
            module.training = flag
    if 'train_fresh' in result and 'validation_fresh' in result:
        result['validation_minus_train'] = {}
        for metric in ['atom_nll', 'coeff_kl']:
            train, val = result['train_fresh'][metric], result['validation_fresh'][metric]
            result['validation_minus_train'][metric] = dict(mean=val['mean']-train['mean'],
                se=math.hypot(val['se'], train['se']))
    return result
