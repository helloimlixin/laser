"""Add identity-initialized recurrent depth passes to the successful Church prior."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.training.rqtransformer import build_model
from archive.scripts.train_church_bar_coefficients import quantized_splits


class GatedDepthLoop(nn.Module):
    """Original four blocks first; gated shared-weight refinement afterwards.

    State-dict names of every pretrained block remain unchanged. Cache state is
    private to each effective layer, including when its weights are shared.
    """
    def __init__(self, blocks, extra_passes=2):
        super().__init__()
        if not blocks or extra_passes < 1:
            raise ValueError('Expected blocks and at least one extra pass')
        self.blocks = nn.ModuleList(blocks)
        self.loop_gates = nn.Parameter(torch.zeros(extra_passes))
        self.init_cache()

    def init_cache(self):
        self._pass_caches = [[{'past_kv': None} for _ in self.blocks]
                             for _ in range(self.loop_gates.numel() + 1)]
        for block in self.blocks:
            block.init_cache()

    def _apply_pass(self, x, index, cached):
        for i, block in enumerate(self.blocks):
            if cached:
                block._cache = self._pass_caches[index][i]
                x = block.cached_forward(x)
            else:
                x = block(x)
        return x

    def _run(self, x, cached):
        x = self._apply_pass(x, 0, cached)
        # Evaluation of the initial checkpoint is exactly the original path.
        if not self.training and not bool(self.loop_gates.detach().count_nonzero()):
            return x
        for i, gate in enumerate(self.loop_gates.tanh()):
            proposal = self._apply_pass(x, i + 1, cached)
            x = x + gate.to(x.dtype) * (proposal - x)
        return x

    def forward(self, x):
        return self._run(x, False)

    def cached_forward(self, x):
        return self._run(x, True)


def source_prior():
    return build_model(18432, 16384, compound=True, coeff_vocab_size=2048,
                       compound_micro_transformer_layers=2,
                       compound_depth_specific_coeff_heads=True,
                       compound_pair_autoregressive=True,
                       sparsity_level=4, model_preset='lsun-church-350m')


def epoch50_prior(checkpoint, variant='looped'):
    if variant not in {'looped', 'control'}:
        raise ValueError('Expected looped or control')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['epoch'] == 50 and saved['global_step'] == 3050
    cfg = saved['config']
    assert cfg['compound_pair_autoregressive'] and cfg['atom_loss_weight'] == 1
    assert cfg['coeff_target_mode'] == 'hard' and cfg['geometry_loss_weight'] == 0
    model = source_prior()
    model.load_state_dict(saved['state_dict'], strict=True)
    if variant == 'looped':
        model.head_transformer = GatedDepthLoop(list(model.head_transformer.blocks))
    return model, {'epoch': saved['epoch'], 'global_step': saved['global_step'],
                   'fid50k': saved['fid'], 'optimizer_restored': False,
                   'config': cfg}


def cached_splits(cache):
    """Recover the original bins; rejoin the full original training population."""
    keys = [set(cache[s]['keys']) for s in ('train', 'holdout', 'validation')]
    assert not any(keys[i] & keys[j] for i in range(3) for j in range(i))
    splits = quantized_splits(cache)
    splits['train_probe'] = splits['holdout']
    splits['train'] = torch.cat((splits['train'], splits.pop('holdout')))
    assert len(splits['train']) == cache['meta']['training_population']
    return splits


def pair_objective(model, aux, packed, amp=True):
    atoms, ids = model.unpack(packed)
    with torch.autocast(packed.device.type, dtype=torch.bfloat16, enabled=amp):
        out = model(packed, model_aux=aux, amp=False)
    atom_nll = F.cross_entropy(out['atom_logits'].float().flatten(0, -2),
                               atoms.flatten(), reduction='none').reshape_as(atoms)
    coefficient_nll = F.cross_entropy(out['coeff_logits'].float().flatten(0, -2),
                                      ids.flatten(), reduction='none').reshape_as(ids)
    loss = .5 * (atom_nll.mean() + coefficient_nll.mean())
    with torch.no_grad():
        probabilities = out['coeff_logits'].float().softmax(-1)
        half = model.coeff_vocab_size // 2
        signs = probabilities[..., half:].sum(-1) >= .5
        physical_bins = aux.coeff_bins * aux.coeff_scales[..., None]
        predicted = (probabilities * physical_bins).sum(-1)
        truth = aux.coeff_bins[ids] * aux.coeff_scales
        metrics = {'loss': float(loss.detach()), 'atom_nll': float(atom_nll.mean()),
                   'coefficient_nll': float(coefficient_nll.mean()),
                   'sign_accuracy': float((signs == (ids >= half)).float().mean()),
                   'coefficient_mean_mae': float((predicted - truth).abs().mean())}
        for d in range(4):
            metrics[f'atom_nll_d{d}'] = float(atom_nll[..., d].mean())
            metrics[f'coefficient_nll_d{d}'] = float(coefficient_nll[..., d].mean())
    return loss, metrics


def continuation_lr(step, total_steps, peak=1e-5, minimum=1e-6, warmup_steps=8):
    if not (0 < minimum <= peak and 0 <= warmup_steps < total_steps):
        raise ValueError('Invalid continuation schedule')
    if warmup_steps and step <= warmup_steps:
        return peak * max(.01, step / warmup_steps)
    fraction = min(1., max(0., (step - warmup_steps) / (total_steps - warmup_steps)))
    return minimum + .5 * (peak - minimum) * (1 + math.cos(math.pi * fraction))
