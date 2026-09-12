"""Church support-first prior with one calibrated joint coefficient decision."""
import torch

from src.training.rqtransformer import SupportFirstLaserRQTransformer, support_first_objective
from src.church_ffhq_recipe import recipe_config
from src.coefficient_pattern_codec import assign_coefficient_patterns, selected_support_grams


class ChurchSupportPatternRQTransformer(SupportFirstLaserRQTransformer):
    """Keep cumulative support context without CUDA's nondeterministic cumsum."""
    def __init__(self, config, num_atoms, coefficient_pattern_vocab_size):
        config = config.copy()
        config.cumsum_depth_ctx = False  # Prefix accumulation is explicit below.
        super().__init__(config, num_atoms, coefficient_pattern_vocab_size)

    def embed_depth_with_model_aux(self, packed, model_aux):
        vectors = super().embed_depth_with_model_aux(packed, model_aux)
        rows = [vectors[..., 0, :]]
        for depth in range(1, vectors.shape[-2]):
            rows.append(rows[-1] + vectors[..., depth, :])
        return torch.stack(rows, -2)


def support_pattern_prior(vocabulary, dropout=.15):
    config = recipe_config('balanced')
    config.body.block.resid_pdrop = float(dropout)
    config.head.block.resid_pdrop = float(dropout)
    return ChurchSupportPatternRQTransformer(config, 16384, vocabulary)


@torch.no_grad()
def pattern_targets(aux, atoms, physical):
    if atoms.shape != physical.shape or atoms.shape[-1] != 4:
        raise ValueError('Expected four atoms and physical coefficients per site')
    with torch.autocast(atoms.device.type, enabled=False):
        grams = selected_support_grams(atoms.reshape(-1, 4), aux.dictionary.t().float())
        ids, _ = assign_coefficient_patterns(physical.reshape(-1, 4).float(),
            aux.coefficient_patterns.float(), grams=grams, chunk_size=4096)
    return ids.reshape(atoms.shape[:-1])


def pattern_objective(model, aux, atoms, physical):
    ids = pattern_targets(aux, atoms, physical)
    with torch.autocast(atoms.device.type, dtype=torch.bfloat16, enabled=atoms.is_cuda):
        output = model(model.pack(atoms, ids), model_aux=aux, amp=atoms.is_cuda)
    loss, values = support_first_objective(output['atom_logits'], output['pattern_logits'], atoms, ids,
                                           atom_weight=1.5, pattern_weight=4.)
    with torch.no_grad():
        prediction = output['pattern_logits'].argmax(-1)
        coefficients = aux.coefficient_patterns[prediction]
        metrics = {'loss':float(loss.detach()), 'atom_nll':float(values['atom_nll'].mean()),
            'pattern_nll':float(values['pattern_nll'].mean()), 'selection_score':float(loss.detach()),
            'pattern_accuracy':float((prediction == ids).float().mean()),
            'coefficient_argmax_mae':float((coefficients-physical).abs().mean()),
            'sign_accuracy':float(((coefficients >= 0) == (physical >= 0)).float().mean()),
            'coefficient_out_of_range':float((physical.abs() > aux.coeff_scales*3).float().mean())}
        for depth in range(4):
            metrics[f'atom_nll_d{depth}'] = float(values['atom_nll'][...,depth].mean())
    return loss, metrics
