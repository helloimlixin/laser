"""Controlled sparse-AR representation factors, with the same prior weights."""
from types import MethodType

import torch
from torch.nn import functional as F

from src.orthogonal_sparse_codec import dictionary_to_orthogonal_coefficients, ordered_support_basis
from src.training import laser_hard_ar


def coefficient_logits(logits, bins):
    if bins == logits.shape[-1]:
        return logits
    if logits.shape[-1] % bins:
        raise ValueError('Coefficient groups must partition the existing classifier')
    return logits.reshape(*logits.shape[:-1], bins, logits.shape[-1] // bins).logsumexp(-1)


def configure(profile, model, aux, *, coordinates='physical', bins=2048):
    """Change one coordinate/precision factor; keep all learned tensors intact.

    Coarse classes exactly marginalize groups of the existing classifier. All
    token IDs within a group have the same physical value, so grouped sampling
    and teacher forcing describe the same ordinary categorical AR problem.
    """
    if coordinates not in ('physical', 'orthogonal') or bins not in (128, 2048):
        raise ValueError('Unknown controlled factor')
    group = 2048 // bins
    if bins != 2048:
        values = torch.linspace(-3, 3, bins, device=aux.coeff_bins.device)
        aux.coeff_bins.copy_(values.repeat_interleave(group))
    else:
        values = aux.coeff_bins
    original_pairs = model._physical_pairs
    if coordinates == 'orthogonal':
        def physical_pairs(self, tokens, tokenizer):
            atoms, vectors, coefficients, _ = original_pairs(tokens, tokenizer)
            basis, _ = ordered_support_basis(vectors)
            return atoms, basis, coefficients, basis * coefficients[..., None]
        model._physical_pairs = MethodType(physical_pairs, model)

    @torch.no_grad()
    def targets(self, tokenizer, images):
        chunks = []
        with torch.autocast(device_type=images.device.type, enabled=False):
            if not hasattr(tokenizer, '_hard_ar_gram'):
                tokenizer._hard_ar_gram = tokenizer.dictionary.T @ tokenizer.dictionary
            for chunk in images.split(int(self.recipe.targets.encoder_batch_size)):
                signals = tokenizer.quant_conv(tokenizer.encoder(chunk.float())).permute(0, 2, 3, 1).float()
                atoms, physical = laser_hard_ar.sparse_encoding(signals, tokenizer.dictionary, 4,
                    gram=tokenizer._hard_ar_gram)
                if coordinates == 'orthogonal':
                    physical, _ = dictionary_to_orthogonal_coefficients(tokenizer.dictionary.T[atoms], physical)
                ids = laser_hard_ar.nearest_coefficient_ids(physical / tokenizer.coeff_scales, values)
                tokens = torch.empty(*atoms.shape[:-1], 8, dtype=torch.long, device=images.device)
                tokens[..., 0::2] = atoms
                tokens[..., 1::2] = tokenizer.num_atoms + ids * group + group // 2
                chunks.append(tokens)
        tokens = torch.cat(chunks)
        return tokens, tokens

    def loss(self, prior, output, tokens):
        atoms = output['atom_logits'].float()
        coefficients = coefficient_logits(output['coeff_logits'].float(), bins)
        a = F.cross_entropy(atoms.reshape(-1, atoms.shape[-1]), tokens[..., 0::2].reshape(-1))
        c = F.cross_entropy(coefficients.reshape(-1, bins),
            ((tokens[..., 1::2] - aux.num_atoms) // group).reshape(-1))
        return (a + c) / 2

    @torch.no_grad()
    def decode(self, tokenizer, tokens):
        atoms = tokens[..., 0::2]
        vectors = tokenizer.dictionary.T[atoms]
        if coordinates == 'orthogonal':
            vectors, _ = ordered_support_basis(vectors)
        coefficients = tokenizer.coeff_bins[tokens[..., 1::2] - tokenizer.num_atoms] * tokenizer.coeff_scales
        latent = (vectors * coefficients[..., None]).sum(-2)
        return tokenizer.decoder(tokenizer.post_quant_conv(latent.permute(0, 3, 1, 2).contiguous())).clamp(-1, 1)

    profile.targets = MethodType(targets, profile)
    profile.loss = MethodType(loss, profile)
    profile.decode = MethodType(decode, profile)
    if bins != 2048:
        import src.models.laser_rq_adapter as adapter
        from rqvae.utils.utils import sample_from_logits
        def grouped_sample(logits, temperature=1., top_k=None, top_p=None):
            if logits.shape[-1] == 2048:
                logits = coefficient_logits(logits.float(), bins)
                if top_k is not None:
                    top_k = min(top_k, bins)
                return sample_from_logits(logits, temperature, top_k, top_p) * group + group // 2
            return sample_from_logits(logits, temperature, top_k, top_p)
        adapter.sample_from_logits = grouped_sample
    return dict(coordinates=coordinates, coefficient_bins=bins,
        effective_coefficient_classes=bins, learned_parameter_shapes_unchanged=True,
        objective='ordinary_categorical_next_token_cross_entropy', auxiliary_loss_terms=0,
        sampled_trajectories=0)
