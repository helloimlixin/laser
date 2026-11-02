"""Explicit parameter/Adam transfer and fixed-support coefficient policies."""
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


def parameter_source(name, atom_count, head_layers):
    """Return the existing tensor and slice used to initialize a pair parameter."""
    identity = lambda tensor: tensor
    if name == 'pos_emb_d':
        return name, lambda tensor: tensor[:, ::2].clone()
    if name.startswith('classifier.linear.'):
        return name, lambda tensor: tensor[:atom_count].clone()
    if name.startswith('coeff_classifier.'):
        _, _, layer, field = name.split('.')
        if layer == '0':
            return 'classifier.layer_norm.' + field, identity
        return 'classifier.linear.' + field, lambda tensor: tensor[atom_count:].clone()
    if name == 'coeff_atom_proj.weight':
        return 'selected_atom_projection.weight', identity
    if name.startswith('coeff_micro_transformer.blocks.'):
        suffix = name.removeprefix('coeff_micro_transformer.blocks.')
        index, rest = suffix.split('.', 1)
        return f'head_transformer.blocks.{head_layers - 2 + int(index)}.{rest}', identity
    if name == 'coeff_micro_pos':
        return 'pos_emb_d', lambda tensor: tensor[:, :2].clone()
    if name.startswith(('coeff_token_embedding.', 'pair_embedding_adapter.')):
        return None, None
    return name, identity


def transplant_state(model, source_state, source_parameter_names, source_optimizer=None):
    """Keep backbone weights/moments; name every copied or new pair parameter.

    New pair embedding parameters start with empty Adam states. Existing
    classifier slices and copied micro-decoder blocks retain their moments;
    this is an explicit architecture continuation, not a fresh model claim.
    """
    atoms = model.num_atoms
    layers = len(model.head_transformer.blocks)
    source_states = {}
    if source_optimizer is not None:
        indices = [i for group in source_optimizer['param_groups'] for i in group['params']]
        assert len(indices) == len(source_parameter_names)
        source_states = {name: source_optimizer['state'][index]
                         for name, index in zip(source_parameter_names, indices)}
    state, mapping, adam = {}, {}, {}
    for name, parameter in model.named_parameters():
        source, transform = parameter_source(name, atoms, layers)
        if source is not None:
            value = transform(source_state[source])
            if name != source:
                value = value.clone()
            assert value.shape == parameter.shape, (name, value.shape, parameter.shape)
            state[name] = value
            if source in source_states:
                adam[name] = {k: transform(v) if torch.is_tensor(v) and v.ndim else v
                              for k, v in source_states[source].items()}
        else:
            module_name, field = name.rsplit('.', 1)
            module = model.get_submodule(module_name)
            value = torch.empty(parameter.shape, dtype=parameter.dtype, device='cpu')
            if field == 'bias' or name.startswith('pair_embedding_adapter.3.'):
                value.zero_()
            elif isinstance(module, nn.LayerNorm):
                value.fill_(1.)
            else:
                value.normal_(std=.02)
            state[name] = value
        mapping[name] = source
    model.load_state_dict(state, strict=True, assign=True)
    return adam, mapping


def restore_named_adam(optimizer, model, states):
    for name, parameter in model.named_parameters():
        if name in states:
            optimizer.state[parameter] = {k: value.to(parameter.device).clone() if torch.is_tensor(value) and value.ndim
                                          else value.clone() if torch.is_tensor(value) else value
                                          for k, value in states[name].items()}


def configure_coefficient_policy(profile, model, aux, *, policy='hard', seed=261015):
    """Gaussian coefficient labels/contexts, with one deterministic support.

    A private generator leaves model-dropout RNG streams unchanged. Validation
    always evaluates hard clean-token likelihood on identical real encodings.
    Soft targets use physical SD 0.125, matching the September pair baseline.
    """
    if policy not in ('hard', 'soft'):
        raise ValueError('Unknown coefficient policy')
    clean_targets = profile.targets
    generator = torch.Generator(device=aux.dictionary.device).manual_seed(seed)

    @torch.no_grad()
    def targets(self, tokenizer, images):
        clean, codes = clean_targets(tokenizer, images)
        if policy == 'hard' or not model.training:
            return clean, codes
        ids = codes[..., 1::2] - tokenizer.num_atoms
        centers = tokenizer.coeff_bins[ids] * tokenizer.coeff_scales
        values = tokenizer.coeff_bins[None, :] * tokenizer.coeff_scales[:, None]
        probabilities = (-(centers[..., None] - values).square() / (2 * .125**2)).softmax(-1)
        sampled = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1,
                                    generator=generator).reshape_as(ids)
        contexts = codes.clone()
        contexts[..., 1::2] = sampled + tokenizer.num_atoms
        return dict(tokens=codes, coefficient_probabilities=probabilities), contexts

    def loss(self, prior, output, target):
        tokens = target['tokens'] if isinstance(target, dict) else target
        atoms = output['atom_logits'].float()
        coefficients = output['coeff_logits'].float()
        a = F.cross_entropy(atoms.reshape(-1, atoms.shape[-1]), tokens[..., 0::2].reshape(-1))
        if isinstance(target, dict):
            c = -(target['coefficient_probabilities'] * coefficients.log_softmax(-1)).sum(-1).mean()
        else:
            c = F.cross_entropy(coefficients.reshape(-1, coefficients.shape[-1]),
                                (tokens[..., 1::2] - tokenizer_atoms).reshape(-1))
        return (a + c) / 2

    tokenizer_atoms = aux.num_atoms
    profile.targets = MethodType(targets, profile)
    profile.loss = MethodType(loss, profile)
    return generator
