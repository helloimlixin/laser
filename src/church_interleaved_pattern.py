"""Interleaved atom/coefficient AR with an unchanged complete-site pattern codec."""
import math

import torch
from torch import nn

from src.training.rqtransformer import CompoundLaserRQTransformer
from src.church_ffhq_recipe import recipe_config
from src.church_pattern_order import ChurchPatternOrderRQTransformer, sample_field
from src.church_support_pattern_training import pattern_targets
from src.models.rqtransformer.transformers import RQTransformer


class CoefficientPrefixTree(nn.Module):
    """Allowed next scalar bins, determined solely by preceding coefficient bins.

    A leaf is exactly one existing joint pattern. This re-factorizes its
    probability into four coefficient decisions without changing the codebook.
    """
    def __init__(self, coefficient_ids, vocabulary=2048):
        super().__init__()
        rows = coefficient_ids.long().cpu()
        if rows.ndim != 2 or rows.shape[1] != 4 or len(rows) < 2:
            raise ValueError('Expected a table of four-coefficient patterns')
        if rows.min() < 0 or rows.max() >= vocabulary:
            raise ValueError('Coefficient bin outside vocabulary')
        tuples = [tuple(row) for row in rows.tolist()]
        if len(set(tuples)) != len(tuples):
            raise ValueError('Pattern rows must have distinct coefficient-bin tuples')
        self.vocabulary = vocabulary
        self.register_buffer('pattern_coefficient_ids', rows)
        prefixes = [{(): 0}]
        for depth in range(1, 5):
            unique = dict.fromkeys(row[:depth] for row in tuples)
            prefixes.append({key: index for index, key in enumerate(unique)})
        for depth in range(4):
            transition = torch.full((len(prefixes[depth]), vocabulary), -1, dtype=torch.int32)
            for prefix, child in prefixes[depth+1].items():
                transition[prefixes[depth][prefix[:-1]], prefix[-1]] = child
            self.register_buffer(f'transition_{depth}', transition)
        leaf = torch.empty(len(tuples), dtype=torch.long)
        for index, row in enumerate(tuples):
            leaf[prefixes[4][row]] = index
        self.register_buffer('leaf_pattern', leaf)

    def transitions(self, depth):
        return getattr(self, f'transition_{depth}')

    def nodes(self, coefficients):
        state = torch.zeros(coefficients.shape[:-1], device=coefficients.device, dtype=torch.long)
        states = []
        for depth in range(4):
            states.append(state)
            state = self.transitions(depth)[state, coefficients[..., depth]].long()
            if (state < 0).any():
                raise ValueError('Coefficient prefix cannot complete an existing pattern')
        return states, self.leaf_pattern[state]

    def mask(self, logits, state, depth):
        return logits.masked_fill(self.transitions(depth)[state] < 0, -float('inf'))


class InterleavedPatternRQTransformer(RQTransformer):
    """Eight causal events: a0,c0,a1,c1,a2,c2,a3,c3; one 67-bit stored code."""
    cached_hidden = ChurchPatternOrderRQTransformer.cached_hidden

    def __init__(self, config, num_atoms, coefficient_ids, coefficient_vocabulary=2048):
        config = config.copy()
        config.cumsum_depth_ctx = False
        super().__init__(config)
        if self.block_size[-1] != 8:
            raise ValueError('Expected eight interleaved fields')
        self.num_atoms = num_atoms
        self.coeff_vocab_size = coefficient_vocabulary
        self.tree = CoefficientPrefixTree(coefficient_ids, coefficient_vocabulary)
        self.coefficient_classifiers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(config.embed_dim), nn.Linear(config.embed_dim, coefficient_vocabulary))
            for _ in range(4)])
        self._teacher_atoms = self._teacher_coefficients = None

    def pack_fields(self, atoms, coefficients):
        if atoms.shape != coefficients.shape or atoms.shape[-1] != 4:
            raise ValueError('Expected four atom/coefficient pairs')
        return torch.stack((atoms, coefficients), -1).flatten(-2).long()

    def pack(self, atoms, pattern_ids):
        return self.pack_fields(atoms, self.tree.pattern_coefficient_ids[pattern_ids.long()])

    def unpack(self, packed):
        if packed.shape[-1] != 8:
            raise ValueError('Expected eight interleaved fields')
        return packed[..., ::2].long(), packed[..., 1::2].long()

    def embed_with_model_aux(self, packed, aux):
        atoms, coefficients = self.unpack(packed)
        contributions = aux.dictionary.t()[atoms] * (aux.coeff_bins[coefficients]*aux.coeff_scales)[..., None]
        z = contributions.sum(-2)
        zero = torch.zeros_like(z)
        return torch.stack((zero,)*7+(z,), -2)

    def embed_depth_with_model_aux(self, packed, aux):
        atoms, ids = self.unpack(packed)
        vectors = aux.dictionary.t()[atoms]
        coefficients = aux.coeff_bins[ids] * aux.coeff_scales
        prefix = torch.zeros_like(vectors[..., 0, :])
        events = []
        for depth in range(4):
            # Before c_d is known, only the chosen atom and completed pairs
            # enter context. After c_d, its signed physical contribution does.
            events.append(prefix + vectors[..., depth, :])
            prefix = prefix + vectors[..., depth, :] * coefficients[..., depth, None]
            events.append(prefix)
        return torch.stack(events, -2)

    def classify_head_outputs(self, hidden):
        if self._teacher_atoms is None:
            raise RuntimeError('Teacher pairs were not set')
        atom_logits = CompoundLaserRQTransformer.mask_seen_atoms(
            self.classifier(hidden[..., ::2, :]), self._teacher_atoms)
        nodes, _ = self.tree.nodes(self._teacher_coefficients)
        coefficients = []
        for depth, classifier in enumerate(self.coefficient_classifiers):
            logits = classifier(hidden[..., 2*depth+1, :])
            coefficients.append(self.tree.mask(logits, nodes[depth], depth))
        return {'atom_logits': atom_logits, 'coefficient_logits': torch.stack(coefficients, -2)}

    def forward(self, packed, model_aux=None, cond=None, amp=False):
        self._teacher_atoms, self._teacher_coefficients = self.unpack(packed)
        try:
            return super().forward(packed, model_aux=model_aux, cond=cond, amp=amp)
        finally:
            self._teacher_atoms = self._teacher_coefficients = None

    @torch.no_grad()
    def sample_compound(self, batch_size, model_aux, atom_top_k=2048, atom_top_p=None,
                        coeff_top_k=0, coeff_top_p=.5, atom_temperature=1., coeff_temperature=1., amp=True):
        height, width, _ = self.block_size
        device = next(self.parameters()).device
        atoms = torch.zeros(batch_size, height, width, 4, dtype=torch.long, device=device)
        coefficients = torch.zeros_like(atoms)
        patterns = torch.zeros_like(atoms[..., 0])
        packed = self.pack_fields(atoms, coefficients)
        self.init_cache()
        try:
            for h in range(height):
                for w in range(width):
                    node = torch.zeros(batch_size, dtype=torch.long, device=device)
                    for depth in range(4):
                        with torch.amp.autocast('cuda', enabled=amp):
                            hidden = self.cached_hidden(packed, model_aux, (h,w,2*depth), amp=amp)
                            logits = self.classifier(hidden)
                        if depth:
                            logits = logits.clone().scatter_(1, atoms[:,h,w,:depth], -float('inf'))
                        atom = sample_field(logits, atom_temperature, atom_top_k, atom_top_p)
                        atoms[:,h,w,depth] = atom
                        packed[:,h,w,2*depth] = atom
                        with torch.amp.autocast('cuda', enabled=amp):
                            hidden = self.cached_hidden(packed, model_aux, (h,w,2*depth+1), amp=amp)
                            logits = self.coefficient_classifiers[depth](hidden)
                        logits = self.tree.mask(logits, node, depth)
                        value = sample_field(logits, coeff_temperature, coeff_top_k, coeff_top_p)
                        coefficients[:,h,w,depth] = value
                        packed[:,h,w,2*depth+1] = value
                        node = self.tree.transitions(depth)[node, value].long()
                    patterns[:,h,w] = self.tree.leaf_pattern[node]
        finally:
            self.init_cache()
        assert torch.equal(coefficients, self.tree.pattern_coefficient_ids[patterns])
        return atoms, patterns


def interleaved_pattern_prior(book, dropout=.15):
    config = recipe_config('balanced')
    config.block_size = [8,8,8]
    config.body.block.resid_pdrop = float(dropout)
    config.head.block.resid_pdrop = float(dropout)
    return InterleavedPatternRQTransformer(config, 16384, book['pattern_coefficient_ids'])


def interleaved_objective(model, aux, atoms, physical):
    patterns = pattern_targets(aux, atoms, physical)
    ids = model.tree.pattern_coefficient_ids[patterns]
    with torch.autocast(atoms.device.type, dtype=torch.bfloat16, enabled=atoms.is_cuda):
        output = model(model.pack(atoms,patterns), model_aux=aux, amp=atoms.is_cuda)
    atom_nll = -output['atom_logits'].float().log_softmax(-1).gather(-1, atoms[...,None]).squeeze(-1)
    coefficient_nll = -output['coefficient_logits'].float().log_softmax(-1).gather(-1, ids[...,None]).squeeze(-1)
    joint = atom_nll.sum(-1)+coefficient_nll.sum(-1)
    loss = joint.mean()/5  # Same complete-code likelihood scale as ordering pilots.
    with torch.no_grad():
        prediction = output['coefficient_logits'].argmax(-1)
        physical_prediction = aux.coeff_bins[prediction]*aux.coeff_scales
        metrics = {'loss':float(loss.detach()),'atom_nll':float(atom_nll.mean()),
            'pattern_nll':float(coefficient_nll.sum(-1).mean()),'joint_nll':float(joint.mean()),
            'joint_bits_per_site':float(joint.mean()/math.log(2)),'selection_score':float(joint.mean()),
            'coefficient_accuracy':float((prediction==ids).float().mean()),
            'coefficient_argmax_mae':float((physical_prediction-physical).abs().mean()),
            'sign_accuracy':float(((physical_prediction>=0)==(physical>=0)).float().mean()),
            'coefficient_out_of_range':float((physical.abs()>aux.coeff_scales*3).float().mean())}
        for depth in range(4):
            metrics[f'atom_nll_d{depth}'] = float(atom_nll[...,depth].mean())
            metrics[f'coefficient_nll_d{depth}'] = float(coefficient_nll[...,depth].mean())
    return loss, metrics
