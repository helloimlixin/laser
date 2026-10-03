"""Fresh full-vocabulary paired targets around a frozen complete sparse vector.

Each coordinate uses the residual after subtracting all *other* current pairs.
The atom draw marginalizes the two signs of that coordinate's current magnitude;
the paired coefficient is then drawn over its entire fixed bin grid. These are
overlapping Gibbs blocks on a symmetric grid, without any continuous solve.

Only the final ascending sweep emits training labels. Its emitted prefix never
changes again, so the saved soft labels describe the same finite-step procedural
teacher as the returned tokens. Finite sweeps are not claimed to reach Gibbs
equilibrium. Production uses FP32; FP64 inputs support independent CPU proofs.
"""
from contextlib import contextmanager
import math

import torch


POLICY_VERSION = 'fresh_signed_paired_gibbs_complete_vector_v1'


@contextmanager
def teacher_precision(device_type):
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _positive_temperature(temperature):
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('temperature must be finite and positive')


def _entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(
        torch.finfo(probabilities.dtype).tiny).log()).sum(-1)


@torch.no_grad()
def fresh_paired_targets(anchor_atoms, physical_coefficients, dictionary,
                         coefficient_values, *, temperature,
                         warmup_sweeps=1, site_chunk_size=256, generator=None,
                         reference_bank_atoms=None, return_trace=False):
    """Sample complete paired sequences; no bank restriction on positive labels.

    Inputs: anchor atoms/physical coefficients [..., D], dictionary [C, A],
    fixed physical coefficient grids [D, B]. Optional reference bank [..., V, D]
    is used only for coverage diagnostics. Warmup sweeps emit no labels. The
    final sweep visits depths 0..D-1, recording each distribution before its draw.

    Site chunking bounds scratch memory, not vocabulary support. All dense soft
    labels remain allocated. Chunk size changes RNG draw order and is therefore
    part of the reproducibility contract. No input tensor is mutated.
    """
    _positive_temperature(temperature)
    if (isinstance(warmup_sweeps, bool) or not isinstance(warmup_sweeps, int)
            or warmup_sweeps < 0):
        raise ValueError('warmup_sweeps must be a nonnegative integer')
    if (isinstance(site_chunk_size, bool) or not isinstance(site_chunk_size, int)
            or site_chunk_size < 1):
        raise ValueError('site_chunk_size must be a positive integer')
    if (anchor_atoms.ndim < 2 or anchor_atoms.shape != physical_coefficients.shape
            or dictionary.ndim != 2 or coefficient_values.ndim != 2
            or coefficient_values.shape[0] != anchor_atoms.shape[-1]
            or coefficient_values.shape[1] < 2 or dictionary.shape[0] < 1
            or dictionary.shape[1] < anchor_atoms.shape[-1]
            or anchor_atoms.numel() == 0):
        raise ValueError('invalid anchor, dictionary, or coefficient-grid shapes')
    if (anchor_atoms.dtype not in (torch.int16, torch.int32, torch.int64)
            or dictionary.dtype not in (torch.float32, torch.float64)
            or physical_coefficients.dtype != dictionary.dtype
            or coefficient_values.dtype != dictionary.dtype
            or any(x.device != dictionary.device for x in
                   (anchor_atoms, physical_coefficients, coefficient_values))):
        raise ValueError('inputs must share a device and FP32/FP64 physical dtype')
    if (any(not bool(torch.isfinite(x).all()) for x in
            (physical_coefficients, dictionary, coefficient_values))
            or bool((anchor_atoms < 0).any())
            or bool((anchor_atoms >= dictionary.shape[1]).any())):
        raise ValueError('inputs must be finite with valid atom IDs')
    if (not torch.equal(coefficient_values, -coefficient_values.flip(-1))
            or not bool((coefficient_values[:, 1:] > coefficient_values[:, :-1]).all())):
        raise ValueError('coefficient grids must be increasing and exactly sign-symmetric')
    leading, depth = anchor_atoms.shape[:-1], anchor_atoms.shape[-1]
    vocab, bins = dictionary.shape[1], coefficient_values.shape[1]
    flat_anchors = anchor_atoms.reshape(-1, depth).long()
    flat_physical = physical_coefficients.reshape_as(flat_anchors)
    prior = torch.tril(torch.ones(depth, depth, dtype=torch.bool,
                                 device=dictionary.device), diagonal=-1)
    if bool(((flat_anchors[:, :, None] == flat_anchors[:, None, :]) & prior).any()):
        raise ValueError('anchor support must contain distinct atoms')
    if reference_bank_atoms is not None:
        if (reference_bank_atoms.shape[:-2] != leading
                or reference_bank_atoms.shape[-1] != depth
                or reference_bank_atoms.shape[-2] < 1
                or reference_bank_atoms.device != dictionary.device
                or reference_bank_atoms.dtype not in (torch.int16, torch.int32, torch.int64)):
            raise ValueError('reference bank must align with anchor sites and depths')
        reference_bank_atoms = reference_bank_atoms.reshape(-1, reference_bank_atoms.shape[-2], depth)

    with teacher_precision(dictionary.device.type):
        count = len(flat_anchors)
        atoms_out = torch.empty_like(flat_anchors)
        ids_out = torch.empty_like(flat_anchors)
        initial_ids = torch.empty_like(flat_anchors)
        atom_probs = dictionary.new_empty(count, depth, vocab)
        coefficient_probs = dictionary.new_empty(count, depth, bins)
        targets = dictionary.new_empty(count, dictionary.shape[0])
        reconstructions = torch.empty_like(targets)
        anchor_distortion = dictionary.new_empty(count)
        traces = None
        if return_trace:
            traces = dict(residuals=dictionary.new_empty(count, depth, dictionary.shape[0]),
                old_magnitudes=dictionary.new_empty(count, depth),
                atoms_before=torch.empty(count, depth, depth, dtype=torch.long, device=dictionary.device),
                coefficient_ids_before=torch.empty(count, depth, depth, dtype=torch.long, device=dictionary.device))
        norms = dictionary.square().sum(0)
        dictionary_rows = dictionary.T
        depths = torch.arange(depth, device=dictionary.device)
        other_depths = [depths[depths != d] for d in range(depth)]
        for start in range(0, count, site_chunk_size):
            stop = min(start + site_chunk_size, count)
            anchors = flat_anchors[start:stop]
            coefficients = flat_physical[start:stop]
            z0 = (dictionary_rows[anchors] * coefficients[..., None]).sum(-2)
            current_atoms = anchors.clone()
            current_ids = (coefficients[..., None] - coefficient_values).abs().argmin(-1)
            initial_ids[start:stop] = current_ids
            current_values = coefficient_values[depths, current_ids]
            current_vectors = dictionary_rows[current_atoms] * current_values[..., None]
            anchor_distortion[start:stop] = (z0 - current_vectors.sum(-2)).square().sum(-1)
            for sweep in range(warmup_sweeps + 1):
                final = sweep == warmup_sweeps
                for d in range(depth):
                    # Recompute from the complete current tuple, avoiding a
                    # stale residual or accumulated incremental-update drift.
                    other = other_depths[d]
                    residual = z0 - current_vectors[:, other].sum(-2)
                    magnitude = current_values[:, d].abs()
                    correlations = residual @ dictionary
                    signed_score = 2 * magnitude[:, None] * correlations / temperature
                    logits = (torch.logaddexp(signed_score, -signed_score)
                              - magnitude[:, None].square() * norms / temperature)
                    logits.scatter_(1, current_atoms[:, other], -torch.inf)
                    qa = logits.softmax(-1)
                    if final and traces is not None:
                        traces['residuals'][start:stop, d] = residual
                        traces['old_magnitudes'][start:stop, d] = magnitude
                        traces['atoms_before'][start:stop, d] = current_atoms
                        traces['coefficient_ids_before'][start:stop, d] = current_ids
                    new_atom = torch.multinomial(qa, 1, generator=generator).squeeze(-1)
                    selected = correlations.gather(1, new_atom[:, None])
                    values = coefficient_values[d]
                    qc = ((2 * selected * values
                           - norms[new_atom, None] * values.square()) / temperature).softmax(-1)
                    new_id = torch.multinomial(qc, 1, generator=generator).squeeze(-1)
                    if final:
                        atom_probs[start:stop, d] = qa
                        coefficient_probs[start:stop, d] = qc
                    # Commit the complete sampled pair. Final-pass prefixes
                    # are never touched by any later coordinate update.
                    new_value = values[new_id]
                    current_atoms[:, d] = new_atom
                    current_ids[:, d] = new_id
                    current_values[:, d] = new_value
                    current_vectors[:, d] = dictionary_rows[new_atom] * new_value[:, None]
            atoms_out[start:stop] = current_atoms
            ids_out[start:stop] = current_ids
            targets[start:stop] = z0
            reconstructions[start:stop] = current_vectors.sum(-2)

        atom_entropy = _entropy(atom_probs)
        coefficient_entropy = _entropy(coefficient_probs)
        sampled_distortion = (targets - reconstructions).square().sum(-1)
        target_energy = targets.square().sum(-1)
        atom_changed = atoms_out != flat_anchors
        coefficient_changed = ids_out != initial_ids
        pair_changed = atom_changed | coefficient_changed
        diagnostics = dict(sampled_distortion=sampled_distortion.mean(),
            anchor_distortion=anchor_distortion.mean(), signal_energy=target_energy.mean(),
            relative_distortion=sampled_distortion.mean()/target_energy.mean().clamp_min(1e-30),
            atom_change_fraction=atom_changed.float().mean(),
            coefficient_change_fraction=coefficient_changed.float().mean(),
            pair_change_fraction=pair_changed.float().mean(),
            combination_change_fraction=pair_changed.any(-1).float().mean(),
            atom_target_entropy=atom_entropy.mean(), coefficient_target_entropy=coefficient_entropy.mean(),
            coefficient_edge_fraction=((ids_out == 0) | (ids_out == bins-1)).float().mean(),
            negative_coefficient_fraction=(coefficient_values[depths, ids_out] < 0).float().mean())
        outside = None
        if reference_bank_atoms is not None:
            outside = ~(atoms_out[:, None, :] == reference_bank_atoms).any(-2)
            support_outside = ~(atoms_out[:, None, :] == reference_bank_atoms).all(-1).any(-1)
            diagnostics.update(atom_outside_bank_fraction=outside.float().mean(),
                               support_outside_bank_fraction=support_outside.float().mean())
        result = dict(packed=(atoms_out*bins + ids_out).reshape(*leading, depth),
            atoms=atoms_out.reshape(*leading, depth), coefficient_ids=ids_out.reshape(*leading, depth),
            atom_probs=atom_probs.reshape(*leading, depth, vocab),
            coefficient_probs=coefficient_probs.reshape(*leading, depth, bins),
            target_vectors=targets.reshape(*leading, dictionary.shape[0]),
            reconstruction=reconstructions.reshape(*leading, dictionary.shape[0]),
            anchor_atoms=anchor_atoms, anchor_coefficient_ids=initial_ids.reshape(*leading, depth),
            sampled_distortion=sampled_distortion.reshape(leading),
            anchor_distortion=anchor_distortion.reshape(leading), target_energy=target_energy.reshape(leading),
            atom_changed=atom_changed.reshape(*leading, depth),
            coefficient_changed=coefficient_changed.reshape(*leading, depth),
            atom_entropy=atom_entropy.reshape(*leading, depth),
            coefficient_entropy=coefficient_entropy.reshape(*leading, depth),
            atom_outside_bank=None if outside is None else outside.reshape(*leading, depth),
            diagnostics=diagnostics)
        result['atom_probabilities'] = result['atom_probs']
        result['coefficient_probabilities'] = result['coefficient_probs']
        if traces is not None:
            result['trace'] = {key:value.reshape(*leading, *value.shape[1:])
                               for key,value in traces.items()}
        return result


class FreshPairedTeacher:
    """FP32 auxiliary-model bridge for banks or already gathered anchor tuples."""
    def __init__(self, aux, temperature, warmup_sweeps=1, site_chunk_size=256):
        _positive_temperature(temperature)
        self.dictionary = aux.dictionary.detach().float()
        self.scales = aux.coeff_scales.detach().float()
        self.coefficient_values = aux.coeff_bins.detach().float()[None, :] * self.scales[:, None]
        self.temperature = float(temperature)
        self.warmup_sweeps = warmup_sweeps
        self.site_chunk_size = site_chunk_size

    @torch.no_grad()
    def from_anchors(self, anchor_atoms, normalized_coefficients, *, generator=None,
                     reference_bank_atoms=None, return_trace=False):
        return fresh_paired_targets(anchor_atoms, normalized_coefficients.float()*self.scales,
            self.dictionary, self.coefficient_values, temperature=self.temperature,
            warmup_sweeps=self.warmup_sweeps, site_chunk_size=self.site_chunk_size,
            generator=generator, reference_bank_atoms=reference_bank_atoms, return_trace=return_trace)

    @torch.no_grad()
    def __call__(self, bank_atoms, bank_coefficients, *, generator=None,
                 bank_choices=None, return_trace=False):
        if (bank_atoms.ndim < 3 or bank_atoms.shape != bank_coefficients.shape
                or bank_atoms.device != bank_coefficients.device or bank_atoms.shape[-2] < 1):
            raise ValueError('paired bank must have shape [..., variants, depth]')
        if bank_choices is None:
            bank_choices = torch.randint(bank_atoms.shape[-2], bank_atoms.shape[:-2],
                                         device=bank_atoms.device, generator=generator)
        if (bank_choices.shape != bank_atoms.shape[:-2] or bank_choices.device != bank_atoms.device
                or bank_choices.dtype not in (torch.int32, torch.int64)
                or bool((bank_choices < 0).any()) or bool((bank_choices >= bank_atoms.shape[-2]).any())):
            raise ValueError('invalid per-site bank choices')
        gather = bank_choices.long()[..., None, None].expand(*bank_choices.shape, 1, bank_atoms.shape[-1])
        anchors = bank_atoms.gather(-2, gather).squeeze(-2)
        coefficients = bank_coefficients.gather(-2, gather).squeeze(-2)
        result = self.from_anchors(anchors, coefficients, generator=generator,
                                  reference_bank_atoms=bank_atoms, return_trace=return_trace)
        result['bank_choices'] = bank_choices
        return result

    def metadata(self):
        return dict(policy_version=POLICY_VERSION, temperature=self.temperature,
            warmup_sweeps=self.warmup_sweeps, final_sweeps=1, final_order='ascending',
            site_chunk_size=self.site_chunk_size, atom_vocabulary=self.dictionary.shape[1],
            coefficient_vocabulary=self.coefficient_values.shape[1], refitting=False,
            finite_candidate_pool=False, stationary_sampling_claimed=False,
            production_dtype='float32', teacher_TF32=False,
            target='Complete continuous frozen anchor vector, with corresponding support/coefficient pairs',
            labels='Full-vocabulary draw-time conditionals of the final ascending procedural sweep')


def loss_parts(outputs, targets, atom_weight=1.5):
    """Historical normalized CE, with only the model's emitted-prefix mask.

    Teacher exclusion of other slots is encoded in its targets, not imposed on
    model logits. This preserves the original native autoregressive vocabulary.
    The caller applies accumulation/global-sample weighting separately.
    """
    if not math.isfinite(atom_weight) or atom_weight <= 0:
        raise ValueError('atom_weight must be finite and positive')
    atoms = targets['atoms'].long()
    atom_logits, coefficient_logits = outputs['atom_logits'], outputs['coeff_logits']
    if (atom_logits.shape != targets['atom_probs'].shape
            or coefficient_logits.shape != targets['coefficient_probs'].shape
            or atom_logits.shape[:-1] != atoms.shape):
        raise ValueError('model logits and teacher targets must align')
    dtype = torch.float64 if atom_logits.dtype == torch.float64 else torch.float32
    masked = atom_logits.to(dtype).clone()
    for d in range(1, atoms.shape[-1]):
        masked[..., d, :].scatter_(-1, atoms[..., :d], -torch.inf)
    log_atoms = masked.log_softmax(-1)
    qa = targets['atom_probs'].to(dtype)
    safe_atoms = torch.where(qa > 0, log_atoms, torch.zeros((), dtype=dtype, device=log_atoms.device))
    atom_ce = -(qa * safe_atoms).sum(-1).mean()
    coefficient_ce = -(targets['coefficient_probs'].to(dtype)
                       * coefficient_logits.to(dtype).log_softmax(-1)).sum(-1).mean()
    total = (float(atom_weight)*atom_ce + coefficient_ce)/(float(atom_weight)+1)
    return total,dict(atom_cross_entropy=atom_ce, coefficient_cross_entropy=coefficient_ce,
        atom_nll=-log_atoms.gather(-1, atoms[..., None]).squeeze(-1).mean(), classification=total,
        atom_entropy=targets['atom_entropy'].mean(), coefficient_entropy=targets['coefficient_entropy'].mean())
