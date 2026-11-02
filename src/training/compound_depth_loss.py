"""Depth-specific CE emphasis with fixed total weight for each pair field."""
import math

from src.training.depth_weighted_ce import normalize_depth_weights


def compound_depth_classification(atom_ce, coefficient_ce, *, atom_weight=1.,
                                   atom_depth_weights=None, coefficient_depth_weights=None):
    """Return classification and depth-weighted CE arrays; retain raw CE upstream.

    Each supplied vector is normalized to mean one. Multiplying a vector by a
    constant therefore does not change the objective or the atom/coeff budget.
    Omitted vectors preserve the original arithmetic and gradients exactly.
    """
    if atom_ce.shape != coefficient_ce.shape or atom_ce.ndim < 1 or not atom_ce.numel():
        raise ValueError('Expected matching, nonempty per-token CE arrays with a depth axis')
    if not math.isfinite(atom_weight) or atom_weight <= 0:
        raise ValueError('atom weight must be positive and finite')
    depth = atom_ce.shape[-1]
    weighted_atom, weighted_coefficient = atom_ce, coefficient_ce
    if atom_depth_weights is not None:
        weighted_atom = atom_ce * atom_ce.new_tensor(normalize_depth_weights(atom_depth_weights, depth))
    if coefficient_depth_weights is not None:
        weighted_coefficient = coefficient_ce * coefficient_ce.new_tensor(
            normalize_depth_weights(coefficient_depth_weights, depth))
    classification = (atom_weight * weighted_atom.sum(-1) + weighted_coefficient.sum(-1)).mean()
    classification = classification / ((atom_weight + 1.) * depth)
    return classification, weighted_atom, weighted_coefficient
