"""One arbitrary-width integer for exact support plus a joint coefficient ID.

The integer contains every component needed to recover a complete sparse code.
At four 16k-way atoms, patterns larger than 128 require more than 63 bits;
Python integers retain those bits exactly. Predictors operate on internal fields.
"""
from __future__ import annotations

import operator

import torch


def _settings(num_atoms, num_patterns, depth):
    values = tuple(operator.index(x) for x in (num_atoms, num_patterns, depth))
    if min(values) < 1:
        raise ValueError('Vocabulary sizes and depth must be positive')
    return values


def pack_support_pattern(atoms, pattern_ids, *, num_patterns, num_atoms=16384):
    if atoms.ndim < 1 or atoms.shape[:-1] != pattern_ids.shape:
        raise ValueError('Support and pattern IDs must have matching site axes')
    num_atoms, num_patterns, depth = _settings(num_atoms, num_patterns, atoms.shape[-1])
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if atoms.dtype not in integer_dtypes or pattern_ids.dtype not in integer_dtypes:
        raise ValueError('Expected integer component tensors')
    if atoms.numel() and (int(atoms.min()) < 0 or int(atoms.max()) >= num_atoms):
        raise ValueError('Atom outside vocabulary')
    if pattern_ids.numel() and (int(pattern_ids.min()) < 0 or int(pattern_ids.max()) >= num_patterns):
        raise ValueError('Pattern outside vocabulary')
    result = []
    for support, pattern in zip(atoms.reshape(-1, depth).tolist(), pattern_ids.flatten().tolist()):
        value = pattern
        for atom in reversed(support):
            value = value * num_atoms + atom
        result.append(value)
    return result


def unpack_support_pattern(values, *, num_patterns, num_atoms=16384, depth=4):
    num_atoms, num_patterns, depth = _settings(num_atoms, num_patterns, depth)
    maximum = num_atoms ** depth * num_patterns
    supports, patterns = [], []
    for value in values:
        value = operator.index(value)
        if not 0 <= value < maximum:
            raise ValueError('Complete-site integer outside vocabulary')
        support = []
        for _ in range(depth):
            value, atom = divmod(value, num_atoms)
            support.append(atom)
        supports.append(support)
        patterns.append(value)
    return torch.tensor(supports, dtype=torch.long).reshape(-1, depth), torch.tensor(patterns, dtype=torch.long)


def decode_support_pattern_integers(values, pattern_coefficient_ids, *, num_atoms=16384):
    """Recover all atoms and signed-bin IDs without any source-support input."""
    if pattern_coefficient_ids.ndim != 2:
        raise ValueError('Expected coefficient pattern table [vocabulary, depth]')
    atoms, patterns = unpack_support_pattern(values, num_atoms=num_atoms,
        num_patterns=len(pattern_coefficient_ids), depth=pattern_coefficient_ids.shape[1])
    return atoms, pattern_coefficient_ids[patterns.to(pattern_coefficient_ids.device)]
