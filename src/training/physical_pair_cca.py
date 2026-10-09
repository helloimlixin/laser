"""Condition contrastive alignment for the existing scalar-pair prior.

CCA uses the probability of the actual sampled sequence. Soft-target cross
entropy is retained by the caller as a separate training anchor; it must not
be substituted for a sequence log likelihood. No unconditional class or new
model parameters are needed. Reference likelihoods are always detached.
"""
import math

import torch
import torch.nn.functional as F


def physical_pair_sequence_log_probability(atom_logits, coefficient_logits,
                                           atoms, coefficient_ids):
    """Sum atom and sampled coefficient log probabilities for each image."""
    if atom_logits.shape[:-1] != coefficient_logits.shape[:-1]:
        raise ValueError('Atom and coefficient predictions must cover the same pairs')
    if atoms.shape != atom_logits.shape[:-1] or coefficient_ids.shape != atoms.shape:
        raise ValueError('Targets must identify each predicted atom/coefficient pair')
    if atoms.ndim < 2:
        raise ValueError('Targets need an image dimension and a pair dimension')
    atom = F.log_softmax(atom_logits.float(), dim=-1).gather(
        -1, atoms.long().unsqueeze(-1)).squeeze(-1)
    coefficient = F.log_softmax(coefficient_logits.float(), dim=-1).gather(
        -1, coefficient_ids.long().unsqueeze(-1)).squeeze(-1)
    return (atom + coefficient).flatten(start_dim=1).sum(dim=1)


def condition_contrastive_alignment(positive_logp, negative_logp,
                                    reference_positive_logp, reference_negative_logp,
                                    positive_labels, negative_labels, *, beta=0.02,
                                    negative_weight=1.0):
    """Paper Eq. 12 with collision masking and explicit gradient normalization.

    A shuffled label equal to the true class is not penalized as a mismatch.
    The negative expectation keeps its original batch denominator, avoiding
    gradient amplification when many shuffled labels collide. Division by
    max(negative_weight, 1) follows the authors' stabilized implementation.
    This collision policy is an explicit adaptation of the paper objective.
    """
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('CCA beta must be finite and positive')
    if not math.isfinite(negative_weight) or negative_weight < 0:
        raise ValueError('CCA negative weight must be finite and nonnegative')
    scores = (positive_logp, negative_logp, reference_positive_logp,
              reference_negative_logp)
    if positive_logp.ndim != 1 or positive_logp.numel() == 0:
        raise ValueError('CCA requires a nonempty image batch of sequence scores')
    if any(score.shape != positive_logp.shape for score in scores):
        raise ValueError('All trainable and reference scores must match the image batch')
    if positive_labels.shape != positive_logp.shape or negative_labels.shape != positive_logp.shape:
        raise ValueError('Class labels must match the image batch')
    positive_gap = positive_logp.float() - reference_positive_logp.detach().float()
    negative_gap = negative_logp.float() - reference_negative_logp.detach().float()
    positive = F.softplus(-beta * positive_gap)
    mismatch = negative_labels.ne(positive_labels)
    negative = torch.where(mismatch, F.softplus(beta * negative_gap), 0.0)
    loss = (positive + negative_weight * negative).mean() / max(negative_weight, 1.0)
    return loss, {
        'positive_relative_log_probability': positive_gap.detach().mean(),
        'negative_relative_log_probability': negative_gap.detach().mean(),
        'negative_pair_fraction': mismatch.float().mean(),
        'positive_term': positive.detach().mean(),
        'negative_term': negative.detach().mean(),
    }
