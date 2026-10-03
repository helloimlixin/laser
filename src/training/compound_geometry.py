"""Physical expectation of an atom-conditional compound distribution."""

import torch


def joint_candidate_contribution(weights, atom_vectors, coefficient_means):
    """Evaluate sum_a p(a|h) D[a] E[c|h,a], with aligned candidate axes."""
    if weights.shape != coefficient_means.shape or atom_vectors.shape[:-1] != weights.shape:
        raise ValueError("Candidate weights, vectors and coefficient means must align")
    return (weights[..., None] * atom_vectors * coefficient_means[..., None]).sum(-2)


def conditional_geometry_prediction(
    model, aux, hidden, atom_logits, coeff_logits, target_atoms, top_k=4,
):
    """Top-k joint expectation, including the teacher atom exactly once.

    The teacher coefficient distribution is conditioned on the teacher atom;
    it cannot be reused for other dictionary vectors. Re-evaluate the same
    coefficient head for each candidate, retaining gradients through both
    conditionals. Reuse the teacher branch for matching candidates so its
    dropout realization agrees with the classification objective.
    """
    count = min(max(int(top_k), 1), atom_logits.shape[-1])
    candidate_logits, candidate_atoms = atom_logits.float().topk(count, -1)
    teacher_logits = atom_logits.float().gather(-1, target_atoms[..., None])
    teacher_logits = teacher_logits.masked_fill(
        (candidate_atoms == target_atoms[..., None]).any(-1, keepdim=True),
        -torch.inf,
    )
    weights = torch.cat((candidate_logits, teacher_logits), -1).softmax(-1)
    vectors = aux.dictionary.t()[candidate_atoms]
    bins = aux.coeff_bins.float()
    scales = aux.coeff_scales.float()
    means = []
    for depth in range(target_atoms.shape[-1]):
        context = hidden[..., depth, None, :].expand(
            *hidden.shape[:-2], count, hidden.shape[-1]
        )
        candidate_coefficients = model.coefficient_logits(
            context, vectors[..., depth, :, :], depth_index=depth,
        )
        candidate_coefficients = torch.where(
            (candidate_atoms[..., depth, :] == target_atoms[..., depth, None])[..., None],
            coeff_logits[..., depth, None, :], candidate_coefficients,
        )
        means.append(
            (candidate_coefficients.float().softmax(-1) * bins).sum(-1) * scales[depth]
        )
    means = torch.stack(means, -2)
    teacher_means = (coeff_logits.float().softmax(-1) * bins).sum(-1) * scales
    means = torch.cat((means, teacher_means[..., None]), -1)
    vectors = torch.cat((vectors, aux.dictionary.t()[target_atoms][..., None, :]), -2)
    return joint_candidate_contribution(weights, vectors.float(), means)
