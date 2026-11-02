"""Read-only decomposition of the single joint categorical training objective."""
import torch


@torch.no_grad()
def joint_integer_balance(logits, atoms, targets, *, classifier_weight=None, row_chunk=32):
    """Measure unit-weight atom/conditional-coefficient CE and gradient sizes.

    This computes diagnostics only; it never supplies another training loss.
    Gradient norms refer to unscaled per-token logits and the input to the
    shared classifier, before the common batch/depth mean and gradient clip.
    """
    if (logits.shape[:-1] != atoms.shape or targets.shape[:-1] != atoms.shape
            or atoms.ndim < 1 or logits.shape[-1] % targets.shape[-1]
            or atoms.dtype != torch.long or row_chunk < 1):
        raise ValueError('Invalid integer logits or sparse joint targets')
    x, a, p = logits.flatten(0, -2), atoms.reshape(-1), targets.reshape(-1, targets.shape[-1])
    depth, levels = atoms.shape[-1], targets.shape[-1]
    metrics = ['joint_ce', 'atom_marginal_ce', 'coefficient_conditional_ce',
        'coefficient_target_entropy', 'atom_logit_gradient_sq', 'coefficient_logit_gradient_sq',
        'joint_logit_gradient_sq', 'atom_context_gradient_sq', 'coefficient_context_gradient_sq',
        'joint_context_gradient_sq']
    sums = torch.zeros(depth, len(metrics), device=logits.device, dtype=torch.float64)
    counts = torch.zeros(depth, device=logits.device, dtype=torch.long)
    ce_error, norm_error, normalization_error = 0., 0., 0.
    for first in range(0, len(x), row_chunk):
        rows, target = x[first:first+row_chunk].float(), p[first:first+row_chunk].float()
        ids = a[first:first+len(rows), None]*levels+torch.arange(levels, device=x.device)
        chosen = rows.gather(-1, ids)
        mass = target.sum(-1)
        normalization_error = max(normalization_error, float((mass-1.).abs().max()))
        logz = rows.logsumexp(-1)
        logza = chosen.logsumexp(-1)
        atom_ce = (logz-logza)*mass
        coefficient_ce = logza*mass-(target*chosen).sum(-1)
        joint_ce = logz*mass-(target*chosen).sum(-1)
        entropy = -(target*target.clamp_min(1.e-30).log()).sum(-1)
        ce_error = max(ce_error, float((joint_ce-atom_ce-coefficient_ce).abs().max()))
        probability = (rows-logz[:, None]).exp()*mass[:, None]
        conditional = (chosen-logza[:, None]).exp()*mass[:, None]
        selected_probability = probability.gather(-1, ids)
        coefficient_gradient = conditional-target
        base_sq = probability.square().sum(-1)
        atom_sq = (base_sq+conditional.square().sum(-1)-2*(selected_probability*conditional).sum(-1)).clamp_min(0)
        coefficient_sq = coefficient_gradient.square().sum(-1)
        joint_sq = (base_sq+target.square().sum(-1)-2*(selected_probability*target).sum(-1)).clamp_min(0)
        inner = ((selected_probability-conditional)*coefficient_gradient).sum(-1)
        norm_error = max(norm_error, float((joint_sq-atom_sq-coefficient_sq-2*inner).abs().max()))
        atom_context_sq = coefficient_context_sq = joint_context_sq = torch.zeros_like(joint_sq)
        if classifier_weight is not None:
            weight = classifier_weight.float()
            selected_weight = weight[ids]
            atom_context = probability@weight-(conditional[..., None]*selected_weight).sum(1)
            coefficient_context = (coefficient_gradient[..., None]*selected_weight).sum(1)
            atom_context_sq = atom_context.square().sum(-1)
            coefficient_context_sq = coefficient_context.square().sum(-1)
            joint_context_sq = (atom_context+coefficient_context).square().sum(-1)
        values = torch.stack((joint_ce, atom_ce, coefficient_ce, entropy, atom_sq, coefficient_sq,
            joint_sq, atom_context_sq, coefficient_context_sq, joint_context_sq), -1).double()
        positions = torch.arange(first, first+len(rows), device=x.device).remainder(depth)
        sums.index_add_(0, positions, values)
        counts.index_add_(0, positions, torch.ones_like(positions))
    assert normalization_error < 2.e-5 and ce_error < 1.e-5 and norm_error < 2.e-5
    return dict(sums=sums.cpu().tolist(), counts=counts.cpu().tolist(), metric_names=metrics,
        maximum_ce_decomposition_error=ce_error, maximum_gradient_decomposition_error=norm_error,
        maximum_target_mass_error=normalization_error,
        objective_weights=dict(atom_marginal=1., coefficient_conditional=1., depth=[1./depth]*depth),
        additional_training_losses=0)
