"""Distributional geometry of complete, signed dictionary contributions.

The auxiliary energy distance compares p(atom, coefficient | history) with
the teacher's soft coefficient distribution in vector space. Atom candidates
are a detached top-k set plus the teacher exactly once; coefficient mass is
pooled into fixed adjacent bins. The original full-vocabulary CE remains the
primary objective. Distances divide out the physical depth scale, giving each
depth the same units without multiplying independent marginal expectations.
"""
import torch


def pool_coefficient_probabilities(probabilities, bins, groups=64):
    count = bins.numel()
    groups = min(groups, count)
    if groups < 2 or count % groups:
        raise ValueError('Coefficient groups must evenly divide the vocabulary')
    return (probabilities.reshape(*probabilities.shape[:-1], groups, count // groups).sum(-1),
            bins.float().reshape(groups, count // groups).mean(-1))


def compound_energy_distance(atom_logits, coefficient_logits, candidate_atoms,
                             teacher_atoms, teacher_probabilities, dictionary,
                             coefficient_bins, groups=64):
    """Energy distance on a restricted joint distribution, not a mean MSE.

    Shapes are N,C; N,C,V; N,C; N; N,V. Dictionary is feature-by-atom.
    Candidate logits must mask duplicate teacher entries with -inf. The pooled
    finite joint supports are deterministic, so gradients flow through both
    conditionals without a discrete sampling/straight-through approximation.
    """
    with torch.autocast(atom_logits.device.type, enabled=False):
        return _energy_distance(atom_logits, coefficient_logits, candidate_atoms,
            teacher_atoms, teacher_probabilities, dictionary, coefficient_bins, groups)


def _energy_distance(atom_logits, coefficient_logits, candidate_atoms,
                     teacher_atoms, teacher_probabilities, dictionary, coefficient_bins, groups):
    atom_probabilities = atom_logits.float().softmax(-1)
    conditional, centers = pool_coefficient_probabilities(
        coefficient_logits.float().softmax(-1), coefficient_bins, groups)
    target, _ = pool_coefficient_probabilities(teacher_probabilities.float(), coefficient_bins, groups)
    joint = atom_probabilities[..., None] * conditional
    vectors = dictionary.T[candidate_atoms].float()
    teacher = dictionary.T[teacher_atoms].float()
    points = (vectors[..., None, :] * centers[None, None, :, None]).flatten(1, 2)
    targets = teacher[:, None, :] * centers[None, :, None]
    mass = joint.flatten(1, 2)
    # Smooth the norm only at zero. This is a negative-type distance and keeps
    # an identical distribution at zero energy distance, including mixtures.
    def distance(left, right):
        square = (left.square().sum(-1)[..., :, None] +
                  right.square().sum(-1)[..., None, :] -
                  2 * (left @ right.transpose(-1, -2)))
        return (square.clamp_min(0) + 1e-8).sqrt() - 1e-4
    cross = (mass * (distance(points, targets) @ target[..., None]).squeeze(-1)).sum(-1)
    predicted_self = (mass * (distance(points, points) @ mass[..., None]).squeeze(-1)).sum(-1)
    target_self = (target * (distance(targets, targets) @ target[..., None]).squeeze(-1)).sum(-1)
    return (2 * cross - predicted_self - target_self).mean()


def candidate_geometry_outputs(model, inputs, history, output, atoms, aux):
    """Replay only candidate coefficient prefixes through the trained head.

    A candidate changes the current atom input, never earlier pair inputs or
    the full-history context. Replaying the causal prefix produces the actual
    p(coefficient | candidate atom, history), with shared weights and gradients.
    No parameters, caches, training RNG streams, or full-vocabulary CE change.
    """
    devices = [inputs.device.index] if inputs.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        seed = 261001 + 131 * model.compound_geometry_cursor
        torch.random.default_generator.manual_seed(seed)
        if inputs.is_cuda:
            torch.cuda.manual_seed(seed)
        return _candidate_outputs(model, inputs, history, output, atoms, aux)


def _candidate_outputs(model, inputs, history, output, atoms, aux):
    config = model.compound_geometry_config
    batch, height, width, depths = atoms.shape
    sites = height * width
    take = min(config['sites_per_image'], sites)
    cursor = model.compound_geometry_cursor
    model.compound_geometry_cursor += 1
    selected = (torch.arange(take, device=atoms.device)[None] * 37 + cursor * 17 +
                torch.arange(batch, device=atoms.device)[:, None] * 13) % sites
    rows = (torch.arange(batch, device=atoms.device)[:, None] * sites + selected).flatten()
    native_inputs = inputs[rows]
    selected_atoms = atoms.reshape(batch * sites, depths)[rows]
    selected_a_logits = output['atom_logits'].reshape(batch * sites, depths, -1)[rows]
    selected_c_logits = output['coeff_logits'].reshape(batch * sites, depths, -1)[rows]
    selected_history = history.reshape(batch * sites, depths, -1)[rows]
    top_logits, candidates = selected_a_logits.float().topk(
        min(config['top_k'], model.num_atoms), -1)
    teacher_logits = selected_a_logits.float().gather(-1, selected_atoms[..., None])
    teacher_logits = teacher_logits.masked_fill(
        (candidates == selected_atoms[..., None]).any(-1, keepdim=True), -torch.inf)
    candidate_atoms = torch.cat((candidates, selected_atoms[..., None]), -1)
    candidate_logits = torch.cat((top_logits, teacher_logits), -1)
    count = candidate_atoms.shape[-1]
    coefficient_outputs = []
    for depth in range(depths):
        length = 2 * depth + 2
        values = native_inputs[:, None, :length].expand(-1, count, -1, -1).clone()
        vectors = aux.dictionary.T[candidate_atoms[:, depth]]
        # The native odd input contains selected_atom_projection(teacher).
        # Replacing that term alone preserves the complete earlier prefix.
        values[:, :, -1] = (native_inputs[:, None, length - 1] +
            model.selected_atom_projection(vectors) -
            model.selected_atom_projection(aux.dictionary.T[selected_atoms[:, depth]])[:, None])
        hidden = model.head_transformer(values.flatten(0, 1))[:, -1].reshape(-1, count, values.shape[-1])
        hidden = hidden + selected_history[:, depth, None] * model.history_gates[1]
        normalized = model.classifier.layer_norm(hidden)
        linear = model.classifier.linear
        logits = torch.nn.functional.linear(normalized, linear.weight[model.num_atoms:],
            None if linear.bias is None else linear.bias[model.num_atoms:])
        # Use the teacher branch already computed for the classification loss.
        logits = torch.where((candidate_atoms[:, depth] == selected_atoms[:, depth, None])[..., None],
                             selected_c_logits[:, depth, None], logits)
        coefficient_outputs.append(logits)
    return dict(geometry_atom_logits=candidate_logits.flatten(0, 1),
                geometry_coefficient_logits=torch.stack(coefficient_outputs, 1).flatten(0, 1),
                geometry_candidate_atoms=candidate_atoms.flatten(0, 1),
                geometry_teacher_atoms=selected_atoms.flatten(), geometry_selected_rows=rows)
