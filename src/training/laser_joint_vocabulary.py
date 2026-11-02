"""Training-only coefficient calibration for a bounded joint vocabulary."""
import heapq

import torch


@torch.no_grad()
def fit_coefficient_candidates(atoms, coefficients, num_atoms, *, counts=(2, 4, 6, 8), iterations=25,
                               prior_weight=4.):
    """Fit signed 1D Lloyd centers; retain a prior for rarely observed atoms."""
    atoms, coefficients = atoms.flatten().long(), coefficients.flatten().float()
    books, distortions = {}, {}
    for count in counts:
        negative = coefficients[coefficients < 0]
        positive = coefficients[coefficients > 0]
        if not len(negative) or not len(positive):
            raise ValueError('Calibration requires both coefficient signs')
        positions = torch.linspace(.1, .9, count // 2, device=coefficients.device)
        initial = torch.cat((torch.quantile(negative, positions), torch.quantile(positive, positions)))
        centers = initial.expand(num_atoms, -1).clone()
        for _ in range(iterations):
            distance = (coefficients[:, None] - centers[atoms]).square()
            # Preserve a negative and positive half even for under-observed atoms.
            same_sign = (coefficients[:, None] < 0) == (centers[atoms] < 0)
            chosen = distance.masked_fill(~same_sign, torch.inf).argmin(-1)
            ids = atoms * count + chosen
            frequency = torch.bincount(ids, minlength=num_atoms * count).reshape(num_atoms, count)
            sums = torch.bincount(ids, weights=coefficients, minlength=num_atoms * count).reshape_as(centers)
            updated = ((sums + prior_weight * initial) / (frequency + prior_weight)).sort(-1).values
            change = (updated - centers).abs().max()
            centers = updated
            if change < 1.e-6:
                break
        error = (coefficients[:, None] - centers[atoms]).square().min(-1).values
        books[count] = centers
        distortions[count] = torch.bincount(atoms, weights=error, minlength=num_atoms)
    return books, distortions


def allocate_coefficients(distortions, budget):
    """Greedily spend equal two-entry increments on measured distortion gains."""
    sizes = sorted(distortions)
    if sizes != list(range(sizes[0], sizes[-1] + 1, 2)):
        raise ValueError('Candidate sizes must increase by two')
    error = {k: v.detach().cpu().double().tolist() for k, v in distortions.items()}
    n = len(error[sizes[0]])
    if not n * sizes[0] <= budget <= n * sizes[-1] or (budget - n * sizes[0]) % 2:
        raise ValueError('Budget incompatible with candidate sizes')
    allocation = [sizes[0]] * n
    queue = [(error[sizes[1]][a] - error[sizes[0]][a], a) for a in range(n)]
    heapq.heapify(queue)
    for _ in range((budget - n * sizes[0]) // 2):
        _, atom = heapq.heappop(queue)
        old = allocation[atom]
        allocation[atom] += 2
        if allocation[atom] < sizes[-1]:
            heapq.heappush(queue, (error[old + 4][atom] - error[old + 2][atom], atom))
    return allocation


def allocate_shared_coefficients(depth_distortions, budget, *, weights=None):
    """Allocate one atom/level layout for every residual depth.

    The classifier is shared across depth positions. A common allocation keeps
    integer IDs attached to the same atom and signed level at every depth.
    Equal default weights minimize the same sum of physical distortion used by
    reconstruction calibration; this retains the existing greedy allocator.
    """
    if not depth_distortions:
        raise ValueError('At least one residual depth is required')
    sizes = sorted(depth_distortions[0])
    shape = depth_distortions[0][sizes[0]].shape
    if weights is None:
        weights = [1.] * len(depth_distortions)
    if len(weights) != len(depth_distortions):
        raise ValueError('One weight per residual depth is required')
    weights = torch.as_tensor(weights, dtype=torch.float64)
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Depth weights must be finite, nonnegative, and nonzero')
    total = {size:torch.zeros(shape, dtype=torch.float64) for size in sizes}
    for weight, errors in zip(weights, depth_distortions):
        if sorted(errors) != sizes:
            raise ValueError('Every depth must use the same candidate sizes')
        for size in sizes:
            value = errors[size].detach().cpu().double()
            if value.shape != shape or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError('Distortions must be finite nonnegative arrays of equal shape')
            total[size] += weight * value
    return allocate_coefficients(total, budget)


def pack_codebook(dictionary, depth_books, allocations):
    atom_rows, coefficient_rows = [], []
    for books, allocation in zip(depth_books, allocations):
        atoms, values = [0], [dictionary.new_zeros(1)]
        for atom, count in enumerate(allocation):
            atoms.extend([atom] * count)
            values.append(books[count][atom])
        atom_rows.append(torch.tensor(atoms, device=dictionary.device, dtype=torch.long))
        coefficient_rows.append(torch.cat(values))
    return dict(kind='laser_joint_atom_coefficient_rq', format_version=1,
        dictionary=dictionary.detach().cpu(), atom_ids=torch.stack(atom_rows).cpu(),
        coefficients=torch.stack(coefficient_rows).cpu())
