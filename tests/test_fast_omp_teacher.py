import pytest
import torch

from src.training import omp_math


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FP32 OMP kernel parity requires CUDA')
@pytest.mark.parametrize('depth', [1, 2, 3, 4])
def test_batched_small_solves_and_residuals_match_torch(depth):
    torch.manual_seed(261014)
    device = torch.device('cuda')
    # Exercise a partial block, several vocabulary tiles, and real Gram systems.
    dictionary = torch.nn.functional.normalize(torch.randn(32, 257, device=device), dim=0)
    gram = dictionary.T @ dictionary
    support = torch.stack([torch.randperm(257, device=device)[:depth] for _ in range(131)])
    active = gram[support[:, :, None], support[:, None, :]]
    chol = torch.linalg.cholesky(active)
    previous=chol[:,:depth-1,:depth-1].contiguous() if depth>1 else None
    extended,invalid=omp_math.extend_cholesky(previous,gram,
        support[:,:depth-1].contiguous(),support[:,-1].contiguous())
    assert not invalid.any()
    torch.testing.assert_close(extended,chol,rtol=2e-5,atol=2e-6)
    rhs = torch.randn(131, depth, 1, device=device)
    lower = omp_math.solve_lower(chol, rhs)
    coefficients = omp_math.solve_cholesky(chol, rhs)
    torch.testing.assert_close(lower, torch.linalg.solve_triangular(chol, rhs, upper=False), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(coefficients, torch.cholesky_solve(rhs, chol), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(active @ coefficients, rhs, rtol=2e-5, atol=2e-6)
    correlation = torch.randn(131, 257, device=device)
    values = coefficients.squeeze(-1)
    residual = omp_math.residual_correlation(correlation, values, gram, support)
    expected = correlation-values.unsqueeze(1).bmm(gram[support]).squeeze(1)
    torch.testing.assert_close(residual, expected, rtol=2e-5, atol=2e-6)
    available=torch.ones_like(correlation,dtype=torch.bool).scatter_(1,support,False)
    probabilities=omp_math.support_probabilities(correlation,available,.4)
    scores=correlation.square().masked_fill(~available,-torch.inf)
    expected_probabilities=((scores-scores.amax(-1,keepdim=True))/.4).softmax(-1)
    torch.testing.assert_close(probabilities,expected_probabilities,rtol=3e-5,atol=3e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA label batching requires CUDA')
def test_large_label_batches_preserve_prefix_conditionals_and_gradients():
    from src.training.omp_joint_targets import omp_bank_joint_targets, omp_joint_cross_entropy
    torch.manual_seed(261014)
    device = torch.device('cuda')
    atoms = torch.rand(1031, 16, 4, 41, device=device).argsort(-1)[..., 0]
    centers = torch.randn(1031, 16, 4, device=device)
    observed = atoms[:, 0]
    ids = torch.randint(67, (1031, 4), device=device)
    bins = torch.linspace(-3, 3, 67, device=device)
    small = omp_bank_joint_targets(atoms, centers, observed, ids, bins, temperature=.6, site_chunk_size=128)
    large = omp_bank_joint_targets(atoms, centers, observed, ids, bins, temperature=.6, site_chunk_size=1024)
    for key in ('atom_ids', 'atom_weights', 'coefficient_probabilities', 'trajectory_weights_given_atom'):
        torch.testing.assert_close(getattr(small, key), getattr(large, key), rtol=0, atol=0)
    a = torch.randn(1031, 4, 41, device=device, requires_grad=True)
    c = torch.randn(1031, 4, 67, device=device, requires_grad=True)
    losses = [omp_joint_cross_entropy(a, c, labels) for labels in (small, large)]
    gradients = [torch.autograd.grad(loss, (a, c), retain_graph=True) for loss in losses]
    for first, second in zip(gradients[0], gradients[1]):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
