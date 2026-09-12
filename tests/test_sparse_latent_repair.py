import torch
from src.sparse_latent_repair import SparseLatentRepair, sample_anchored_span, synthetic_corruption
from tests.test_coefficient_history_training import tiny_model


def test_repair_starts_as_exact_identity_and_backpropagates_through_frozen_decoder():
    torch.manual_seed(7)
    repair = SparseLatentRepair(channels=4, width=12, layers=2, heads=3, height=2, grid_width=2)
    clean = torch.randn(3, 4, 2, 2)
    repair.set_normalization(clean)
    noisy = clean + torch.randn_like(clean) * .2
    decoder = torch.nn.Conv2d(4, 3, 1).requires_grad_(False)
    original = {k: v.clone() for k, v in decoder.state_dict().items()}
    assert torch.equal(repair(noisy), noisy)
    loss = (decoder(repair(noisy)) - decoder(clean)).square().mean()
    loss.backward()
    assert repair.output.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in decoder.parameters())
    torch.optim.Adam(repair.parameters(), lr=.01).step()
    assert not torch.equal(repair(noisy), noisy)
    assert all(torch.equal(v, original[k]) for k, v in decoder.state_dict().items())


def test_full_pair_rollout_cannot_read_targets_within_generated_span():
    torch.manual_seed(8)
    model, aux = tiny_model()
    original = torch.tensor([[[[8, 19], [35, 44]]]])
    modified = original.clone()
    modified[:, :, 1] = torch.tensor([[[[1, 53]]]])
    uniform = torch.rand(1, 2, 2)
    first = sample_anchored_span(model, aux, original, 1, 1, uniform, top_k=7)
    second = sample_anchored_span(model, aux, modified, 1, 1, uniform, top_k=7)
    assert torch.equal(first, second)
    assert torch.equal(first[:, :, :1], original[:, :, :1])
    assert (first[:, :, 1, 0] // 8 != first[:, :, 1, 1] // 8).all()
    assert all(p.grad is None for p in model.parameters())


def test_synthetic_errors_preserve_input_and_valid_unique_supports():
    original = torch.randint(2048, (128, 8, 8, 4)) + torch.arange(4) * 2048
    before = original.clone()
    neighbors = (torch.arange(32)[:, None] + torch.arange(1, 9)[None]) % 32
    result = synthetic_corruption(original, neighbors, torch.Generator().manual_seed(3))
    assert torch.equal(original, before)
    assert (result != original).any()
    assert result.min() >= 0 and result.max() < 32 * 2048
    sorted_atoms = (result // 2048).sort(-1).values
    assert (sorted_atoms[..., 1:] != sorted_atoms[..., :-1]).all()
