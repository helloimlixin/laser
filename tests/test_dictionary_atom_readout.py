import torch
from torch import nn
from src.models.dictionary_atom_readout import DictionaryAtomReadout
from src.models.dictionary_atom_readout import ClusteredAtomReadout, cluster_dictionary_directions
from src.models.dictionary_atom_readout import install_projected_dictionary_readout
from src.orthogonal_sparse_codec import ordered_support_basis


class Original(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(12)
        self.linear = nn.Linear(12, 7)


def test_direction_sign_symmetry_and_shared_parameters():
    torch.manual_seed(9)
    original = Original()
    dictionary = torch.nn.functional.normalize(torch.randn(4, 7), dim=0)
    dictionary[:, 1] = -dictionary[:, 0]
    head = DictionaryAtomReadout(original, dictionary)
    assert head.layer_norm is original.layer_norm and head.linear is original.linear
    x = torch.randn(2, 3, 12)
    correction = head(x)-original.linear(original.layer_norm(x))
    torch.testing.assert_close(correction[..., 0], correction[..., 1])
    head(x).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_full_and_single_position_outputs_and_checkpoint_restore():
    torch.manual_seed(7)
    dictionary = torch.nn.functional.normalize(torch.randn(4, 7), dim=0)
    head = DictionaryAtomReadout(Original(), dictionary).eval()
    x = torch.randn(2, 3, 12)
    torch.testing.assert_close(head(x)[:, 1], head(x[:, 1]))
    restored = DictionaryAtomReadout(Original(), dictionary).eval()
    restored.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), head(x), rtol=0, atol=0)
    assert 'dictionary' not in head.state_dict()


def test_coarse_correction_starts_exact_and_routes_cached_depth():
    original = Original()
    head = ClusteredAtomReadout(original, torch.tensor([0, 0, 1, 1, 2, 2, 2]), depth=3, groups=3)
    x = torch.randn(2, 3, 12)
    torch.testing.assert_close(head(x), original.linear(original.layer_norm(x)), rtol=0, atol=0)
    with torch.no_grad():
        head.coarse[2].bias.copy_(torch.tensor([1., 2., 3.]))
    head.active_depth = 2
    torch.testing.assert_close(head(x)[:, 2], head(x[:, 2]))
    labels = cluster_dictionary_directions(torch.tensor([[1., -1., 0., 0.], [0., 0., 1., -1.]]), groups=2)
    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_projected_readout_excludes_current_atom_and_matches_cached_positions():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = Original()

        def forward(self, packed, hidden):
            return self.classifier(hidden)

        def cached_head_output(self, packed, aux, cond, location, amp=True):
            return cond

    torch.manual_seed(21)
    model = Model().eval()
    dictionary = torch.nn.functional.normalize(torch.randn(4, 7), dim=0)
    install_projected_dictionary_readout(model, dictionary)
    packed = torch.tensor([[[[[0., 1.], [2., 1.], [4., 1.]]]]])
    hidden = torch.randn(1, 1, 1, 3, 12)
    projected = []
    model.classifier.query.register_forward_hook(lambda m, x, y: projected.append(y))
    full = model(packed, hidden)
    basis, _ = ordered_support_basis(dictionary.T[packed[..., 0].long()])
    for d in range(1, 3):
        dot = (projected[0][..., d, :].unsqueeze(-2)*basis[..., :d, :]).sum(-1)
        torch.testing.assert_close(dot, torch.zeros_like(dot), atol=1e-7, rtol=0)
    changed = packed.clone()
    changed[..., 1, 0] = 3
    torch.testing.assert_close(model(changed, hidden)[..., :2, :], full[..., :2, :], atol=0, rtol=0)
    for d in range(3):
        h = model.cached_head_output(packed, None, hidden[:, 0, 0, d], (0, 0, d))
        torch.testing.assert_close(model.classifier(h), full[:, 0, 0, d], atol=3e-6, rtol=1e-5)
