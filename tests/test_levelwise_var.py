import torch

from scripts.train_official_rqtransformer_laser_stage2 import LevelwiseLaserVAR


class TinyAux(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("dictionary", torch.eye(4))
        self.register_buffer("coeff_bins", torch.tensor([-1.0, 0.0, 1.0]))
        self.register_buffer("coeff_scales", torch.tensor([1.0, 0.5]))

    def compound_embeddings(self, atoms, coeff_ids):
        vectors = self.dictionary.t()[atoms.long()]
        coefficients = self.coeff_bins[coeff_ids.long()]
        scales = self.coeff_scales.view(*([1] * (coefficients.ndim - 1)), -1)
        return vectors * (coefficients * scales).unsqueeze(-1)


def tiny_model():
    return LevelwiseLaserVAR(
        height=2,
        width=2,
        sparsity_level=2,
        input_dim=4,
        embed_dim=16,
        num_layers=2,
        num_heads=4,
        num_atoms=4,
        coeff_vocab_size=3,
        num_condition_classes=1,
        depth_specific_coeff_heads=True,
    )


def test_level_mask_is_parallel_within_level_and_causal_across_levels():
    mask = tiny_model().level_attention_allowed

    assert mask.shape == (8, 8)
    assert mask[:4, :4].all()
    assert not mask[:4, 4:].any()
    assert mask[4:, :].all()


def test_future_level_inputs_cannot_change_earlier_level_hidden():
    torch.manual_seed(3)
    model = tiny_model().eval()
    first = torch.randn(2, 1, 2, 2, 4)
    future_a = torch.zeros(2, 1, 2, 2, 4)
    future_b = torch.randn(2, 1, 2, 2, 4) * 100

    hidden_a = model.level_hidden(torch.cat((first, future_a), dim=1))
    hidden_b = model.level_hidden(torch.cat((first, future_b), dim=1))

    assert torch.allclose(hidden_a[:, 0], hidden_b[:, 0], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(hidden_a[:, 1], hidden_b[:, 1])


def test_levelwise_forward_and_two_step_parallel_sampling_shapes():
    torch.manual_seed(7)
    model = tiny_model().eval()
    aux = TinyAux()
    atoms = torch.randint(0, 4, (2, 2, 2, 2))
    coeff_ids = torch.randint(0, 3, atoms.shape)
    packed = atoms * 3 + coeff_ids

    outputs = model(packed, model_aux=aux, cond=torch.zeros(2, dtype=torch.long))
    sampled_atoms, sampled_coeffs = model.sample_compound(
        2,
        aux,
        cond=torch.zeros(2, dtype=torch.long),
        atom_top_k=4,
        atom_top_p=1.0,
        coeff_top_k=3,
        coeff_top_p=1.0,
        amp=False,
    )

    assert outputs["atom_logits"].shape == (2, 2, 2, 2, 4)
    assert outputs["coeff_logits"].shape == (2, 2, 2, 2, 3)
    assert sampled_atoms.shape == atoms.shape
    assert sampled_coeffs.shape == coeff_ids.shape
    assert (sampled_atoms[..., 0] != sampled_atoms[..., 1]).all()


def test_single_level_contribution_uses_that_levels_physical_scale():
    model = tiny_model()
    aux = TinyAux()
    atoms = torch.tensor([[[1]]])
    coeff_ids = torch.tensor([[[2]]])

    first = model.single_level_contributions(aux, atoms, coeff_ids, 0)
    second = model.single_level_contributions(aux, atoms, coeff_ids, 1)

    assert torch.equal(first, torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]]))
    assert torch.equal(second, 0.5 * first)
