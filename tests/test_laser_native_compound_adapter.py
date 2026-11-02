from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.models.laser_native_compound_adapter import build_native_compound_model
from src.training.laser_compound_replay_recipe import CompoundReplayRecipe


def case():
    torch.manual_seed(261016)
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4],
        embed_dim=12, input_embed_dim=8, shared_tok_emb=True, shared_cls_emb=True,
        input_emb_vqvae=True, head_emb_vqvae=True, cumsum_depth_ctx=True,
        vocab_size=11, vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.))))
    model = build_native_compound_model(arch, num_atoms=11, coefficient_bins=33).eval()
    with torch.no_grad():
        model.input_mlp.bias.normal_(std=.2)
        model.head_mlp.bias.normal_(std=.2)
    aux = SimpleNamespace(dictionary=torch.nn.functional.normalize(torch.randn(8, 11), dim=0),
        num_atoms=11, coeff_vocab_size=33, sparsity_level=4,
        coeff_bins=torch.linspace(-3, 3, 33), coeff_scales=torch.tensor([3., 2., 1., .5]))
    aux.compound_embeddings = lambda atoms, ids: aux.dictionary.T[atoms]*(aux.coeff_bins[ids]*aux.coeff_scales)[..., None]
    tokens = torch.empty(2, 2, 2, 8, dtype=torch.long)
    tokens[..., 0::2] = torch.tensor([0, 1, 2, 3])
    tokens[..., 1::2] = 11+torch.tensor([10, 15, 20, 23])
    return arch, model, aux, tokens, torch.tensor([0, 999])


@torch.no_grad()
def test_archived_parameter_keys_and_full_logits_match_with_trained_projection_bias():
    from src.training.rqtransformer import CompoundLaserRQTransformer
    from src.models.rqtransformer.configs import RQTransformerConfig
    arch, native, aux, tokens, labels = case()
    archived = CompoundLaserRQTransformer(RQTransformerConfig.create(arch), 11, 33,
        micro_transformer_layers=2, depth_specific_coeff_heads=True, pair_autoregressive=True).eval()
    archived.load_state_dict(native.state_dict(), strict=True)
    assert list(dict(archived.named_parameters())) == list(dict(native.named_parameters()))
    expected, actual = archived(native.pack(tokens), aux, labels), native(tokens, aux, labels)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1.e-5, atol=3.e-6)


@torch.no_grad()
def test_native_cache_and_selected_atom_causality():
    _, model, aux, truth, labels = case()
    expected = model(truth, aux, labels)
    generated = torch.zeros_like(truth)
    generated[..., 1::2] = 11+16
    model.init_cache()
    for h in range(2):
        for w in range(2):
            for event in range(8):
                value = model.cached_forward(generated[:, :h+1], aux, labels, sample_loc=(h, w, event))
                key = 'atom_logits' if event%2 == 0 else 'coeff_logits'
                actual = value[:, :11] if event%2 == 0 else value[:, 11:]
                torch.testing.assert_close(actual, expected[key][:, h, w, event//2], rtol=1.e-5, atol=3.e-6)
                generated[:, h, w, event] = truth[:, h, w, event]
    changed = truth.clone()
    changed[:, 0, 0, 4] = 9
    actual = model(changed, aux, labels)
    torch.testing.assert_close(actual['atom_logits'][:, 0, 0, :3], expected['atom_logits'][:, 0, 0, :3], rtol=0, atol=0)
    torch.testing.assert_close(actual['coeff_logits'][:, 0, 0, :2], expected['coeff_logits'][:, 0, 0, :2], rtol=0, atol=0)
    assert not torch.allclose(actual['coeff_logits'][:, 0, 0, 2], expected['coeff_logits'][:, 0, 0, 2])


def test_weighted_categorical_loss_and_gradient_match_archived_objective():
    from src.training.rqtransformer import compound_objective
    _, model, aux, tokens, labels = case()
    profile = CompoundReplayRecipe.__new__(CompoundReplayRecipe)
    profile.recipe = OmegaConf.create(dict(atom_ce_weight=1.5))
    output = model(tokens, aux, labels)
    probabilities = torch.softmax(torch.randn(2, 2, 2, 4, 33), -1)
    target = dict(atoms=tokens[..., 0::2], coefficient_probabilities=probabilities)
    actual = profile.loss(model, output, target)
    expected, _ = compound_objective(output['atom_logits'], output['coeff_logits'], None,
        target['atoms'], probabilities, torch.zeros_like(target['atoms'], dtype=torch.float32),
        atom_weight=1.5, geometry_weight=0., accumulation=1)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_soft_targets_use_continuous_normalized_source_values_and_preserve_support():
    profile = CompoundReplayRecipe.__new__(CompoundReplayRecipe)
    profile.recipe = OmegaConf.create(dict(targets=dict(temperature=.03125)))
    aux = SimpleNamespace(num_atoms=5, coeff_vocab_size=33, sparsity_level=4,
                          coeff_bins=torch.linspace(-3, 3, 33))
    atoms = torch.tensor([[[[0, 1, 2, 3]]]])
    values = torch.tensor([[[[.03, -.04, .11, -.12]]]])
    profile.clean_components = lambda tokenizer, images: (atoms, values)
    targets, codes = profile.targets(aux, None)
    expected = (-(values[..., None]-aux.coeff_bins).square()/.03125).softmax(-1)
    torch.testing.assert_close(targets['coefficient_probabilities'], expected, rtol=0, atol=0)
    assert torch.equal(codes[..., 0::2], atoms)
    assert ((codes[..., 1::2]-5 >= 0) & (codes[..., 1::2]-5 < 33)).all()
    assert profile.verify_targets(aux, targets, codes)['sampled_trajectories'] == 0


def test_fresh_compound_initialization_keeps_the_reference_backbone_exact():
    from src.training.rq_imagenet_recipes import RQRecipe
    from rqvae.utils.config import augment_arch_defaults
    arch, _, _, _, _ = case()
    arch = augment_arch_defaults(arch)
    reference_arch = OmegaConf.create(OmegaConf.to_container(arch))
    reference_arch.vocab_size = 13
    profile = CompoundReplayRecipe.__new__(CompoundReplayRecipe)
    profile.recipe = OmegaConf.create(dict(tokenizer=dict(num_atoms=11, coefficient_bins=33),
        initialization=dict(paired_fresh_backbone=True, reference_vocab_size=13)))
    torch.manual_seed(261001)
    reference = RQRecipe.model(profile, reference_arch)
    torch.manual_seed(261001)
    model = profile.model(arch)
    for name, tensor in reference.state_dict().items():
        expected = tensor[:11] if name.startswith('classifier.linear.') else tensor
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
    with torch.device('meta'):
        restored = profile.model(arch)
    restored.load_state_dict(model.state_dict(), strict=True, assign=True)
    assert restored.coeff_classifier[0][1].weight.device.type == 'cpu'
