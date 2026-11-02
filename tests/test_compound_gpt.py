"""Autoregressive and field-conditioning contracts of the flat compound GPT."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.models.compound_gpt import CompoundGPT, CompoundGPTConfig


def fixture():
    torch.manual_seed(41)
    config = CompoundGPTConfig(block_size=(1, 2, 4), num_atoms=19,
        coeff_vocab_size=8, dictionary_dim=12, num_classes=3,
        width=32, heads=4, layers=2, dropout=0.)
    model = CompoundGPT(config).eval()
    aux = SimpleNamespace(dictionary=F.normalize(torch.randn(12, 19), dim=0),
        coeff_bins=torch.linspace(-3., 3., 8), coeff_scales=torch.tensor([4., 2., 1., .5]))
    atoms = torch.tensor([[[[0, 1, 2, 3], [4, 5, 6, 7]]],
                          [[[8, 9, 10, 11], [12, 13, 14, 15]]]])
    bins = torch.randint(8, atoms.shape)
    packed = atoms * 8 + bins
    return model, aux, packed, torch.tensor([0, 2])


def test_current_coefficient_and_future_pairs_cannot_leak():
    model, aux, packed, labels = fixture()
    baseline = model(packed, model_aux=aux, cond=labels)
    changed = packed.clone().flatten(1)
    changed[:, 3] = (changed[:, 3] // 8) * 8 + (changed[:, 3] % 8 + 3) % 8
    changed[:, 4:] = changed[:, 4:].roll(1, 1)
    result = model(changed.reshape_as(packed), model_aux=aux, cond=labels)
    for key in ('atom_logits', 'coeff_logits'):
        torch.testing.assert_close(result[key].flatten(1, 3)[:, :4],
                                   baseline[key].flatten(1, 3)[:, :4])
    assert not torch.allclose(result['coeff_logits'].flatten(1, 3)[:, 4],
                              baseline['coeff_logits'].flatten(1, 3)[:, 4])


def test_current_atom_conditions_only_current_coefficient_and_future():
    model, aux, packed, labels = fixture()
    baseline = model(packed, model_aux=aux, cond=labels)
    changed = packed.clone()
    changed[:, 0, 0, 2] = 18 * 8 + packed[:, 0, 0, 2] % 8
    result = model(changed, model_aux=aux, cond=labels)
    torch.testing.assert_close(result['atom_logits'][:, 0, 0, :3],
                               baseline['atom_logits'][:, 0, 0, :3])
    torch.testing.assert_close(result['coeff_logits'][:, 0, 0, :2],
                               baseline['coeff_logits'][:, 0, 0, :2])
    assert not torch.allclose(result['coeff_logits'][:, 0, 0, 2],
                              baseline['coeff_logits'][:, 0, 0, 2])


@torch.no_grad()
def test_cached_teacher_prefix_matches_full_forward_across_site_boundary():
    model, aux, packed, labels = fixture()
    full = model(packed, model_aux=aux, cond=labels)
    atoms, _ = model.unpack(packed)
    cache = previous = None
    for position in range(model.events):
        hidden, cache = model.decode_step(previous, aux, labels, position, cache)
        logits = model.atom_classifier(hidden)
        depth = position % 4
        if depth:
            logits.scatter_(1, atoms.flatten(1)[:, position-depth:position], -float('inf'))
        torch.testing.assert_close(logits, full['atom_logits'].flatten(1, 3)[:, position], atol=3e-7, rtol=1e-5)
        coefficients = model.coefficient_logits(hidden, atoms.flatten(1)[:, position], aux)
        torch.testing.assert_close(coefficients, full['coeff_logits'].flatten(1, 3)[:, position], atol=3e-7, rtol=1e-5)
        previous = packed.flatten(1)[:, position]
        assert all(state[0].shape[2] == position + 1 for state in cache)


def test_both_losses_reach_shared_backbone_at_every_depth():
    model, aux, packed, labels = fixture()
    model.capture_shared_hidden = True
    result = model(packed, model_aux=aux, cond=labels)
    atoms, bins = model.unpack(packed)
    hidden = result['shared_hidden']
    for depth in range(4):
        for key, target in (('atom_logits', atoms), ('coeff_logits', bins)):
            logits = result[key][..., depth, :]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target[..., depth].flatten())
            gradient, = torch.autograd.grad(loss, hidden, retain_graph=True)
            assert gradient[..., depth, :].norm() > 0
    loss = sum(F.cross_entropy(result[key].flatten(0, 3), target.flatten())
               for key, target in (('atom_logits', atoms), ('coeff_logits', bins)))
    loss.backward()
    for name in ('atom_embedding.weight', 'coefficient_embedding.weight',
                 'physical_projection.weight', 'blocks.0.attn.qkv.weight',
                 'coefficient_atom_projection.weight'):
        assert model.get_parameter(name).grad.norm() > 0, name


@torch.no_grad()
def test_sampling_fields_in_range_and_distinct_support():
    model, aux, _, labels = fixture()
    atoms, bins = model.sample_compound(2, aux, cond=labels, amp=False)
    assert atoms.shape == bins.shape == (2, 1, 2, 4)
    assert 0 <= atoms.min() and atoms.max() < 19
    assert 0 <= bins.min() and bins.max() < 8
    ordered = atoms.sort(-1).values
    assert (ordered[..., 1:] != ordered[..., :-1]).all()


def test_checkpointed_backward_matches_uncheckpointed():
    model, aux, packed, labels = fixture()
    model.train()
    losses, gradients = [], []
    for enabled in (False, True):
        model.zero_grad(set_to_none=True)
        model.activation_checkpointing = enabled
        result = model(packed, model_aux=aux, cond=labels)
        loss = result['coeff_logits'].square().mean()
        loss.backward()
        losses.append(loss.detach())
        gradients.append(model.blocks[0].attn.qkv.weight.grad.clone())
    torch.testing.assert_close(*losses)
    torch.testing.assert_close(*gradients)


def test_default_is_approximately_1_4b_without_joint_vocabulary():
    with torch.device('meta'):
        model = CompoundGPT(CompoundGPTConfig())
    count = sum(p.numel() for p in model.parameters())
    assert 1.38e9 < count < 1.42e9, count
    assert len(model.blocks) == 26
    assert model.atom_embedding.num_embeddings == 16384
    assert model.coefficient_embedding.num_embeddings == 512
    assert model.atom_classifier.out_features == 16384
    assert model.coefficient_classifier[-1].out_features == 512


def test_incomplete_cache_is_rejected():
    model, aux, _, labels = fixture()
    with pytest.raises(ValueError, match='preceding pair'):
        model.decode_step(torch.tensor([1, 2]), aux, labels, 1)


def test_training_factory_and_parser_select_compound_gpt():
    from src.training.rqtransformer import build_model, build_parser
    args = build_parser().parse_args(['--checkpoint', '/tmp/unused.pt',
        '--data', '/tmp/unused', '--output', '/tmp/unused-output',
        '--backbone', 'gpt', '--compound-tokens', '--compound-pair-autoregressive',
        '--coeff-vocab-size', '512', '--sparsity-level', '4'])
    assert args.backbone == 'gpt' and args.compound_tokens and args.coeff_vocab_size == 512
    with torch.device('meta'):
        model = build_model(16896, 16384, backbone='gpt', compound=True,
            compound_pair_autoregressive=True, coeff_vocab_size=512, sparsity_level=4)
    assert isinstance(model, CompoundGPT)
    with pytest.raises(ValueError, match='plain ImageNet compound'):
        build_model(16896, 16384, backbone='gpt', compound=False)
