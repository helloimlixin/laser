import torch

from src import ffhq_v4_archived as archived
from src.training.rqtransformer import (
    CompoundLaserRQTransformer, LaserAux, compound_objective,
)
from tests.test_compound_pair_autoregressive import tiny_aux, tiny_config


def test_ffhq_recipe_matches_archived_logits_objective_and_gradients():
    torch.manual_seed(21)
    config = tiny_config(depth=4)
    config.block_size = [2, 2, 4]
    # Exercise actual class conditioning as well as all four sparse depths.
    config.vocab_size_cond = 1000
    original = archived.CompoundLaserRQTransformer(
        config, 7, 5, micro_transformer_layers=2,
        depth_specific_coeff_heads=True,
    ).eval()
    current = CompoundLaserRQTransformer(
        config, 7, 5, micro_transformer_layers=2,
        depth_specific_coeff_heads=True, pair_autoregressive=True,
        mask_seen_atoms_training=False,
    ).eval()
    current.load_state_dict(original.state_dict(), strict=True)
    aux = tiny_aux(depth=4)
    atoms = torch.arange(32).reshape(2, 2, 2, 4).remainder(7)
    packed = atoms * 5 + torch.arange(32).reshape_as(atoms).remainder(5)
    labels = torch.tensor([5, 999])
    left = original(packed, model_aux=aux, cond=labels)
    right = current(packed, model_aux=aux, cond=labels)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    changed = current(packed, model_aux=aux, cond=torch.tensor([6, 998]))
    assert not torch.equal(right['atom_logits'], changed['atom_logits'])
    targets = torch.randn(2, 2, 2, 4, 5).softmax(-1)
    physical = aux.compound_embeddings(atoms, packed.remainder(5))
    settings = dict(atom_weight=1.5, geometry_weight=.05, accumulation=1,
        distribution_geometry=True, geometry_dictionary=aux.dictionary,
        geometry_coeff_bins=aux.coeff_bins, geometry_coeff_scales=aux.coeff_scales,
        geometry_top_k=4)
    a, _ = archived.compound_objective(left['atom_logits'], left['coeff_logits'],
        None, atoms, targets, physical, **settings)
    b, _ = compound_objective(right['atom_logits'], right['coeff_logits'],
        None, atoms, targets, physical, **settings)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.backward()
    b.backward()
    for (name, p), (other, q) in zip(original.named_parameters(), current.named_parameters()):
        assert name == other
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_normalized_coefficient_targets_and_stochastic_context_match_ffhq():
    aux = tiny_aux(depth=4)
    aux.coeff_vocab_size = 5
    aux.sparsity_level = 4
    aux.soft_target_physical = False
    # Values outside the bin interval also remain continuous target values;
    # quantization does not silently clamp the soft-target distribution.
    coefficients = torch.tensor([[[[-2.5, -.3, .7, 2.5]]]])
    for stochastic in (False, True):
        torch.manual_seed(33)
        expected = archived.LaserAux.compound_coeff_ids(
            aux, coefficients, stochastic=stochastic, temp=.5,
        )
        torch.manual_seed(33)
        actual = LaserAux.compound_coeff_ids(
            aux, coefficients, stochastic=stochastic, temp=.5, hard=False,
        )
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    _, clamped_targets = LaserAux.compound_coeff_ids(
        aux, coefficients.clamp(-1, 1), stochastic=False, temp=.5,
    )
    assert not torch.equal(actual[1], clamped_targets)


def test_upload_fallback_does_not_require_filesystem_metadata(monkeypatch, tmp_path):
    from src.training import rqtransformer as training
    source = tmp_path / 'source.pt'
    source.write_bytes(b'checkpoint')
    def unsupported(*args, **kwargs):
        raise PermissionError('metadata operations are unsupported')
    monkeypatch.setattr(training.os, 'link', unsupported)
    monkeypatch.setattr(training.shutil, 'copy2', unsupported)
    for function in (training.snapshot_checkpoint, training._replace_hard_link):
        destination = tmp_path / function.__name__
        function(source, destination)
        assert destination.read_bytes() == b'checkpoint'
