import io
import hashlib

import pytest
import torch

from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ, SharedAtomCoefficientRQ
from src.training.laser_shared_joint_vocabulary import pack_shared_codebook, refit_shared_book


def book(depth=4):
    return SharedAtomCoefficientRQ(torch.eye(3),
        torch.tensor([0, 0, 0, 1, 1, 2, 2]),
        torch.tensor([0., -2., 2., -1., 1., -.5, .5]), depth)


def test_one_id_has_the_same_pair_and_vector_at_every_depth_after_reload():
    q = book()
    ids = torch.arange(q.vocab_size).long()
    atoms, coefficients = q.pairs_from_ids(ids)
    assert torch.equal(q.ids_from_pairs(atoms, coefficients), ids)
    expected = q.dictionary.T[atoms] * coefficients[:, None]
    for d in range(q.depth):
        assert torch.equal(q.embed_depth(ids, d), expected)
        assert q.atom_ids[d].data_ptr() == q.shared_atom_ids.data_ptr()
        assert q.coefficients[d].data_ptr() == q.shared_coefficients.data_ptr()
    stream = io.BytesIO()
    torch.save(q.state_dict(), stream)
    stream.seek(0)
    restored = book()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert set(restored.state_dict()) == {'dictionary', 'norms', 'shared_atom_ids', 'shared_coefficients'}
    codes = ids[:, None].expand(-1, q.depth)
    assert torch.equal(restored.embed(codes).sum(-2), expected * q.depth)


def test_shared_greedy_and_stochastic_targets_match_the_expanded_pair_reference():
    q = book()
    reference = JointAtomCoefficientRQ(q.dictionary, q.atom_ids, q.coefficients)
    x = torch.randn(2, 2, 2, 3, generator=torch.Generator().manual_seed(51))
    actual, expected = q.quantize(x), reference.quantize(x)
    assert torch.equal(actual['codes'], expected['codes'])
    assert torch.equal(actual['quantized'], q.embed(actual['codes']).sum(-2))
    for stochastic in (False, True):
        torch.manual_seed(7)
        p, codes = q.get_soft_codes(x, .2, stochastic=stochastic, chunk_size=3)
        torch.manual_seed(7)
        rp, rc = reference.get_soft_codes(x, .2, stochastic=stochastic, chunk_size=3)
        assert torch.equal(codes, rc) and torch.equal(p, rp)
        assert codes.shape == (2, 2, 2, 4)  # One integer per contribution.


def test_depth_dependent_pairs_are_rejected_even_when_layout_is_shared():
    q = book()
    values = q.coefficients.clone()
    values[1, 1] *= 2
    with pytest.raises(ValueError, match='same physical pair'):
        SharedAtomCoefficientRQ(q.dictionary, q.atom_ids, values, q.depth)


def test_pair_lookup_handles_shapes_and_rejects_unknown_physical_coefficients():
    q = book()
    ids = torch.tensor([[0, 4], [2, 6]])
    pairs = q.pairs_from_ids(ids)
    assert torch.equal(q.ids_from_pairs(*pairs), ids)
    with pytest.raises(ValueError, match='absent'):
        q.ids_from_pairs(torch.tensor([0]), torch.tensor([.123]))
    with pytest.raises(ValueError):
        q.ids_from_pairs(torch.tensor([-1]), torch.tensor([1.]))


def test_shared_fitting_updates_canonical_storage_and_handles_repeated_ids():
    q = book()
    codes = torch.randint(0, q.vocab_size, (80, q.depth), generator=torch.Generator().manual_seed(5))
    codes[:10] = 2
    x = q.embed(codes).sum(-2) * 1.1
    report = refit_shared_book(q, x, codes)
    assert report['fixed_assignment_mse_after'] < report['fixed_assignment_mse_before']
    for d in range(1, q.depth):
        assert torch.equal(q.expanded_codebook(d), q.expanded_codebook(0))


def test_artifact_contains_one_pair_table_and_rejects_aliases():
    q = book()
    spec = pack_shared_codebook(q.dictionary, q.atom_ids, q.coefficients, q.depth)
    assert spec['atom_ids'].shape == spec['coefficients'].shape == (q.vocab_size,)
    assert spec['unique_physical_banks'] == 1 and spec['format_version'] == 4
    values = q.shared_coefficients.clone()
    values[2] = values[1]
    with pytest.raises(ValueError, match='distinct'):
        pack_shared_codebook(q.dictionary, q.shared_atom_ids, values, q.depth)


@pytest.mark.parametrize('codes', [torch.tensor([-1]), torch.tensor([7]), torch.tensor([1.])])
def test_invalid_pair_ids_are_rejected(codes):
    with pytest.raises(ValueError):
        book().pairs_from_ids(codes)


@pytest.mark.parametrize('format_version', [2, 4])
def test_frozen_tokenizer_loads_canonical_book_and_checks_cnn_lineage(tmp_path, monkeypatch, format_version):
    from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer
    from src.models import laser_tokenizer_adapter

    class Backbone(torch.nn.Module):
        def __init__(self, checkpoint, **kwargs):
            super().__init__()
            self.encoder = self.decoder = self.quant_conv = self.post_quant_conv = torch.nn.Identity()
            self.register_buffer('dictionary', torch.eye(3))

    monkeypatch.setattr(laser_tokenizer_adapter, 'FrozenLaserTokenizer', Backbone)
    checkpoint = tmp_path / 'cnn.pt'
    torch.save({'state_dict': {'quantizer.dictionary': torch.eye(3)}}, checkpoint)
    q = book()
    spec = pack_shared_codebook(q.dictionary, q.atom_ids, q.coefficients, q.depth,
        learned_compact_dictionary=True, encoder_decoder_frozen=True,
        source_stage1_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    if format_version == 2:
        spec.update(format_version=2, atom_ids=q.atom_ids.clone(), coefficients=q.coefficients.clone())
    path = tmp_path / 'pairs.pt'
    torch.save(spec, path)
    aux = FrozenJointLaserTokenizer(checkpoint, path)
    assert type(aux.quantizer) is (SharedAtomCoefficientRQ if format_version == 4 else JointAtomCoefficientRQ)
    codes = torch.randint(q.vocab_size, (2, 8, 8, q.depth))
    assert torch.equal(aux.decode_code(codes), q.embed(codes).sum(-2).permute(0, 3, 1, 2))
    assert all(not p.requires_grad for p in aux.parameters())
    if format_version == 4:
        invalid = dict(spec, shared_codebook_across_depths=False)
        torch.save(invalid, path)
        with pytest.raises(ValueError, match='canonical'):
            FrozenJointLaserTokenizer(checkpoint, path)
    spec['source_stage1_checkpoint_sha256'] = 'bad'
    torch.save(spec, path)
    with pytest.raises(ValueError, match='SHA256'):
        FrozenJointLaserTokenizer(checkpoint, path)


def test_fit_pipeline_saves_and_verifies_a_larger_single_bank(tmp_path):
    import json
    import numpy as np
    from scripts.tools.fit_shared_pair_vocabulary import fit

    q = book()
    source = pack_shared_codebook(q.dictionary, q.atom_ids, q.coefficients, q.depth)
    source_path = tmp_path / 'source.pt'
    torch.save(source, source_path)
    paths = [tmp_path / 'fit.npy', tmp_path / 'held.npy']
    rng = np.random.default_rng(51)
    for path, count in zip(paths, (100, 40)):
        np.save(path, rng.normal(size=(count, 3)).astype('float32'))
    config = dict(source_codebook=str(source_path), depth=4, vocabulary=9,
        chunk_size=11, coefficient_candidates=[2, 4], fitting_passes=1,
        stage1_checkpoint_sha256='test', fitting_images=100)
    fit(config, paths, tmp_path, torch.device('cpu'))
    spec = torch.load(tmp_path / 'joint-codebook.pt', weights_only=True)
    assert spec['atom_ids'].shape == spec['coefficients'].shape == (9,)
    result = json.loads((tmp_path / 'verification.json').read_text())
    assert result['passed'] and result['unique_physical_banks'] == 1
