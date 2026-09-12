import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src import ffhq_v4_archived as archive
from src.church_ffhq_archived import (
    ARCHIVE_SHA256, ROOT, FullBatchEpochStream, make_prior, targets,
    cosine_lr, church_config, objective,
)


def tiny():
    cfg = archive.RQTransformerConfig.create(OmegaConf.create({
        'block_size': [2, 2, 4], 'embed_dim': 24, 'input_embed_dim': 4,
        'shared_tok_emb': True, 'shared_cls_emb': True,
        'input_emb_vqvae': True, 'head_emb_vqvae': True, 'cumsum_depth_ctx': True,
        'vocab_size': 7, 'vocab_size_cond': 1, 'block_size_cond': 1,
        'body': {'n_layer': 2, 'block': {'n_head': 3, 'resid_pdrop': 0.}},
        'head': {'n_layer': 4, 'block': {'n_head': 3, 'resid_pdrop': 0.}},
    }))
    model = make_prior(config=cfg, num_atoms=7, coeff_vocab_size=8).eval()
    dictionary = torch.nn.functional.normalize(torch.randn(4, 7), dim=0)
    aux = SimpleNamespace(dictionary=dictionary, coeff_bins=torch.linspace(-3, 3, 8),
        coeff_scales=torch.tensor([7., 4., 2.5, 1.6]), coeff_vocab_size=8)
    aux.compound_embeddings = lambda a, c: dictionary.t()[a] * (aux.coeff_bins[c] * aux.coeff_scales)[..., None]
    aux.compound_coeff_ids = lambda *a, **kw: archive.LaserAux.compound_coeff_ids(aux, *a, **kw)
    atoms = torch.tensor([0, 2, 4, 6]).expand(2, 2, 2, 4).clone()
    packed = atoms * 8 + torch.randint(8, atoms.shape)
    return model, aux, packed


def test_exact_archive_and_official_church_dimensions_without_checkpoint(monkeypatch):
    assert hashlib.sha256((ROOT/'src/ffhq_v4_archived.py').read_bytes()).hexdigest() == ARCHIVE_SHA256
    def forbidden(*args, **kwargs):
        raise AssertionError('Scratch construction must not load weights')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.nn.Module, 'load_state_dict', forbidden)
    with torch.device('meta'):
        control, looped = make_prior(), make_prior('looped')
    assert type(control) is archive.CompoundLaserRQTransformer
    assert list(control.block_size) == [8, 8, 4]
    assert len(control.body_transformer.blocks) == 24
    assert len(control.head_transformer.blocks) == 4
    assert control.config.embed_dim == 1024
    assert sum(p.numel() for p in control.parameters()) == 404738048
    assert sum(p.numel() for p in looped.parameters()) == 404738050


def test_archived_targets_use_normalized_soft_distribution_and_random_context():
    torch.manual_seed(6)
    _, aux, packed = tiny()
    atoms = packed // 8
    normalized = torch.full_like(atoms, .3, dtype=torch.float32)
    physical = normalized * aux.coeff_scales
    expected = (-(normalized[..., None] - aux.coeff_bins).square()/.5).softmax(-1)
    deterministic, probs = targets(aux, atoms, physical, stochastic=False)
    torch.testing.assert_close(probs, expected)
    assert torch.equal(deterministic % 8, expected.argmax(-1))
    torch.manual_seed(19)
    a, _ = targets(aux, atoms, physical)
    torch.manual_seed(20)
    b, _ = targets(aux, atoms, physical)
    assert not torch.equal(a % 8, b % 8)
    assert torch.equal(a // 8, atoms)


@pytest.mark.parametrize('gates', [None, (0., 0.), (.12, -.08)])
def test_archived_teacher_and_cache_agree_and_pairs_condition_future(gates):
    torch.manual_seed(11)
    model, aux, packed = tiny()
    if gates is not None:
        from src.church_epoch50_loop import GatedDepthLoop
        model.head_transformer = GatedDepthLoop(list(model.head_transformer.blocks))
        model.head_transformer.loop_gates.data.copy_(torch.tensor(gates))
        model.eval()
    with torch.no_grad():
        teacher = model(packed, model_aux=aux)
        # The archived implementation does not apply the later training mask.
        assert torch.isfinite(teacher['atom_logits']).all()
        changed = packed.clone()
        changed[:, 0, 0, 0] = (packed[:, 0, 0, 0] // 8)*8 + (packed[:, 0, 0, 0]+1)%8
        other = model(changed, model_aux=aux)
        for key in teacher:
            assert torch.equal(teacher[key][:, 0, 0, 0], other[key][:, 0, 0, 0])
            assert not torch.equal(teacher[key][:, 0, 0, 1], other[key][:, 0, 0, 1])
            assert not torch.equal(teacher[key][:, 0, 1, 0], other[key][:, 0, 1, 0])
        model.init_cache()
        atoms = packed // 8
        for h in range(2):
            for w in range(2):
                for d in range(4):
                    hidden = model.cached_head_output(packed, aux, None, (h, w, d), amp=False)
                    logits = model.classifier(hidden)
                    coefficient = model.coefficient_logits(hidden, aux.dictionary.t()[atoms[:, h, w, d]], d)
                    torch.testing.assert_close(logits, teacher['atom_logits'][:, h, w, d], atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(coefficient, teacher['coeff_logits'][:, h, w, d], atol=2e-6, rtol=2e-5)


def test_zero_gates_preserve_archived_samples_and_can_learn():
    from src.church_epoch50_loop import GatedDepthLoop
    torch.manual_seed(17)
    model, aux, packed = tiny()
    looped = copy.deepcopy(model)
    looped.head_transformer = GatedDepthLoop(list(looped.head_transformer.blocks))
    samples = []
    for m in (model, looped):
        m.eval()
        torch.manual_seed(7)
        samples.append(m.sample_compound(2, aux, atom_top_k=7, atom_top_p=1., coeff_top_p=.85, amp=False))
    assert all(torch.equal(a, b) for a, b in zip(*samples))
    looped.train()
    loss, metrics = objective(looped, aux, packed // 8,
        aux.coeff_bins[packed % 8] * aux.coeff_scales, .05)
    loss.backward()
    assert torch.isfinite(loss) and metrics['geometry'] > 0
    assert torch.isfinite(looped.head_transformer.loop_gates.grad).all()
    assert looped.head_transformer.loop_gates.grad.abs().sum() > 0


def test_official_cosine_schedule_and_full_batches_resume():
    assert cosine_lr(0, 100) == 5e-4
    assert cosine_lr(50, 100) == pytest.approx(2.5e-4)
    assert cosine_lr(100, 100) == cosine_lr(200, 100) == 0.
    stream = FullBatchEpochStream(126227, 0)
    count = 0
    while True:
        indices, _, ended = stream.next(256)
        assert len(indices) == 256
        count += 1
        if ended:
            break
    assert count == 493
    resumed = FullBatchEpochStream(126227, 99)
    resumed.load_state_dict(stream.state_dict())
    a, pa, ea = stream.next(256)
    b, pb, eb = resumed.next(256)
    assert torch.equal(a, b) and pa == pb and ea == eb
