from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from src.training.rqtransformer import CompoundLaserRQTransformer, LaserAux
from src.church_ffhq_recipe import make_prior, early_decay_lr, recipe_targets
from src.coefficient_history_training import EpochStream
from src.models.rqtransformer.configs import RQTransformerConfig


def tiny():
    config = RQTransformerConfig.create(OmegaConf.create({
        'block_size': [2, 2, 4], 'embed_dim': 24, 'input_embed_dim': 4,
        'shared_tok_emb': True, 'shared_cls_emb': True,
        'input_emb_vqvae': True, 'head_emb_vqvae': True, 'cumsum_depth_ctx': True,
        'vocab_size': 7, 'vocab_size_cond': 1, 'block_size_cond': 1,
        'body': {'n_layer': 2, 'block': {'n_head': 3, 'resid_pdrop': 0.}},
        'head': {'n_layer': 3, 'block': {'n_head': 3, 'resid_pdrop': 0.}},
    }))
    model = CompoundLaserRQTransformer(config, 7, 8, pair_autoregressive=True,
               micro_transformer_layers=2, depth_specific_coeff_heads=True).eval()
    dictionary = torch.nn.functional.normalize(torch.randn(4, 7), dim=0)
    bins, scales = torch.linspace(-3, 3, 8), torch.tensor([7., 4., 2.5, 1.6])
    aux = SimpleNamespace(dictionary=dictionary, coeff_bins=bins, coeff_scales=scales,
                          coeff_vocab_size=8, sparsity_level=4, soft_target_physical=False)
    aux.compound_embeddings = lambda a, c: dictionary.t()[a] * (bins[c] * scales)[..., None]
    aux.compound_coeff_ids = lambda *a, **kw: LaserAux.compound_coeff_ids(aux, *a, **kw)
    atoms = torch.tensor([0, 2, 4, 6]).expand(2, 2, 2, 4).clone()
    tokens = atoms * 8 + torch.randint(8, atoms.shape)
    return model, aux, tokens


def test_full_spatial_and_depth_cache_matches_teacher_forcing():
    torch.manual_seed(12)
    model, aux, tokens = tiny()
    with torch.no_grad():
        teacher = model(tokens, model_aux=aux)
        model.init_cache()
        for site in range(4):
            h, w = divmod(site, 2)
            for d in range(4):
                hidden = model.cached_head_output(tokens, aux, None, (h, w, d), amp=False)
                atom_logits = model.classifier(hidden)
                if d:
                    atom_logits.scatter_(1, tokens[:, h, w, :d] // 8, -torch.inf)
                current_atom = tokens[:, h, w, d] // 8
                coeff_logits = model.coefficient_logits(hidden, aux.dictionary.t()[current_atom], d)
                torch.testing.assert_close(atom_logits, teacher['atom_logits'][:, h, w, d], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(coeff_logits, teacher['coeff_logits'][:, h, w, d], atol=1e-6, rtol=1e-5)


def test_future_pairs_cannot_affect_current_predictions():
    torch.manual_seed(13)
    model, aux, tokens = tiny()
    changed = tokens.clone().flatten(1)
    # Change current coefficient and all later pairs; current atom stays fixed.
    changed[:, 6:] = (changed[:, 6:] // 8) * 8 + (changed[:, 6:] + 3) % 8
    with torch.no_grad():
        a = model(tokens, model_aux=aux)
        b = model(changed.reshape_as(tokens), model_aux=aux)
    for key in ['atom_logits', 'coeff_logits']:
        torch.testing.assert_close(a[key].flatten(1, 3)[:, :7], b[key].flatten(1, 3)[:, :7], atol=0, rtol=0)
    assert not torch.equal(a['atom_logits'][:, 1], b['atom_logits'][:, 1])


def test_recovered_normalized_target_is_invariant_to_physical_depth_scale():
    model, aux, tokens = tiny()
    atoms = tokens // 8
    normalized = torch.full_like(tokens, .7, dtype=torch.float32)
    physical = normalized * aux.coeff_scales
    packed, probabilities = recipe_targets(aux, atoms, physical, stochastic=False)
    expected = (-(normalized[..., None] - aux.coeff_bins).square() / .5).softmax(-1)
    torch.testing.assert_close(probabilities, expected)
    assert torch.equal(packed % 8, expected.argmax(-1))
    aux.soft_target_physical = True
    with pytest.raises(ValueError, match='normalized-distance'):
        recipe_targets(aux, atoms, physical)


def test_decay_is_continuous_monotone_after_warmup_and_stays_at_floor():
    assert early_decay_lr(1) == pytest.approx(2e-4)
    assert early_decay_lr(30) == pytest.approx(5e-5)
    assert early_decay_lr(200) == pytest.approx(2e-6)
    assert early_decay_lr(500) == pytest.approx(2e-6)
    assert early_decay_lr(10) < 1.31e-4
    rates = [early_decay_lr(float(e)) for e in torch.linspace(1, 200, 1000)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    for point in (1, 30):
        assert early_decay_lr(point - 1e-6) == pytest.approx(early_decay_lr(point + 1e-6), rel=3e-6)


def test_resume_preserves_epoch_schedule_across_partial_batches():
    stream = EpochStream(19, 12)
    for _ in range(5):
        stream.next(8)
    resumed = EpochStream(19, 999)
    resumed.load_state_dict(stream.state_dict())
    for _ in range(10):
        a, pa, ea = stream.next(8)
        b, pb, eb = resumed.next(8)
        assert torch.equal(a, b) and ea == eb
        assert early_decay_lr(pa) == early_decay_lr(pb)


def test_capacity_moves_layers_to_depth_and_retains_full_context():
    with torch.device('meta'):
        reference, balanced = make_prior('reference'), make_prior('balanced')
    assert sum(p.numel() for p in balanced.parameters()) == 218802176
    assert sum(p.numel() for p in reference.parameters()) == 404738048
    assert len(balanced.head_transformer.blocks) == 6
    assert len(balanced.body_transformer.blocks) == 20
    assert balanced.pair_autoregressive
    assert list(balanced.block_size) == [8, 8, 4]
