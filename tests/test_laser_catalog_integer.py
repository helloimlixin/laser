from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from src import rq_upstream
from src.models.laser_catalog_integer_tokenizer import FrozenCatalogIntegerLaserTokenizer, validate_catalog
from src.training.laser_catalog_calibration import fit_catalog_bins, ordered_signed_grid_ids
from src.training.packed_pair_loss import sparse_pair_cross_entropy


def small_aux():
    aux = FrozenCatalogIntegerLaserTokenizer.__new__(FrozenCatalogIntegerLaserTokenizer)
    torch.nn.Module.__init__(aux)
    aux.num_atoms, aux.coeff_vocab_size, aux.sparsity_level, aux.levels_per_atom, aux.vocab_size = 3, 8, 4, 4, 12
    aux.register_buffer('dictionary', F.normalize(torch.randn(5, 3), dim=0))
    aux.register_buffer('coeff_bins', torch.linspace(-3, 3, 8))
    aux.register_buffer('coeff_scales', torch.tensor([4., 2., 1., .5]))
    aux.register_buffer('catalog_bin_ids', torch.tensor([[0, 2, 5, 7], [1, 3, 4, 6], [0, 3, 4, 7]]))
    return aux


def test_catalog_ids_keep_exact_source_pair_identity_at_every_depth():
    torch.manual_seed(5)
    aux = small_aux()
    codes = torch.tensor([[0, 0, 0, 0], [6, 6, 6, 6], [11, 11, 11, 11]])
    atoms, bins = aux.unpack_catalog(codes)
    assert torch.equal(atoms, torch.tensor([[0]*4, [1]*4, [2]*4]))
    assert torch.equal(bins, torch.tensor([[0]*4, [4]*4, [7]*4]))
    emb = aux.get_code_emb_with_depth(codes)[0]
    expected = aux.dictionary.T[atoms]*aux.coeff_bins[bins][..., None]*aux.coeff_scales[None, :, None]
    torch.testing.assert_close(emb, expected)
    torch.testing.assert_close(aux.get_code_emb_with_depth(codes[:, :2])[0], emb[:, :2])
    assert torch.equal(aux.pack_nearest(atoms, aux.coeff_bins[bins]), codes)
    for invalid in [torch.tensor([[-1]*4]), torch.tensor([[12]*4])]:
        with pytest.raises(ValueError): aux.unpack_catalog(invalid)


def test_calibration_retains_unique_signed_512_grid_pairs_and_reduces_error():
    torch.manual_seed(6)
    bins = torch.linspace(-3., 3., 512)
    a = torch.randint(3, (1000, 4))
    x = torch.randn(1000, 4)*.2 + torch.tensor([.4, .5, .6])[a]*torch.randn(1000, 4).sign()
    ids = fit_catalog_bins(a, x, torch.tensor([4., 2., 1., .5]), 3, bins, 8)
    validate_catalog(ids, 3, 512)
    assert (ids[:, :4] < 256).all() and (ids[:, 4:] >= 256).all()
    error = (x[..., None]-bins[ids[a]]).square().min(-1).values.mean()
    assert error < .01
    centers = torch.zeros(3, 16)
    projected = ordered_signed_grid_ids(centers, bins)
    validate_catalog(projected, 3, 512)


def test_catalog_ce_is_one_native_joint_softmax_including_all_other_atoms():
    torch.manual_seed(7)
    logits = torch.randn(2, 4, 12, requires_grad=True)
    atoms = torch.randint(3, (2, 4))
    p = torch.randn(2, 4, 4).softmax(-1)
    expected_targets = torch.zeros_like(logits)
    expected_targets.scatter_(-1, atoms[..., None]*4+torch.arange(4), p)
    expected = -(expected_targets*logits.log_softmax(-1)).sum(-1).mean()
    actual = sparse_pair_cross_entropy(logits, atoms, p, row_chunk_size=3)
    torch.testing.assert_close(actual, expected)
    left, = torch.autograd.grad(actual, logits, retain_graph=True)
    right, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(left, right)


def test_catalog_physical_embeddings_match_native_full_cached_categorical_forward():
    torch.manual_seed(8)
    aux = small_aux()
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4], embed_dim=12,
        input_embed_dim=5, shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=12,
        vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.))))
    model = rq_upstream.RQTransformer(rq_upstream.RQTransformerConfig.create(arch)).eval()
    target = torch.randint(12, (2, 2, 2, 4))
    labels = torch.tensor([106, 269])
    with torch.no_grad():
        expected = model(target, aux, labels)
        prefix = torch.zeros_like(target)
        model.init_cache()
        for h in range(2):
            for w in range(2):
                for d in range(4):
                    actual = model.cached_forward(prefix[:, :h+1], aux, labels, sample_loc=(h, w, d))
                    torch.testing.assert_close(actual, expected[:, h, w, d], rtol=1.e-5, atol=1.e-6)
                    prefix[:, h, w, d] = target[:, h, w, d]


def test_catalog_recipe_keeps_native_head_backbone_and_whole_token_sampler():
    from src.training.laser_catalog_integer_recipe import CatalogIntegerLaserRecipe
    from src.training.rq_imagenet_recipes import RQRecipe
    from rqvae.utils.config import augment_arch_defaults
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4], embed_dim=12,
        input_embed_dim=5, shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=7,
        vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.))))
    arch = augment_arch_defaults(arch)
    profile = CatalogIntegerLaserRecipe.__new__(CatalogIntegerLaserRecipe)
    profile.recipe = OmegaConf.create(dict(initialization=dict(reference_vocab_size=7),
        sampling=dict(temperature=.9, top_k=0, top_p=.95)))
    profile.vocabulary, profile.generation_batch, profile.shape = 12, 2, (2, 2, 4)
    torch.manual_seed(125)
    reference = RQRecipe('.').model(OmegaConf.create(OmegaConf.to_container(arch)))
    torch.manual_seed(125)
    model = profile.model(arch).eval()
    assert type(model) is rq_upstream.RQTransformer and type(model.classifier.linear) is torch.nn.Linear
    assert model.classifier.linear.weight.shape == (12, 12)
    for name, tensor in model.state_dict().items():
        if not name.startswith('classifier.'):
            torch.testing.assert_close(tensor, reference.state_dict()[name], rtol=0, atol=0)
    aux = small_aux()
    labels = torch.tensor([106, 269])
    with torch.no_grad():
        torch.manual_seed(126)
        expected = model.sample(torch.zeros(2, 2, 2, 4, dtype=torch.long), aux, labels,
                                temperature=.9, top_k=12, top_p=.95, amp=True)
        torch.manual_seed(126)
        actual = profile.sample(model, aux, 2, labels)
    assert torch.equal(actual, expected)
