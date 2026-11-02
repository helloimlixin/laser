import hashlib

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer, JointAtomCoefficientRQ
from src.training.laser_joint_recipe import JointLaserRecipe


def test_compact_dictionary_preserves_source_cnn_and_requires_verified_lineage(tmp_path, monkeypatch):
    from src.models import laser_tokenizer_adapter

    source_dictionary = torch.nn.functional.normalize(torch.randn(3, 5), dim=0)
    checkpoint = tmp_path/'source.pt'
    torch.save({'state_dict': {'quantizer.dictionary': source_dictionary}}, checkpoint)

    class SourceCNN(nn.Module):
        def __init__(self, path, *, num_atoms, **kwargs):
            super().__init__()
            assert path == checkpoint and num_atoms == 5
            self.encoder = nn.Conv2d(3, 3, 1)
            self.decoder = nn.Conv2d(3, 3, 1)
            self.quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()
            self.register_buffer('dictionary', source_dictionary.clone())

    monkeypatch.setattr(laser_tokenizer_adapter, 'FrozenLaserTokenizer', SourceCNN)
    dictionary = torch.nn.functional.normalize(torch.randn(3, 2), dim=0)
    book = dict(kind='laser_joint_atom_coefficient_rq', format_version=2,
        learned_compact_dictionary=True, encoder_decoder_frozen=True,
        source_stage1_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        dictionary=dictionary, atom_ids=torch.tensor([[0, 0, 1]]*4),
        coefficients=torch.tensor([[0., -1., 1.]]*4))
    path = tmp_path/'book.pt'
    torch.save(book, path)
    torch.manual_seed(901)
    expected = SourceCNN(checkpoint, num_atoms=5)
    torch.manual_seed(901)
    actual = FrozenJointLaserTokenizer(checkpoint, path)
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(tensor, actual.backbone.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(actual.quantizer.dictionary, dictionary, rtol=0, atol=0)
    assert not actual.training and all(not p.requires_grad for p in actual.parameters())
    images = torch.randn(2, 3, 2, 2)
    torch.testing.assert_close(actual.encode(images), expected.encoder(images).permute(0, 2, 3, 1))
    codes = torch.ones(2, 2, 2, 4, dtype=torch.long)
    latent = dictionary[:, 0].view(1, 3, 1, 1).expand(2, 3, 2, 2)*-4
    torch.testing.assert_close(actual.decode_code(codes), expected.decoder(latent))
    book['source_stage1_checkpoint_sha256'] = '0'*64
    torch.save(book, path)
    with pytest.raises(ValueError, match='checkpoint SHA256 mismatch'):
        FrozenJointLaserTokenizer(checkpoint, path)


def test_legacy_dictionary_mismatch_remains_rejected(tmp_path, monkeypatch):
    from src.models import laser_tokenizer_adapter

    class Source(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.register_buffer('dictionary', torch.ones(3, 2))

    monkeypatch.setattr(laser_tokenizer_adapter, 'FrozenLaserTokenizer', Source)
    path = tmp_path/'book.pt'
    torch.save(dict(kind='laser_joint_atom_coefficient_rq', dictionary=torch.randn(3, 2),
        atom_ids=torch.tensor([[0, 0, 1]]*4), coefficients=torch.tensor([[0., -1., 1.]]*4)), path)
    with pytest.raises(AssertionError):
        FrozenJointLaserTokenizer(tmp_path/'unused.pt', path)


def test_native_backbone_is_paired_across_vocabulary_change_and_meta_reload():
    from src import rq_upstream
    profile = JointLaserRecipe.__new__(JointLaserRecipe)
    profile.recipe = OmegaConf.create(dict(tokenizer=dict(vocab_size=13)))
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4], embed_dim=12,
        input_embed_dim=7, shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=3,
        vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.))))
    from rqvae.utils.config import augment_arch_defaults
    arch = augment_arch_defaults(arch)
    torch.manual_seed(1029)
    reference = profile.model(arch)
    profile.recipe.tokenizer.vocab_size = 7
    profile.recipe.initialization = dict(reference_vocab_size=13, paired_fresh_backbone=True)
    torch.manual_seed(1029)
    compact = profile.model(arch).eval()
    assert type(compact) is rq_upstream.RQTransformer and arch.vocab_size == 3
    assert compact.vocab_size == compact.config.vocab_size == compact.classifier.logit_mask.vocab_size == [7]*4
    assert compact.state_dict().keys() == reference.state_dict().keys()
    for name, tensor in compact.state_dict().items():
        if not name.startswith('classifier.linear.'):
            torch.testing.assert_close(tensor, reference.state_dict()[name], rtol=0, atol=0)
    q = JointAtomCoefficientRQ(torch.randn(7, 3), torch.tensor([[0, 0, 0, 1, 1, 2, 2]]*4),
        torch.tensor([[0., -1., 1., -1., 1., -1., 1.]]*4))
    tokens = torch.randint(7, (2, 2, 2, 4))
    labels = torch.tensor([0, 999])
    with torch.no_grad():
        expected = compact(tokens, q, labels)
        assert expected.shape == (2, 2, 2, 4, 7)
        compact.init_cache()
        for h in range(2):
            for w in range(2):
                for d in range(4):
                    actual = compact.cached_forward(tokens[:, :h+1], q, labels, sample_loc=(h, w, d))
                    torch.testing.assert_close(actual, expected[:, h, w, d], rtol=1.e-5, atol=1.e-6)
    with torch.device('meta'):
        reloaded = profile.model(arch)
    reloaded.load_state_dict(compact.state_dict(), strict=True, assign=True)
    assert reloaded.classifier.linear.weight.device.type == 'cpu'
    for name, tensor in reloaded.state_dict().items():
        torch.testing.assert_close(tensor, compact.state_dict()[name], rtol=0, atol=0)
