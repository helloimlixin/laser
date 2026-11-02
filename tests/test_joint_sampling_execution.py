import pytest
import torch
from omegaconf import OmegaConf

from src import rq_upstream
from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.laser_joint_recipe import JointLaserRecipe


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA autocast cache required')
def test_sampling_cast_cache_preserves_codes_rng_and_updated_weights():
    device = torch.device('cuda:0')
    profile = JointLaserRecipe.__new__(JointLaserRecipe)
    profile.shape = (2, 2, 4)
    profile.recipe = OmegaConf.create(dict(tokenizer=dict(vocab_size=5),
        sampling=dict(temperature=1., top_k=5, top_p=.95)))
    quantizer = JointAtomCoefficientRQ(torch.randn(7, 3),
        torch.tensor([[0, 0, 1, 1, 2]]*4),
        torch.tensor([[0., -1., 1., -2., 2.]]*4)).to(device)
    config = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4],
        embed_dim=12, input_embed_dim=7, shared_tok_emb=True, shared_cls_emb=True,
        input_emb_vqvae=True, head_emb_vqvae=True, cumsum_depth_ctx=True,
        vocab_size=5, vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.))))
    model = rq_upstream.RQTransformer(rq_upstream.RQTransformerConfig.create(config)).to(device).eval()
    labels = torch.tensor([0, 999], device=device)
    original_autocast = torch.is_autocast_enabled('cuda')
    with torch.no_grad():
        for _ in range(2):
            profile.execution_options = {}
            torch.cuda.manual_seed(456)
            expected = profile.sample(model, quantizer, len(labels), labels)
            expected_rng = torch.cuda.get_rng_state(device).clone()
            profile.execution_options = dict(sampling_cast_cache=True)
            torch.cuda.manual_seed(456)
            actual = profile.sample(model, quantizer, len(labels), labels)
            assert torch.equal(actual, expected)
            assert torch.equal(torch.cuda.get_rng_state(device), expected_rng)
            assert torch.is_autocast_enabled('cuda') == original_autocast
            # A later sampling call must use parameters updated since the
            # previous cast-cache scope, as happens after a training epoch.
            model.classifier.linear.weight.add_(torch.randn_like(model.classifier.linear.weight)*.2)
