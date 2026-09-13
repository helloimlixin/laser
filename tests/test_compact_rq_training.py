import torch
from src.compact_rq_training import TrainingAdaptiveScaledAtomRQ


def test_compact_soft_targets_and_commitment_match_original_rq_with_same_book():
    from rqvae.models.rqvae.quantizations import RQBottleneck
    torch.manual_seed(71)
    dictionary=torch.randn(7,11)
    levels=torch.rand(11,2).add(.3)*torch.tensor([-1.,1.])
    model=TrainingAdaptiveScaledAtomRQ(dictionary,levels)
    original=RQBottleneck([2,2,7],[2,2,4],model.vocab_size,
                          shared_codebook=True,restart_unused_codes=False).eval()
    with torch.no_grad():
        original.codebooks[0].weight[:-1].copy_(model.expanded_codebook())
    x=torch.randn(3,2,2,7)
    expected,commitment,codes=original(x)
    actual,actual_commitment,actual_codes=model(x)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(actual_commitment,commitment)
    assert torch.equal(codes,actual_codes)
    p,c=original.get_soft_codes(x,temp=.125,stochastic=False)
    p2,c2=model.get_soft_codes(x,temp=.125,stochastic=False,chunk_size=5)
    torch.testing.assert_close(p2,p,atol=4e-6,rtol=5e-5)
    assert torch.equal(c,c2)
    torch.manual_seed(87)
    p,c=original.get_soft_codes(x,temp=.125,stochastic=True)
    torch.manual_seed(87)
    p2,c2=model.get_soft_codes(x,temp=.125,stochastic=True,chunk_size=128)
    torch.testing.assert_close(p2,p,atol=4e-6,rtol=5e-5)
    assert torch.equal(c,c2)


def test_compact_levels_and_atoms_both_condition_future_predictions():
    from omegaconf import OmegaConf
    from rqvae.models.rqtransformer.configs import RQTransformerConfig
    from rqvae.models.rqtransformer.transformers import RQTransformer
    torch.manual_seed(719)
    quantizer=TrainingAdaptiveScaledAtomRQ(torch.randn(7,11),
        torch.rand(11,2).add(.3)*torch.tensor([-1.,1.]))
    config=RQTransformerConfig.create(OmegaConf.create(dict(
        vocab_size=quantizer.vocab_size,block_size=[2,2,4],embed_dim=32,input_embed_dim=7,
        input_emb_vqvae=True,head_emb_vqvae=True,cumsum_depth_ctx=True,
        shared_tok_emb=True,shared_cls_emb=True,
        body=dict(n_layer=2,block=dict(n_head=4,resid_pdrop=0.)),
        head=dict(n_layer=2,block=dict(n_head=4,resid_pdrop=0.)))))
    model=RQTransformer(config).eval()
    codes=torch.ones(1,2,2,4,dtype=torch.long)
    with torch.no_grad():
        before=model(codes,model_aux=quantizer).reshape(16,-1)
        for altered in (2,3):
            changed=codes.clone()
            changed[0,0,0,1]=altered
            after=model(changed,model_aux=quantizer).reshape(16,-1)
            torch.testing.assert_close(before[:2],after[:2],rtol=0,atol=0)
            assert (before[2:4]-after[2:4]).abs().max()>1e-5
            assert (before[4:]-after[4:]).abs().max()>1e-5
