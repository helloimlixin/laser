import torch
import pytest
from omegaconf import OmegaConf

from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.laser_joint_refinement import append_refinement_depth,refit_last_depth


def test_refinement_preserves_vocabulary_prefix_dictionary_and_reduces_error():
    torch.manual_seed(26101751)
    dictionary=torch.nn.functional.normalize(torch.randn(7,3),dim=0)
    atoms=torch.cat((torch.zeros(1,dtype=torch.long),torch.arange(3).repeat_interleave(4))).expand(4,-1)
    coefficients=torch.cat((torch.zeros(4,1),torch.tensor([-2.,-.7,.7,2.]).repeat(4,3)),1)
    original=JointAtomCoefficientRQ(dictionary,atoms,coefficients)
    x=torch.randn(200,7)
    prior=original.quantize(x)
    extended=append_refinement_depth(original)
    result=extended.quantize(x)
    assert extended.vocab_size==original.vocab_size and extended.depth==5
    assert torch.equal(result['codes'][:,:4],prior['codes'])
    report=refit_last_depth(extended,x,result['codes'])
    assert report['fixed_assignment_mse_after']<=report['fixed_assignment_mse_before']
    torch.testing.assert_close(extended.dictionary,original.dictionary,rtol=0,atol=0)
    torch.testing.assert_close(extended.coefficients[:4],original.coefficients,rtol=0,atol=0)
    after=extended.quantize(x)
    assert torch.equal(after['codes'][:,:4],prior['codes'])
    assert (extended.coefficients[:,-1]>0).all()


@pytest.mark.parametrize('depth',[5,6])
def test_native_transformer_supports_refinement_depths_with_exact_causal_cache(depth):
    from src import rq_upstream
    from src.training.laser_joint_recipe import JointLaserRecipe
    from rqvae.utils.config import augment_arch_defaults
    torch.manual_seed(26101752)
    dictionary=torch.nn.functional.normalize(torch.randn(7,3),dim=0)
    atoms=torch.cat((torch.zeros(1,dtype=torch.long),torch.arange(3).repeat_interleave(4))).expand(4,-1)
    coefficients=torch.cat((torch.zeros(4,1),torch.tensor([-2.,-.7,.7,2.]).repeat(4,3)),1)
    q=JointAtomCoefficientRQ(dictionary,atoms,coefficients)
    while q.depth<depth:q=append_refinement_depth(q)
    profile=JointLaserRecipe.__new__(JointLaserRecipe)
    profile.recipe=OmegaConf.create(dict(tokenizer=dict(vocab_size=q.vocab_size,depth=depth)))
    arch=OmegaConf.create(dict(type='rq-transformer',block_size=[2,2,4],embed_dim=12,
        input_embed_dim=7,shared_tok_emb=True,shared_cls_emb=True,input_emb_vqvae=True,
        head_emb_vqvae=True,cumsum_depth_ctx=True,vocab_size=3,vocab_size_cond=1000,block_size_cond=1,
        body=dict(n_layer=1,block=dict(n_head=3,resid_pdrop=0.)),
        head=dict(n_layer=1,block=dict(n_head=3,resid_pdrop=0.))))
    model=profile.model(augment_arch_defaults(arch)).eval()
    assert type(model) is rq_upstream.RQTransformer and tuple(model.block_size)==(2,2,depth)
    assert arch.block_size[-1]==4
    codes=torch.randint(q.vocab_size,(2,2,2,depth));labels=torch.tensor([106,269])
    with torch.no_grad():
        expected=model(codes,q,labels);prefix=torch.zeros_like(codes);model.init_cache()
        for h in range(2):
            for w in range(2):
                for d in range(depth):
                    actual=model.cached_forward(prefix[:,:h+1],q,labels,sample_loc=(h,w,d))
                    torch.testing.assert_close(actual,expected[:,h,w,d],rtol=1.e-5,atol=1.e-6)
                    prefix[:,h,w,d]=codes[:,h,w,d]
        torch.manual_seed(453)
        cached=model.sample(torch.zeros_like(codes),q,labels,top_k=7,top_p=.95)
        torch.manual_seed(453)
        full=model.sample(torch.zeros_like(codes),q,labels,top_k=7,top_p=.95,cached=False)
        assert torch.equal(cached,full)
