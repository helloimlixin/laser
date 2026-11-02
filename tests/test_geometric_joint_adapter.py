import copy
import torch
from src.models.geometric_joint_adapter import GeometricJointLinear


def test_identity_gradient_pooling_signs_and_unchanged_null_token():
    torch.manual_seed(442)
    base=torch.nn.Linear(5,7).double()
    reference=copy.deepcopy(base)
    dictionary=torch.nn.functional.normalize(torch.randn(3,2),dim=0)
    atoms=torch.tensor([0,0,0,0,1,1,1]);signs=torch.tensor([0.,-1.,1.,1.,-1.,-1.,1.])
    rng=torch.get_rng_state().clone()
    head=GeometricJointLinear(base,dictionary,atoms,signs).double()
    assert torch.equal(rng,torch.get_rng_state())
    x=torch.randn(2,4,5,dtype=torch.float64,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    actual,expected=head(x),reference(y)
    assert torch.equal(actual,expected)
    upstream=torch.randn_like(actual)
    (actual*upstream).sum().backward();(expected*upstream).sum().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
    torch.testing.assert_close(head.weight.grad,reference.weight.grad,rtol=0,atol=0)
    token_vectors=head.geometry_dictionary.T[atoms]*signs[:,None]
    torch.testing.assert_close(head.geometry_weight.grad,(upstream@token_vectors).flatten(0,-2).T@x.detach().flatten(0,-2))
    with torch.no_grad():head.geometry_weight.normal_(std=.1)
    delta=head(x.detach())-reference(x.detach())
    torch.testing.assert_close(delta[...,0],torch.zeros_like(delta[...,0]))
    torch.testing.assert_close(delta[...,2],delta[...,3])
    torch.testing.assert_close(delta[...,1],-delta[...,2])
    torch.testing.assert_close(delta[...,4],delta[...,5])
    assert head.weight is base.weight and head.bias is base.bias


def test_full_cached_predictions_and_future_causality_with_learned_geometry():
    from omegaconf import OmegaConf
    from src import rq_upstream
    from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
    from src.models.geometric_joint_adapter import add_geometric_joint_classifier
    torch.manual_seed(443)
    dictionary=torch.nn.functional.normalize(torch.randn(7,3),dim=0)
    atoms=torch.tensor([[0,0,0,1,1,1,1,2,2]]*4)
    coefficients=torch.tensor([[0.,-1.,1.,-2.,-.5,.5,2.,-1.,1.]]*4)*torch.tensor([1.,.5,.3,.2])[:,None]
    q=JointAtomCoefficientRQ(dictionary,atoms,coefficients)
    arch=OmegaConf.create(dict(type='rq-transformer',block_size=[2,2,4],embed_dim=12,
        input_embed_dim=7,shared_tok_emb=True,shared_cls_emb=True,input_emb_vqvae=True,
        head_emb_vqvae=True,cumsum_depth_ctx=True,vocab_size=9,vocab_size_cond=1000,
        block_size_cond=1,body=dict(n_layer=1,block=dict(n_head=3,resid_pdrop=0.)),
        head=dict(n_layer=1,block=dict(n_head=3,resid_pdrop=0.))))
    model=rq_upstream.RQTransformer(rq_upstream.RQTransformerConfig.create(arch)).eval()
    add_geometric_joint_classifier(model,q)
    torch.nn.init.normal_(model.classifier.linear.geometry_weight,std=.1)
    tokens=torch.randint(9,(2,2,2,4));cond=torch.tensor([0,999])
    with torch.no_grad():
        full=model(tokens,q,cond)
        changed=tokens.clone();changed[:,-1,-1,-1]=(changed[:,-1,-1,-1]+1)%9
        torch.testing.assert_close(model(changed,q,cond),full,rtol=0,atol=0)
        prefix=torch.zeros_like(tokens);model.init_cache()
        for h in range(2):
            for w in range(2):
                for d in range(4):
                    actual=model.cached_forward(prefix[:,:h+1],q,cond,sample_loc=(h,w,d))
                    torch.testing.assert_close(actual,full[:,h,w,d],rtol=1e-5,atol=1e-6)
                    prefix[:,h,w,d]=tokens[:,h,w,d]
