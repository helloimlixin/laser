import torch
from src.models.trainable_joint_var_tokenizer import MultiScaleGeometry
from src.models.residual_vector_var import ResidualVectorQuantizer, ResidualVectorVAR


def quantizer():
    torch.manual_seed(4)
    book=torch.randn(2,2,12,4);book[:,:,0]=0
    return ResidualVectorQuantizer(MultiScaleGeometry(channels=4,patch_nums=(1,2)),book,temperature=.1)


def test_integer_roundtrip_and_sampled_prefix():
    q=quantizer();latent=torch.randn(3,4,2,2)
    for stochastic in [False,True]:
        encoded=q.decompose(latent,stochastic=stochastic)
        reconstructed,inputs=q.from_codes(encoded['codes'])
        torch.testing.assert_close(reconstructed,encoded['latent'],rtol=0,atol=0)
        torch.testing.assert_close(inputs,encoded['inputs'],rtol=0,atol=0)
        offset=0
        for s,pn in enumerate(q.v_patch_nums):
            lo,hi=offset,offset+pn*pn
            expected=encoded['residuals'][:,lo:hi,0]-q.embed(encoded['codes'][:,lo:hi,0],s,0)
            torch.testing.assert_close(encoded['residuals'][:,lo:hi,1],expected,rtol=0,atol=0)
            offset=hi


def test_soft_loss_class_gradient_and_causality():
    q=quantizer();model=ResidualVectorVAR(q,depth=2,width=32,heads=2,num_classes=3)
    model.cond_drop_rate=0
    encoded=q.decompose(torch.randn(2,4,2,2),stochastic=True)
    loss=model(torch.tensor([0,1]),encoded['inputs'],encoded['codes'],encoded['residuals'])
    loss.backward()
    assert torch.isfinite(loss) and model.class_emb.weight.grad.norm()>0
    assert not any(p.grad is not None for p in q.parameters())
    features=torch.randn(2,5,32);codes=encoded['codes']
    def prefix(c):
        return torch.cat([q.embed(c[:,lo:hi,0],s,0) for s,(lo,hi) in enumerate(model.begin_ends)],1)
    changed=codes.clone();changed[...,1]=(changed[...,1]+1)%q.vocab_size
    torch.testing.assert_close(prefix(codes),prefix(changed),rtol=0,atol=0)
    sampled=model.sample(torch.tensor([0,1]),top_k=5,seed=1)
    assert sampled.shape==(2,4,2,2) and sampled.isfinite().all()


def test_anchored_catalog_preserves_stage1_and_compensates_sampled_prefix():
    from src.models.joint_multiscale_var import JointMultiScaleQuantizer
    from src.models.residual_vector_var import AnchoredResidualVectorQuantizer
    torch.manual_seed(17)
    geometry=MultiScaleGeometry(channels=4,patch_nums=(1,2))
    directions=torch.nn.functional.normalize(torch.randn(4,6),dim=0)
    levels=torch.tensor([-1.,1.]).expand(2,6,2).clone()
    anchor=JointMultiScaleQuantizer(geometry,directions,levels)
    book=torch.zeros(2,2,16,4)
    values=(directions.T[None,:,None,:]*levels[...,None]).reshape(2,12,4)
    book[:,:,1:13]=values[:,None]
    book[:,:,13:]=values[:,None,:3]*.5
    q=AnchoredResidualVectorQuantizer(anchor,book,temperature=.02)
    latent=torch.randn(3,4,2,2)
    old=anchor.decompose(latent);new=q.decompose(latent)
    assert torch.equal(new['codes'],old['codes']+1)
    torch.testing.assert_close(new['latent'],old['latent'],rtol=0,atol=0)
    torch.testing.assert_close(new['inputs'],old['inputs'],rtol=0,atol=0)
    sampled=q.decompose(latent,stochastic=True);decoded,inputs=q.from_codes(sampled['codes'])
    torch.testing.assert_close(decoded,sampled['latent'],rtol=0,atol=0)
    torch.testing.assert_close(inputs,sampled['inputs'],rtol=0,atol=0)
