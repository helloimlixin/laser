import pytest
import torch

from src.models.joint_multiscale_var import JointMultiScaleQuantizer
from src.models.residual_vector_var import SharedResidualVectorQuantizer, SharedAnchoredResidualVectorQuantizer, ResidualVectorVAR
from src.models.trainable_joint_var_tokenizer import MultiScaleGeometry


def make_quantizer():
    torch.manual_seed(91)
    geometry = MultiScaleGeometry(channels=4,patch_nums=(1,2))
    directions = torch.nn.functional.normalize(torch.randn(4,6),dim=0)
    coefficients = torch.tensor([-1.,1.]).expand(2,6,2).clone()
    coefficients[1] *= 2
    anchor = JointMultiScaleQuantizer(geometry,directions,coefficients)
    vectors = torch.cat([torch.zeros(1,4),directions.T,-directions.T,2*directions.T,-2*directions.T])
    return SharedAnchoredResidualVectorQuantizer(anchor,vectors,temperature=.04,chunk_size=8)


def test_one_physical_book_is_identical_across_all_scales_and_depths():
    q = make_quantizer()
    assert q.vectors.ndim == 2 and q.vectors.shape == (25,4)
    ids = torch.arange(25)
    for scale in range(2):
        for depth in range(2):
            torch.testing.assert_close(q.embed(ids,scale,depth),q.vectors,rtol=0,atol=0)
    with pytest.raises(ValueError):
        SharedResidualVectorQuantizer(q.source,q.vectors[None,None])


def test_shared_codec_hard_and_stochastic_integer_roundtrip():
    q = make_quantizer()
    latent = torch.randn(3,4,2,2)
    for stochastic in [False,True]:
        encoded = q.decompose(latent,stochastic=stochastic)
        decoded,inputs = q.from_codes(encoded['codes'])
        torch.testing.assert_close(decoded,encoded['latent'],rtol=0,atol=0)
        torch.testing.assert_close(inputs,encoded['inputs'],rtol=0,atol=0)
        decoder_only=SharedResidualVectorQuantizer(q.source,q.vectors)
        shared_decoded,shared_inputs=decoder_only.from_codes(encoded['codes'])
        torch.testing.assert_close(shared_decoded,encoded['latent'],rtol=0,atol=0)
        torch.testing.assert_close(shared_inputs,encoded['inputs'],rtol=0,atol=0)
        assert encoded['codes'].shape == (3,5,2)
        assert encoded['codes'].min() >= 0 and encoded['codes'].max() < 25


def test_shared_integer_decoding_is_exact_inside_autocast():
    class ScaleFilter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer=torch.nn.Conv2d(4,4,1)
        def __getitem__(self,scale):
            return self.layer
    q=make_quantizer()
    q.source.quant_resi=ScaleFilter().requires_grad_(False)
    encoded=q.decompose(torch.randn(2,4,2,2))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        decoded,inputs=q.from_codes(encoded['codes'])
    assert decoded.dtype==torch.float32 and inputs.dtype==torch.float32
    # CPU convolution layouts may differ by a float32 ULP. A BF16 leak is much larger.
    torch.testing.assert_close(decoded,encoded['latent'],rtol=1.e-6,atol=1.e-7)
    torch.testing.assert_close(inputs,encoded['inputs'],rtol=1.e-6,atol=1.e-7)


def test_shared_book_preserves_anchor_targets_with_prefix_compensation():
    q = make_quantizer()
    latent = torch.randn(2,4,2,2)
    encoded = q.decompose(latent,stochastic=True)
    signal = torch.nn.functional.interpolate(latent,(1,1),mode='area').flatten(2).transpose(1,2)
    old_ids,_ = q._anchor[0].quantize_scale(signal,0)
    desired = q._anchor[0].embed_joint(old_ids,0).sum(-2)
    first = q.embed(encoded['codes'][:,:1,0],0,0)
    torch.testing.assert_close(first+encoded['residuals'][:,:1,1],desired)


def test_shared_codec_prior_loss_and_sampling():
    q = make_quantizer()
    encoded = q.decompose(torch.randn(2,4,2,2),stochastic=True)
    model = ResidualVectorVAR(q,depth=2,width=32,heads=2,num_classes=3)
    model.cond_drop_rate = 0
    loss = model(torch.tensor([0,1]),encoded['inputs'],encoded['codes'],encoded['residuals'])
    loss.backward()
    assert torch.isfinite(loss) and model.class_emb.weight.grad.norm() > 0
    assert not any(p.grad is not None for p in q.parameters())
    images = model.sample(torch.tensor([0,1]),top_k=5,seed=17)
    assert images.shape == (2,4,2,2) and images.isfinite().all()
