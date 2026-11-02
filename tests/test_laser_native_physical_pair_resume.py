import torch
from omegaconf import OmegaConf

from tests.test_physical_pair_scalar_prior import case
from src.models.laser_rq_adapter import LaserPairRQTransformer
from src.rq_upstream import RQTransformerConfig
from rqvae.models.rqtransformer.attentions import AttentionStack


def pair():
    old,aux,tokens,cond=case()
    with torch.no_grad():
        old.input_mlp.bias.normal_(std=.2)
        old.head_mlp.bias.normal_(std=.2)
    native=LaserPairRQTransformer(RQTransformerConfig.create(OmegaConf.create(
        OmegaConf.to_container(old.config,resolve=True))),num_atoms=7).eval()
    assert list(old.state_dict())==list(native.state_dict())
    native.load_state_dict(old.state_dict(),strict=True)
    return old,native,aux,tokens,cond


def test_original_rq_stacks_match_legacy_loss_and_all_parameter_gradients():
    old,native,aux,tokens,cond=pair()
    assert isinstance(native.body_transformer,AttentionStack)
    assert isinstance(native.head_transformer,AttentionStack)
    outputs=[model(tokens,aux,cond) for model in (old,native)]
    for key in outputs[0]:torch.testing.assert_close(outputs[0][key],outputs[1][key],rtol=1.e-5,atol=2.e-6)
    for output in outputs:
        atom=-output['atom_logits'].log_softmax(-1).gather(-1,tokens[...,0::2,None]).mean()
        coefficient=output['coeff_logits'].log_softmax(-1).square().mean()
        (atom+coefficient).backward()
    for a,b in zip(old.parameters(),native.parameters()):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad,b.grad,rtol=3.e-5,atol=3.e-6)


@torch.no_grad()
def test_original_cache_matches_legacy_and_teacher_forcing_at_every_event():
    old,native,aux,tokens,cond=pair()
    expected=native(tokens,aux,cond)
    prefix=torch.zeros_like(tokens);prefix[...,1::2]=9
    old.init_cache();native.init_cache()
    for h in range(2):
        for w in range(2):
            for event in range(8):
                left=old.cached_forward(prefix[:,:h+1],aux,cond,sample_loc=(h,w,event))
                right=native.cached_forward(prefix[:,:h+1],aux,cond,sample_loc=(h,w,event))
                torch.testing.assert_close(left,right,rtol=1.e-5,atol=2.e-6)
                key='atom_logits' if event%2==0 else 'coeff_logits'
                selected=right[:,:7] if event%2==0 else right[:,7:]
                torch.testing.assert_close(selected,expected[key][:,h,w,event//2],rtol=1.e-5,atol=2.e-6)
                prefix[:,h,w,event]=tokens[:,h,w,event]


@torch.no_grad()
def test_original_cache_keeps_joint_candidate_sampler_and_seeded_outputs():
    from src.physical_pair_sampling import PairSamplingPolicy,sample_physical_pairs
    old,native,aux,tokens,cond=pair()
    settings=PairSamplingPolicy(candidate_atoms=4,coefficient_temperatures=(.4,.55,.75,1.))
    torch.manual_seed(73);before=sample_physical_pairs(old,1,aux,cond,policy=settings,amp=False)
    torch.manual_seed(73);after=sample_physical_pairs(native,1,aux,cond,policy=settings,amp=False)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
