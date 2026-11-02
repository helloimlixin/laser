import torch

from src.training.rqtransformer import build_parser, compound_objective, scheduled_geometry_weight
from src.training.scalar_pair_loss import scalar_pair_objective, scalar_pair_metrics
from tests.fixtures.ffhq_reference_objective import compound_objective as reference_objective


def case():
    torch.manual_seed(440)
    atoms=torch.tensor([0,1,2,3]).expand(2,2,2,4)
    al=torch.randn(2,2,2,4,7,requires_grad=True)
    cl=torch.randn(2,2,2,4,9,requires_grad=True)
    bins=torch.linspace(-3,3,9)
    values=torch.randn(2,2,2,4)
    probs=(-(values[...,None]-bins).square()/.5).softmax(-1)
    dictionary=torch.randn(5,7)
    scales=torch.tensor([4.,2.,1.5,1.])
    target=dictionary.T[atoms]*(values*scales)[...,None]
    return al,cl,atoms,probs,bins,dictionary,scales,target


def test_scalar_recipe_loss_and_gradients_match_recovered_ffhq_objective():
    al,cl,atoms,probs,bins,dictionary,scales,target=case()
    expected,original=reference_objective(al,cl,None,atoms,probs,target,
        atom_weight=1.5,geometry_weight=.05,accumulation=4,
        distribution_geometry=True,geometry_dictionary=dictionary,
        geometry_coeff_bins=bins,geometry_coeff_scales=scales,geometry_top_k=4)
    actual,details=scalar_pair_objective(al,cl,atoms,probs,bins,atom_weight=1.5,
        accumulation=4,geometry_weight=.05,target_physical=target,
        geometry_dictionary=dictionary,geometry_coeff_scales=scales,
        geometry_top_k=4,geometry_estimator='ffhq_reference')
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-7)
    for key in ['classification','geometry','geometry_pair_mse','geometry_spatial_mse']:
        torch.testing.assert_close(details[key],original[key],rtol=2e-6,atol=1e-7)
    left=torch.autograd.grad(actual,(al,cl),retain_graph=True)
    right=torch.autograd.grad(expected,(al,cl),retain_graph=True)
    for a,b in zip(left,right):torch.testing.assert_close(a,b,rtol=3e-6,atol=1e-7)
    # The physical auxiliary supplies a real gradient to both predicted fields,
    # including the weakest depths; it is not an unused reconstruction head.
    ga,gc=torch.autograd.grad(details['geometry'],(al,cl))
    for gradient in (ga,gc):
        assert torch.isfinite(gradient).all()
        assert (gradient.flatten(0,2).norm(dim=-1).sum(0)>0).all()
    metrics=scalar_pair_metrics(al,cl,atoms,probs,bins,scales,details,
                               atom_weight=1.5,geometry_weight=.05)
    assert metrics['train/coeff_target_entropy']>0
    assert metrics['train/coeff_kl']>=0
    assert abs(metrics['train/image_loss']-float(actual)*4)<1e-5


def test_reference_warmup_and_explicit_estimator_cli():
    assert scheduled_geometry_weight(.05,2.,2.,3.)==0
    assert scheduled_geometry_weight(.05,3.5,2.,3.)==.025
    assert scheduled_geometry_weight(.05,5.,2.,3.)==.05
    args=build_parser().parse_args(['--checkpoint','stage1.pt','--data','data',
        '--output','out','--geometry-estimator','ffhq_reference'])
    assert args.geometry_estimator=='ffhq_reference'
    # Existing candidate-conditional geometry stays the default.
    default=build_parser().parse_args(['--checkpoint','stage1.pt','--data','data','--output','out'])
    assert default.geometry_estimator=='candidate_conditional'
