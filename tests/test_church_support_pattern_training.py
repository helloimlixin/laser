from types import SimpleNamespace

from omegaconf import OmegaConf
import torch

from src.training.rqtransformer import SupportFirstLaserRQTransformer
from src.church_support_pattern_training import pattern_targets, pattern_objective, support_pattern_prior, ChurchSupportPatternRQTransformer
from src.models.rqtransformer.configs import RQTransformerConfig


def make_aux():
    dictionary = torch.nn.functional.normalize(torch.randn(6, 8), dim=0)
    patterns = torch.randn(16, 4)
    return SimpleNamespace(dictionary=dictionary, coefficient_patterns=patterns, coeff_scales=torch.ones(4),
        coefficient_pattern_latents=lambda a,p: (dictionary.t()[a]*patterns[p][...,None]).sum(-2))


def test_pattern_assignment_matches_brute_force_physical_sparse_latent_distance():
    torch.manual_seed(27)
    aux = make_aux()
    atoms = torch.stack([torch.randperm(8)[:4] for _ in range(20)]).reshape(2,2,5,4)
    physical = torch.randn_like(atoms.float())
    ids = pattern_targets(aux,atoms,physical)
    differences = physical[...,None,:]-aux.coefficient_patterns
    error = (differences[...,None]*aux.dictionary.t()[atoms][...,None,:,:]).sum(-2).square().sum(-1)
    assert torch.equal(ids,error.argmin(-1))


def test_joint_objective_backpropagates_to_atom_and_pattern_heads_with_intended_weights():
    torch.manual_seed(28)
    config = RQTransformerConfig.create(OmegaConf.create({
        'type':'rq-transformer','block_size':[1,1,4],'embed_dim':12,'input_embed_dim':6,
        'shared_tok_emb':True,'shared_cls_emb':True,'input_emb_vqvae':True,'head_emb_vqvae':True,
        'cumsum_depth_ctx':True,'vocab_size':8,'vocab_size_cond':1,'block_size_cond':1,
        'body':{'n_layer':1,'block':{'n_head':3,'resid_pdrop':0.}},
        'head':{'n_layer':1,'block':{'n_head':3,'resid_pdrop':0.}}}))
    model = ChurchSupportPatternRQTransformer(config,8,16)
    aux = make_aux()
    atoms = torch.tensor([[[[1,3,2,5]]],[[[0,2,4,7]]]])
    physical = torch.randn(2,1,1,4)
    ids = pattern_targets(aux, atoms, physical)
    packed = model.pack(atoms, ids)
    torch.testing.assert_close(model.embed_depth_with_model_aux(packed, aux),
                               aux.dictionary.t()[atoms].cumsum(-2), atol=1e-6, rtol=1e-6)
    reference = SupportFirstLaserRQTransformer(config,8,16)
    reference.load_state_dict(model.state_dict())
    model.eval()
    reference.eval()
    for key,value in model(packed, model_aux=aux).items():
        torch.testing.assert_close(value,reference(packed,model_aux=aux)[key],atol=1e-6,rtol=1e-6)
    loss, metrics = pattern_objective(model,aux,atoms,physical)
    assert abs(metrics['loss']-(.6*metrics['atom_nll']+.4*metrics['pattern_nll']))<1e-6
    loss.backward()
    assert model.classifier.linear.weight.grad.abs().sum()>0
    assert model.pattern_classifier[-1].weight.grad.abs().sum()>0


def test_production_prior_uses_one_pattern_head_and_requested_capacity():
    with torch.device('meta'):
        model = support_pattern_prior(2048)
    assert model.pattern_classifier[-1].out_features == 2048
    assert len(model.body_transformer.blocks) == 20
    assert len(model.head_transformer.blocks) == 6
    assert model.config.embed_dim == 768
