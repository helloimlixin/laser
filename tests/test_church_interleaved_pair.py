import itertools
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from src.church_interleaved_pair import InterleavedPairRQTransformer, interleaved_pair_objective, nearest_coefficient_ids, pack_complete_sites
from src.models.rqtransformer.configs import RQTransformerConfig
from src.complete_sparse_codec import pack_exact, unpack_exact


def setup():
    torch.manual_seed(10101)
    config=RQTransformerConfig.create(OmegaConf.create({
        'type':'rq-transformer','block_size':[2,2,8],'embed_dim':12,'input_embed_dim':6,
        'shared_tok_emb':True,'shared_cls_emb':True,'input_emb_vqvae':True,'head_emb_vqvae':True,
        'cumsum_depth_ctx':False,'vocab_size':8,'vocab_size_cond':1,'block_size_cond':1,
        'body':{'n_layer':2,'block':{'n_head':3,'resid_pdrop':0.}},
        'head':{'n_layer':2,'block':{'n_head':3,'resid_pdrop':0.}}}))
    table=torch.tensor(list(itertools.product([2,5],repeat=4)))
    model=InterleavedPairRQTransformer(config,8,8).eval()
    dictionary=torch.nn.functional.normalize(torch.randn(6,8),dim=0)
    bins=torch.linspace(-3,3,8);scales=torch.tensor([3.,2.,1.,.5])
    patterns=bins[table]*scales
    aux=SimpleNamespace(dictionary=dictionary,coeff_bins=bins,coeff_scales=scales,coefficient_patterns=patterns,
        coefficient_pattern_latents=lambda a,p:(dictionary.t()[a]*patterns[p][...,None]).sum(-2))
    atoms=torch.stack([torch.randperm(8)[:4] for _ in range(8)]).reshape(2,2,2,4)
    patterns=torch.randint(8,(2,2,2,4))
    return model,aux,atoms,patterns


def test_all_coefficients_remain_free_to_condition_on_their_atoms():
    model,aux,atoms,ids=setup()
    output=model(model.pack(atoms,ids),aux)
    assert torch.isfinite(output['coefficient_logits']).all()
    assert output['coefficient_logits'].shape==(2,2,2,4,8)
    physical=aux.coeff_bins[ids]*aux.coeff_scales
    assert torch.equal(nearest_coefficient_ids(aux,physical),ids)
    value=pack_exact([16383]*4,[2047]*4)
    assert value.bit_length()==100
    assert unpack_exact(value)==([16383]*4,[2047]*4)


def test_every_future_pair_uses_both_atom_and_coefficient_without_target_leakage():
    model,aux,atoms,patterns=setup()
    packed=model.pack(atoms,patterns)
    with torch.no_grad():
        baseline=model(packed,aux)
        for depth in range(4):
            # Current coefficient is changed along with its valid pattern leaf.
            changed=packed.clone()
            changed[:,0,0,2*depth+1]=7-changed[:,0,0,2*depth+1]
            output=model(changed,aux)
            assert torch.equal(output['atom_logits'][:,0,0,:depth+1],baseline['atom_logits'][:,0,0,:depth+1])
            assert torch.equal(output['coefficient_logits'][:,0,0,:depth+1],baseline['coefficient_logits'][:,0,0,:depth+1])
            if depth<3:
                assert not torch.equal(output['atom_logits'][:,0,0,depth+1],baseline['atom_logits'][:,0,0,depth+1])
            assert not torch.equal(output['atom_logits'][:,0,1,0],baseline['atom_logits'][:,0,1,0])
            changed=packed.clone()
            other=next(a for a in range(8) if a not in atoms[0,0,0].tolist())
            changed[0,0,0,2*depth]=other
            output=model(changed,aux)
            assert torch.equal(output['atom_logits'][:,0,0,:depth+1],baseline['atom_logits'][:,0,0,:depth+1])
            assert not torch.equal(output['coefficient_logits'][:,0,0,depth],baseline['coefficient_logits'][:,0,0,depth])
        # Future spatial sites cannot change any earlier site's logits.
        changed=packed.clone();changed[:,1,1,7]=7-changed[:,1,1,7]
        output=model(changed,aux)
        for key in baseline:
            assert torch.equal(output[key].flatten(1,2)[:,:3],baseline[key].flatten(1,2)[:,:3])


def test_completed_pair_context_is_its_signed_physical_reconstruction():
    model,aux,atoms,patterns=setup();packed=model.pack(atoms,patterns)
    vectors=model.embed_depth_with_model_aux(packed,aux)
    contributions=aux.dictionary.t()[atoms]*(aux.coeff_bins[patterns]*aux.coeff_scales)[...,None]
    torch.testing.assert_close(vectors[...,1::2,:],contributions.cumsum(-2))
    torch.testing.assert_close(model.embed_with_model_aux(packed,aux).sum(-2),contributions.sum(-2))


def test_all_eight_cached_fields_match_teacher_forcing_over_full_spatial_grid():
    model,aux,atoms,patterns=setup();packed=model.pack(atoms,patterns)
    with torch.no_grad():
        teacher=model(packed,aux);model.init_cache()
        for h in range(2):
            for w in range(2):
                for event in range(8):
                    depth=event//2
                    hidden=model.cached_hidden(packed,aux,(h,w,event),amp=False)
                    if event%2:
                        logits=model.coefficient_classifiers[depth](hidden)
                        target=teacher['coefficient_logits'][:,h,w,depth]
                    else:
                        logits=model.classifier(hidden)
                        if depth: logits.scatter_(1,atoms[:,h,w,:depth],-float('inf'))
                        target=teacher['atom_logits'][:,h,w,depth]
                    torch.testing.assert_close(logits,target,atol=2e-6,rtol=2e-6)
        model.init_cache()


def test_joint_likelihood_backpropagates_to_atoms_and_each_coefficient_head():
    model,aux,atoms,patterns=setup()
    loss,metrics=interleaved_pair_objective(model,aux,atoms,aux.coeff_bins[patterns]*aux.coeff_scales)
    assert abs(float(loss)*8-metrics['joint_nll'])<1e-5
    assert abs(metrics['joint_nll']-4*metrics['atom_nll']-4*metrics['coefficient_nll'])<1e-5
    loss.backward()
    assert model.classifier.linear.weight.grad.abs().sum()>0
    for head in model.coefficient_classifiers:
        assert head[-1].weight.grad.abs().sum()>0


def test_sampling_returns_exact_complete_codes_with_four_independent_coefficients():
    model,aux,_,_=setup()
    atoms,ids=model.sample_compound(3,aux,atom_top_k=8,coeff_top_p=.5,amp=False)
    integers=pack_complete_sites(atoms,ids)
    decoded=[unpack_exact(value) for value in integers]
    a=torch.tensor([value[0] for value in decoded]).reshape_as(atoms)
    c=torch.tensor([value[1] for value in decoded]).reshape_as(ids)
    assert torch.equal(a,atoms) and torch.equal(c,ids)
    assert all(len(set(v))==4 for v in atoms.reshape(-1,4).tolist())
