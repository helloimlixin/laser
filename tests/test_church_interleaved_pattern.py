import itertools
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from src.church_interleaved_pattern import CoefficientPrefixTree, InterleavedPatternRQTransformer, interleaved_objective
from src.models.rqtransformer.configs import RQTransformerConfig
from src.support_pattern_integer_codec import pack_support_pattern, decode_support_pattern_integers


def setup():
    torch.manual_seed(10101)
    config=RQTransformerConfig.create(OmegaConf.create({
        'type':'rq-transformer','block_size':[2,2,8],'embed_dim':12,'input_embed_dim':6,
        'shared_tok_emb':True,'shared_cls_emb':True,'input_emb_vqvae':True,'head_emb_vqvae':True,
        'cumsum_depth_ctx':False,'vocab_size':8,'vocab_size_cond':1,'block_size_cond':1,
        'body':{'n_layer':2,'block':{'n_head':3,'resid_pdrop':0.}},
        'head':{'n_layer':2,'block':{'n_head':3,'resid_pdrop':0.}}}))
    table=torch.tensor(list(itertools.product([2,5],repeat=4)))
    model=InterleavedPatternRQTransformer(config,8,table,8).eval()
    dictionary=torch.nn.functional.normalize(torch.randn(6,8),dim=0)
    bins=torch.linspace(-3,3,8);scales=torch.tensor([3.,2.,1.,.5])
    patterns=bins[table]*scales
    aux=SimpleNamespace(dictionary=dictionary,coeff_bins=bins,coeff_scales=scales,coefficient_patterns=patterns,
        coefficient_pattern_latents=lambda a,p:(dictionary.t()[a]*patterns[p][...,None]).sum(-2))
    atoms=torch.stack([torch.randperm(8)[:4] for _ in range(8)]).reshape(2,2,2,4)
    patterns=torch.randint(16,(2,2,2))
    return model,aux,atoms,patterns


def test_prefix_tree_probability_normalizes_over_exact_existing_patterns():
    model,_,_,_=setup();tree=model.tree
    table=tree.pattern_coefficient_ids
    nodes,recovered=tree.nodes(table)
    assert torch.equal(recovered,torch.arange(len(table)))
    torch.manual_seed(81)
    probability=torch.ones(len(table))
    for depth in range(4):
        logits=torch.randn(len(tree.transitions(depth)),8)[nodes[depth]]
        probabilities=tree.mask(logits,nodes[depth],depth).softmax(-1)
        probability*=probabilities.gather(-1,table[:,depth,None]).squeeze(-1)
    torch.testing.assert_close(probability.sum(),torch.tensor(1.))
    wrong=table.clone();wrong[0,2]=7
    with pytest.raises(ValueError): tree.nodes(wrong)
    with pytest.raises(ValueError): CoefficientPrefixTree(torch.cat((table,table[:1])),8)


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
    contributions=aux.dictionary.t()[atoms]*aux.coefficient_patterns[patterns][...,None]
    torch.testing.assert_close(vectors[...,1::2,:],contributions.cumsum(-2))
    torch.testing.assert_close(model.embed_with_model_aux(packed,aux).sum(-2),contributions.sum(-2))


def test_all_eight_cached_fields_match_teacher_forcing_over_full_spatial_grid():
    model,aux,atoms,patterns=setup();packed=model.pack(atoms,patterns)
    coefficients=model.tree.pattern_coefficient_ids[patterns]
    nodes,_=model.tree.nodes(coefficients)
    with torch.no_grad():
        teacher=model(packed,aux);model.init_cache()
        for h in range(2):
            for w in range(2):
                for event in range(8):
                    depth=event//2
                    hidden=model.cached_hidden(packed,aux,(h,w,event),amp=False)
                    if event%2:
                        logits=model.tree.mask(model.coefficient_classifiers[depth](hidden),nodes[depth][:,h,w],depth)
                        target=teacher['coefficient_logits'][:,h,w,depth]
                    else:
                        logits=model.classifier(hidden)
                        if depth: logits.scatter_(1,atoms[:,h,w,:depth],-float('inf'))
                        target=teacher['atom_logits'][:,h,w,depth]
                    torch.testing.assert_close(logits,target,atol=2e-6,rtol=2e-6)
        model.init_cache()


def test_joint_likelihood_backpropagates_to_atoms_and_each_coefficient_head():
    model,aux,atoms,patterns=setup()
    loss,metrics=interleaved_objective(model,aux,atoms,aux.coefficient_patterns[patterns])
    assert abs(float(loss)*5-metrics['joint_nll'])<3e-6
    assert abs(metrics['joint_nll']-4*metrics['atom_nll']-metrics['pattern_nll'])<3e-6
    loss.backward()
    assert model.classifier.linear.weight.grad.abs().sum()>0
    for head in model.coefficient_classifiers:
        assert head[-1].weight.grad.abs().sum()>0


def test_sampling_returns_existing_patterns_and_exact_complete_integer_codes():
    model,aux,_,_=setup()
    atoms,patterns=model.sample_compound(3,aux,atom_top_k=8,coeff_top_p=.5,amp=False)
    integers=pack_support_pattern(atoms,patterns,num_patterns=16,num_atoms=8)
    a,c=decode_support_pattern_integers(integers,model.tree.pattern_coefficient_ids,num_atoms=8)
    assert torch.equal(a.reshape_as(atoms),atoms)
    assert torch.equal(c.reshape_as(atoms),model.tree.pattern_coefficient_ids[patterns])
    assert all(len(set(v))==4 for v in atoms.reshape(-1,4).tolist())
