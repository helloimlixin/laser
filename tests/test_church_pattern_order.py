from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from src.church_pattern_order import ChurchPatternOrderRQTransformer, order_objective, pattern_order_prior, sample_field
from src.models.rqtransformer.configs import RQTransformerConfig
from src.support_pattern_integer_codec import pack_support_pattern, unpack_support_pattern


def setup(order):
    torch.manual_seed(1901)
    config = RQTransformerConfig.create(OmegaConf.create({
        'type':'rq-transformer','block_size':[2,2,5],'embed_dim':12,'input_embed_dim':6,
        'shared_tok_emb':True,'shared_cls_emb':True,'input_emb_vqvae':True,'head_emb_vqvae':True,
        'cumsum_depth_ctx':False,'vocab_size':8,'vocab_size_cond':1,'block_size_cond':1,
        'body':{'n_layer':2,'block':{'n_head':3,'resid_pdrop':0.}},
        'head':{'n_layer':2,'block':{'n_head':3,'resid_pdrop':0.}}}))
    model = ChurchPatternOrderRQTransformer(config,8,16,order).eval()
    dictionary = torch.nn.functional.normalize(torch.randn(6,8),dim=0)
    patterns = torch.randn(16,4)
    aux = SimpleNamespace(dictionary=dictionary,coefficient_patterns=patterns,coeff_scales=torch.ones(4),
        coefficient_pattern_latents=lambda a,p: (dictionary.t()[a]*patterns[p][...,None]).sum(-2))
    atoms = torch.stack([torch.randperm(8)[:4] for _ in range(8)]).reshape(2,2,2,4)
    ids = torch.randint(16,(2,2,2))
    return model,aux,atoms,ids


@pytest.mark.parametrize('order',['pattern-first','support-first'])
def test_transport_decodes_same_latent_and_same_complete_integer(order):
    model,aux,atoms,ids = setup(order)
    packed = model.pack(atoms,ids)
    a,p = model.unpack(packed)
    assert torch.equal(a,atoms) and torch.equal(p,ids)
    torch.testing.assert_close(model.embed_with_model_aux(packed,aux).sum(-2),aux.coefficient_pattern_latents(atoms,ids))
    integers = pack_support_pattern(a,p,num_patterns=16,num_atoms=8)
    aa,pp = unpack_support_pattern(integers,num_patterns=16,num_atoms=8)
    assert torch.equal(aa.reshape_as(a),a) and torch.equal(pp.reshape_as(p),p)


@pytest.mark.parametrize('order',['pattern-first','support-first'])
def test_current_and_future_fields_do_not_leak_into_their_predictions(order):
    model,aux,atoms,ids = setup(order)
    with torch.no_grad():
        baseline = model(model.pack(atoms,ids),aux)
        for site in range(4):
            h,w = divmod(site,2)
            changed_ids = ids.clone()
            changed_ids[:,h,w] = (ids[:,h,w]+3)%16
            changed = model(model.pack(atoms,changed_ids),aux)
            torch.testing.assert_close(changed['pattern_logits'][:,:h+1].reshape(2,-1,16)[:,:site+1],
                baseline['pattern_logits'][:,:h+1].reshape(2,-1,16)[:,:site+1],atol=0,rtol=0)
            if order == 'support-first':
                assert torch.equal(changed['atom_logits'][:,h,w],baseline['atom_logits'][:,h,w])
            else:
                assert not torch.equal(changed['atom_logits'][:,h,w,0],baseline['atom_logits'][:,h,w,0])
            for depth in range(4):
                other = atoms.clone()
                # Swapping the remaining suffix preserves distinct support.
                replacement = next(a for a in range(8) if a not in atoms[0,h,w].tolist())
                other[0,h,w,depth] = replacement
                changed = model(model.pack(other,ids),aux)
                assert torch.equal(changed['atom_logits'][:,h,w,:depth+1],baseline['atom_logits'][:,h,w,:depth+1])
                if order == 'pattern-first':
                    assert torch.equal(changed['pattern_logits'][:,h,w],baseline['pattern_logits'][:,h,w])


@pytest.mark.parametrize('order',['pattern-first','support-first'])
def test_full_grid_cached_logits_match_teacher_forcing(order):
    model,aux,atoms,ids = setup(order)
    packed = model.pack(atoms,ids)
    with torch.no_grad():
        teacher = model(packed,aux)
        model.init_cache()
        for h in range(2):
            for w in range(2):
                for event in range(5):
                    hidden = model.cached_hidden(packed,aux,(h,w,event),amp=False)
                    if event == (0 if order == 'pattern-first' else 4):
                        logits = model.pattern_classifier(hidden)
                        target = teacher['pattern_logits'][:,h,w]
                    else:
                        depth = event-1 if order == 'pattern-first' else event
                        logits = model.classifier(hidden)
                        if depth: logits.scatter_(1,atoms[:,h,w,:depth],-float('inf'))
                        target = teacher['atom_logits'][:,h,w,depth]
                    torch.testing.assert_close(logits,target,atol=2e-6,rtol=2e-6)
        model.init_cache()


def test_pattern_first_context_uses_signed_weighted_prefixes():
    model,aux,atoms,ids = setup('pattern-first')
    vectors = model.embed_depth_with_model_aux(model.pack(atoms,ids),aux)
    expected = aux.dictionary.t()[atoms]*aux.coefficient_patterns[ids][...,None]
    torch.testing.assert_close(vectors[...,1:,:]-vectors[...,:-1,:],expected)


@pytest.mark.parametrize('order',['pattern-first','support-first'])
def test_objective_is_joint_likelihood_and_trains_both_heads(order):
    model,aux,atoms,ids = setup(order)
    physical = aux.coefficient_patterns[ids]
    loss,metrics = order_objective(model,aux,atoms,physical)
    assert abs(float(loss)*5-metrics['joint_nll'])<3e-6
    assert abs(metrics['joint_nll']-(4*metrics['atom_nll']+metrics['pattern_nll']))<3e-6
    loss.backward()
    assert model.pattern_classifier[-1].weight.grad.abs().sum()>0
    assert model.classifier.linear.weight.grad.abs().sum()>0
    if order == 'pattern-first':
        assert model.pattern_embedding[0].weight.grad.abs().sum()>0


@pytest.mark.parametrize('order',['pattern-first','support-first'])
def test_sampling_keeps_distinct_atoms_and_valid_patterns(order):
    model,aux,_,_ = setup(order)
    atoms,ids = model.sample_compound(3,aux,atom_top_k=8,coeff_top_p=.5,amp=False)
    assert atoms.shape == (3,2,2,4) and ids.shape == (3,2,2)
    assert ids.min()>=0 and ids.max()<16
    assert all(len(set(row))==4 for row in atoms.reshape(-1,4).tolist())


def test_two_arms_have_identical_initial_parameters():
    first,_,_,_ = setup('pattern-first')
    last,_,_,_ = setup('support-first')
    for key,value in first.state_dict().items():
        assert torch.equal(value,last.state_dict()[key])
    with torch.device('meta'):
        first = pattern_order_prior(2048,ordering='pattern-first')
        last = pattern_order_prior(2048,ordering='support-first')
    assert sum(p.numel() for p in first.parameters()) == sum(p.numel() for p in last.parameters())
    assert first.block_size == (8,8,5)
    assert len(first.body_transformer.blocks)==20 and len(first.head_transformer.blocks)==6


def test_nucleus_sampler_excludes_low_probability_tail():
    torch.manual_seed(91)
    values = sample_field(torch.tensor([[3.,2.,-8.]]).repeat(100,1),top_p=.5)
    assert torch.equal(values,torch.zeros_like(values))
