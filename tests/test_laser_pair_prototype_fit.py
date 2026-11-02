import torch

from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.laser_compact_joint_vocabulary import reconstruct_fixed_codes
from src.training.laser_pair_prototype_fit import untie_pair_directions,sparse_dictionary_solve,refit_pair_prototypes


def case():
    torch.manual_seed(26101746)
    dictionary=torch.nn.functional.normalize(torch.randn(7,3),dim=0)
    atoms=torch.cat((torch.zeros(1,dtype=torch.long),torch.arange(3).repeat_interleave(4))).expand(4,-1)
    values=torch.cat((torch.zeros(4,1),torch.tensor([-2.,-.7,.7,2.]).repeat(4,3)),-1)
    return JointAtomCoefficientRQ(dictionary,atoms,values)


def test_untying_pairs_preserves_all_physical_codewords_and_integer_codes():
    original=case();actual=untie_pair_directions(original)
    assert actual.vocab_size==original.vocab_size and actual.dictionary.shape[1]==12
    for d in range(4):
        torch.testing.assert_close(actual.expanded_codebook(d),original.expanded_codebook(d),rtol=0,atol=0)
    x=torch.randn(33,7)
    a,b=original.quantize(x),actual.quantize(x)
    assert torch.equal(a['codes'],b['codes'])
    torch.testing.assert_close(a['quantized'],b['quantized'])


def test_sparse_solver_matches_dense_normal_equations_including_repeated_ids():
    torch.manual_seed(26101747)
    x=torch.randn(40,5);prior=torch.randn(5,7)
    atoms=torch.randint(7,(40,4));coefficients=torch.randn(40,4)
    design=torch.zeros(40,7).scatter_add_(1,atoms,coefficients)
    diagonal=torch.zeros(7)
    for d in range(4):diagonal.index_add_(0,atoms[:,d],coefficients[:,d].square())
    ridge=.01*diagonal.clamp_min(1.)
    expected=torch.linalg.solve(design.T@design+ridge.diag(),design.T@x+ridge[:,None]*prior.T).T
    actual,report=sparse_dictionary_solve(x,atoms,coefficients,prior,iterations=30,tolerance=1.e-7)
    torch.testing.assert_close(actual,expected,rtol=1.e-4,atol=1.e-5)
    assert report['relative_residual_norm']<1.e-5


def test_free_pair_refit_reduces_final_error_and_keeps_vocabulary_and_signs():
    q=untie_pair_directions(case());codes=torch.randint(q.vocab_size,(180,4))
    truth=q.dictionary+torch.randn_like(q.dictionary)*.1
    signals=sum(truth.T[q.atom_ids[d,codes[:,d]]]*q.coefficients[d,codes[:,d],None] for d in range(4))
    signs=q.coefficients.sign().clone()
    report=refit_pair_prototypes(q,signals,codes,cg_iterations=30)
    assert report['fixed_assignment_mse_after']<report['fixed_assignment_mse_before']
    torch.testing.assert_close(q.dictionary.norm(dim=0),torch.ones(12))
    assert q.vocab_size==13 and torch.equal(q.coefficients.sign(),signs)
    actual=(signals-reconstruct_fixed_codes(q,codes)).square().mean()
    torch.testing.assert_close(actual,torch.tensor(report['fixed_assignment_mse_after']),atol=1.e-6,rtol=1.e-5)
