import itertools
import torch
from src.training.sparse_combination_targets import (
    CombinationBank, propose_combination_bank, combination_targets,
    combination_soft_cross_entropy, propose_swap_bank, propose_expanded_bank,
    _coefficient_kernel, _unique_ordered_supports)


def test_bank_errors_use_full_refit_combination_and_gram_covariance():
    g = torch.Generator().manual_seed(301)
    dictionary = torch.nn.functional.normalize(torch.randn(7, 13, dtype=torch.float64, generator=g), dim=0)
    x = torch.randn(6, 7, dtype=torch.float64, generator=g)
    bank = propose_combination_bank(x, dictionary, depth=4, variants=5, generator=g)
    a = dictionary.T[bank.atoms].transpose(-1, -2)
    fitted = torch.linalg.lstsq(a, x[:, None, :, None]).solution.squeeze(-1)
    torch.testing.assert_close(bank.centers, fitted, atol=1e-10, rtol=1e-10)
    energy = (x[:, None] - (a @ fitted[..., None]).squeeze(-1)).square().sum(-1)
    torch.testing.assert_close(bank.errors, energy, atol=1e-10, rtol=1e-10)
    covariance = bank.covariance_cholesky @ bank.covariance_cholesky.transpose(-1,-2)
    torch.testing.assert_close(covariance, torch.linalg.inv(a.transpose(-1,-2) @ a), atol=1e-10, rtol=1e-10)
    assert (bank.atoms.sort(-1).values.diff(dim=-1) > 0).all()


def test_soft_labels_match_enumerated_complete_pair_trajectory_distribution():
    # Different complete supports share a prefix, with correlated coefficients.
    atoms = torch.tensor([[[0,1],[0,2],[3,2]]])
    centers = torch.tensor([[[.3,-.1],[-.2,.4],[.5,-.4]]],dtype=torch.float64)
    chol = torch.tensor([[[[1.,0.],[.6,.8]],[[.7,0.],[-.3,.9]],[[1.2,0.],[.4,.6]]]],dtype=torch.float64)
    bank = CombinationBank(atoms,centers,chol,torch.tensor([[.2,.5,.1]],dtype=torch.float64),torch.ones(1,3,dtype=torch.bool))
    bins=torch.tensor([[-1.,0.,1.],[-.8,0.,.8]],dtype=torch.float64)
    t=.4; s=.3
    target=combination_targets(bank,bins,support_temperature=s,coefficient_temperature=t,generator=torch.Generator().manual_seed(15))
    prior=(-bank.errors[0]/s).softmax(-1)
    law=[]
    for v,c0,c1 in itertools.product(range(3),range(3),range(3)):
        p0=(-((bins[0]-centers[0,v,0])/chol[0,v,0,0]).square()/t).softmax(-1)
        mean1=centers[0,v,1]+chol[0,v,1,0]*(bins[0,c0]-centers[0,v,0])/chol[0,v,0,0]
        p1=(-((bins[1]-mean1)/chol[0,v,1,1]).square()/t).softmax(-1)
        law.append(([int(atoms[0,v,0]),c0,int(atoms[0,v,1]),c1],float(prior[v]*p0[c0]*p1[c1])))
    prefix=[]
    for d in range(2):
        matching=[(events,p) for events,p in law if events[:2*d]==prefix]
        denom=sum(p for _,p in matching)
        expected=torch.tensor([sum(p for events,p in matching if events[2*d]==a)/denom for a in range(4)],dtype=torch.float64)
        actual=torch.zeros(4,dtype=torch.float64).scatter_add_(0,target.atom_target_ids[0,d],target.atom_target_weights[0,d])
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        prefix.append(int(target.atoms[0,d]))
        matching=[(events,p) for events,p in law if events[:2*d+1]==prefix]
        denom=sum(p for _,p in matching)
        expected=torch.tensor([sum(p for events,p in matching if events[2*d+1]==c)/denom for c in range(3)],dtype=torch.float64)
        torch.testing.assert_close(target.coefficient_probabilities[0,d],expected,atol=1e-12,rtol=1e-12)
        prefix.append(int(target.coefficient_ids[0,d]))


def test_soft_loss_gradients_match_dense_labels_with_duplicate_atom_ids():
    torch.manual_seed(100)
    atom=torch.randn(2,3,7,requires_grad=True)
    coeff=torch.randn(2,3,5,requires_grad=True)
    ids=torch.randint(0,7,(2,3,4));weights=torch.rand(2,3,4).softmax(-1)
    q=torch.rand(2,3,5).softmax(-1)
    loss,*_=combination_soft_cross_entropy(atom,coeff,ids,weights,q)
    dense=torch.zeros_like(atom).scatter_add_(-1,ids,weights)
    ref=(-(dense*atom.log_softmax(-1)).sum(-1)-(q*coeff.log_softmax(-1)).sum(-1)).mean()/2
    torch.testing.assert_close(loss,ref)
    ga=torch.autograd.grad(loss,(atom,coeff),retain_graph=True)
    gb=torch.autograd.grad(ref,(atom,coeff))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b)


def test_duplicate_supports_do_not_gain_prior_mass_and_rng_replays():
    x=torch.tensor([[2.,.7,.1]],dtype=torch.float64)
    bank=propose_combination_bank(x,torch.eye(3,dtype=torch.float64),depth=2,variants=6,proposal_temperature=1e-4)
    assert bank.unique.tolist()==[[True,False,False,False,False,False]]
    bins=torch.linspace(-3,3,21,dtype=torch.float64).repeat(2,1)
    g=torch.Generator().manual_seed(40);rng=g.get_state()
    a=combination_targets(bank,bins,support_temperature=.1,coefficient_temperature=.03,generator=g)
    g.set_state(rng)
    b=combination_targets(bank,bins,support_temperature=.1,coefficient_temperature=.03,generator=g)
    assert torch.equal(a.coefficient_ids,b.coefficient_ids)
    assert torch.equal(a.atom_entropy,torch.zeros_like(a.atom_entropy))


def test_replacement_candidates_match_exhaustive_full_combination_distances():
    g=torch.Generator().manual_seed(98)
    d=torch.nn.functional.normalize(torch.randn(6,11,dtype=torch.float64,generator=g),dim=0)
    x=torch.randn(2,6,dtype=torch.float64,generator=g)
    bank=propose_swap_bank(x,d,depth=4,swaps_per_atom=2)
    again=propose_swap_bank(x,d,depth=4,swaps_per_atom=2)
    assert torch.equal(bank.atoms,again.atoms)
    for row in range(2):
        original=bank.atoms[row,0]
        for position in range(4):
            energies=[]
            for atom in range(11):
                if atom in original:continue
                support=original.clone();support[position]=atom
                a=d[:,support];c=torch.linalg.lstsq(a,x[row,:,None]).solution[:,0]
                energies.append((float((x[row]-a@c).square().sum()),atom))
            expected=[v[1] for v in sorted(energies)[:2]]
            assert bank.atoms[row,1+position*2:3+position*2,position].tolist()==expected
    a=d.T[bank.atoms].transpose(-1,-2)
    c=torch.linalg.lstsq(a,x[:,None,:,None]).solution.squeeze(-1)
    torch.testing.assert_close(bank.centers,c,rtol=1e-10,atol=1e-10)


def test_expanded_search_scores_simultaneous_changes_and_retains_best_combinations():
    g=torch.Generator().manual_seed(552)
    d=torch.nn.functional.normalize(torch.randn(6,10,dtype=torch.float64,generator=g),dim=0)
    x=torch.randn(3,6,dtype=torch.float64,generator=g)
    bank,diagnostic=propose_expanded_bank(x,d,depth=3,swaps_per_atom=5,
        cartesian_per_atom=3,max_variants=30,return_diagnostics=True)
    assert ((bank.atoms != bank.atoms[:,:1]).sum(-1)>=2).any()
    assert bank.unique.all()
    a=d.T[bank.atoms].transpose(-1,-2)
    c=torch.linalg.lstsq(a,x[:,None,:,None]).solution.squeeze(-1)
    torch.testing.assert_close(bank.centers,c,rtol=1e-10,atol=1e-10)
    direct=(x[:,None]-(a@c[...,None]).squeeze(-1)).square().sum(-1)
    torch.testing.assert_close(bank.errors,direct,rtol=1e-10,atol=1e-10)
    searched=diagnostic['errors'];searched[:,0]=torch.inf
    torch.testing.assert_close(bank.errors[:,1:],searched.topk(29,largest=False).values,rtol=1e-10,atol=1e-10)


def test_ordered_dedup_preserves_first_without_merging_permutations():
    a=torch.tensor([[[2,1,3],[0,3,2],[2,1,3],[1,2,3],[0,3,2]]])
    assert _unique_ordered_supports(a,4).tolist()==[[True,True,False,True,False]]


def test_compact_coefficient_kernel_matches_complete_grid_and_falls_back():
    bins=torch.linspace(-3,3,2048,dtype=torch.float64)
    means=torch.tensor([[-4.,-1.27,0.,.329,4.]],dtype=torch.float64)
    width=torch.tensor([[1.,.8,1.2,2.,1.]],dtype=torch.float64)
    ids,logp=_coefficient_kernel(bins,means,width,1e-6,16)
    assert ids.shape[-1]==33
    actual=torch.zeros(1,5,2048,dtype=torch.float64).scatter_add_(-1,ids,logp.exp())
    _,dense=_coefficient_kernel(bins,means,width,1e-6,None)
    torch.testing.assert_close(actual,dense.exp(),atol=1e-12,rtol=1e-10)
    ids,logp=_coefficient_kernel(bins,means,width,.5,16)
    assert ids.shape[-1]==2048
    _,dense=_coefficient_kernel(bins,means,width,.5,None)
    torch.testing.assert_close(logp,dense)
