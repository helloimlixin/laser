"""Distribution tests for whole-pair temperature, including unequal conditionals."""
import pytest
import torch

from src.compound_pair_sampling import draw_joint_pair


@pytest.mark.parametrize('temperature',[.8,1.,1.2])
@pytest.mark.parametrize('proposal_width',[1,8])
def test_samples_match_explicit_joint_softmax(temperature,proposal_width):
    torch.manual_seed(8150)
    # One atom has diffuse coefficients; the other has a concentrated row.
    # Separate temperatures give the wrong atom marginal in this example.
    pa = torch.tensor([.45,.55])
    pc = torch.tensor([[.5,.5],[.99,.01]])
    count = 80000
    atoms,coefficients,stats = draw_joint_pair(pa.log().expand(count,-1),
        lambda rows,ids:pc[ids].log(),temperature=temperature,proposal_width=proposal_width)
    frequencies = torch.bincount(atoms*2+coefficients,minlength=4).float()/count
    target = (pa[:,None]*pc).pow(1/temperature).flatten()
    target /= target.sum()
    torch.testing.assert_close(frequencies,target,rtol=0.,atol=.006)
    assert stats['accepted'] == count
    if temperature == 1.:
        assert stats['proposals'] == count


def test_joint_normalization_changes_atom_marginal():
    torch.manual_seed(6150)
    pa = torch.tensor([.5,.5])
    pc = torch.tensor([[.5,.5],[.999,.001]])
    atoms,_,_ = draw_joint_pair(pa.log().expand(50000,-1),
        lambda rows,ids:pc[ids].log(),temperature=.5)
    # Independently tempering p(a) leaves its marginal exactly 50/50.
    assert .65 < float(atoms.float().mean()) < .68


def test_exhausted_normalization_raises_instead_of_returning_biased_tokens():
    with pytest.raises(RuntimeError,match='no biased fallback'):
        draw_joint_pair(torch.zeros(2,2),lambda rows,ids:torch.zeros(len(rows),2),max_rounds=0)


@pytest.mark.parametrize('temperature',[.8,1.,1.2])
@pytest.mark.parametrize('proposal_width',[1,8])
def test_joint_support_floor_matches_explicit_filtered_table(temperature,proposal_width):
    torch.manual_seed(4481)
    pa = torch.tensor([.45,.55])
    pc = torch.tensor([[.5,.5],[.99,.01]])
    ratio,count = .25,80000
    joint = pa[:,None]*pc
    threshold = ratio*pa.max()/pc.shape[-1]
    target = joint.pow(1/temperature)*(joint>=threshold)
    target = target.flatten()/target.sum()
    atoms,coefficients,_ = draw_joint_pair(pa.log().expand(count,-1),
        lambda rows,ids:pc[ids].log(),temperature=temperature,
        proposal_width=proposal_width,support_floor_ratio=ratio)
    frequency = torch.bincount(atoms*2+coefficients,minlength=4).float()/count
    torch.testing.assert_close(frequency,target,rtol=0.,atol=.006)
    assert frequency[3]==0


def test_floor_depends_on_whole_pair_and_handles_empty_conditional_support():
    torch.manual_seed(684)
    pa = torch.tensor([.9,.1])
    pc = torch.tensor([[.05,.95],[.5,.5]])
    # Threshold .2*.9/2=.09 removes both coefficients for atom 1, even
    # though each has conditional probability .5, and the rare c for atom 0.
    atoms,coefficients,_ = draw_joint_pair(pa.log().expand(10000,-1),
        lambda rows,ids:pc[ids].log(),temperature=1.,support_floor_ratio=.2)
    assert torch.equal(atoms,torch.zeros_like(atoms))
    assert torch.equal(coefficients,torch.ones_like(coefficients))


@pytest.mark.parametrize('ratio',[-1.,float('nan'),1.01])
def test_rejects_invalid_support_floor(ratio):
    with pytest.raises(ValueError,match='support floor'):
        draw_joint_pair(torch.zeros(2,2),lambda rows,ids:torch.zeros(len(rows),2),support_floor_ratio=ratio)
