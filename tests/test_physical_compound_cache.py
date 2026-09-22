import numpy as np
import pytest
import torch
from scripts.tools.prepare_physical_compound_cache import physical_coefficients,fit_scalar_lloyd,nearest_centers


def test_unit_conversion_preserves_physical_sparse_reconstruction():
    dictionary=torch.nn.functional.normalize(torch.randn(11,7,generator=torch.Generator().manual_seed(9)),dim=-1)
    atoms=torch.tensor([[[[0,2,5,9],[3,7,2,10]]]])
    normalized=torch.randn(atoms.shape,generator=torch.Generator().manual_seed(11))
    scales=torch.tensor([5.4,2.6,1.7,1.1])
    physical=physical_coefficients(normalized,scales)
    before=(dictionary[atoms]*(normalized*scales)[...,None]).sum(-2)
    after=(dictionary[atoms]*physical[...,None]).sum(-2)
    torch.testing.assert_close(before,after,atol=0,rtol=0)
    assert torch.equal(physical,physical_coefficients(physical,[1,1,1,1]))


def test_physical_bin_means_same_coefficient_at_every_depth():
    centers=torch.tensor([-6.,-1.,0.,2.,5.])
    values=torch.tensor([[2.2,2.2,2.2,2.2],[-.9,-.9,-.9,-.9]])
    decoded,ids=nearest_centers(values,centers)
    assert ids.tolist()==[[3,3,3,3],[1,1,1,1]]
    assert decoded.tolist()==[[2.,2.,2.,2.],[-1.,-1.,-1.,-1.]]


def test_lloyd_centers_are_empirical_cluster_means():
    values=torch.tensor([-9.,-8.,-7.,-1.,0.,1.,7.,8.,9.])
    centers,report=fit_scalar_lloyd(values,num_bins=3,iterations=100)
    decoded,ids=nearest_centers(values,centers)
    for i in range(3):
        torch.testing.assert_close(centers[i],values[ids==i].mean())
    assert abs(report['final_physical_coefficient_mse']-float((decoded-values).double().square().mean()))<1e-8
    errors=[r['physical_coefficient_mse'] for r in report['trace']]
    assert all(b<=a+1e-9 for a,b in zip(errors,errors[1:]))


@pytest.mark.parametrize('scales',[[1,2],[1,0,1,1],[1,float('nan'),1,1]])
def test_invalid_scales_rejected(scales):
    with pytest.raises(ValueError):physical_coefficients(torch.ones(2,4),scales)
