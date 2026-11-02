import torch

from src.training.laser_joint_redundancy import nearest_cross_atom_pairs


def test_nearest_pair_uses_signed_physical_vectors_and_excludes_same_atom_and_null():
    dictionary=torch.tensor([[1.,-1.,0.],[0.,0.,2.]])
    atoms=torch.tensor([[0,0,0,1,1,2,2]]*2)
    coefficients=torch.tensor([[0.,-1.,1.,-1.,1.,-1.,1.],
                               [0.,-2.,2.,-2.,2.,-2.,2.]])
    result=nearest_cross_atom_pairs(dictionary,atoms,coefficients,top_atoms=2,chunk_size=1)
    assert result['nearest_code'].tolist()[1:5]==[4,3,2,1]
    assert (result['relative_vector_rmse'][1:5]==0).all()
    assert (atoms[0,result['nearest_code'][1:]]!=atoms[0,1:]).all()
    assert (result['nearest_code'][1:]>0).all()


def test_neighbor_search_matches_brute_force_and_omission_bound():
    torch.manual_seed(26101721)
    dictionary=torch.randn(7,5)
    atoms=torch.cat((torch.zeros(1,dtype=torch.long),torch.arange(5).repeat_interleave(2))).expand(4,-1)
    coefficients=torch.cat((torch.zeros(4,1),torch.tensor([-1.,1.]).repeat(4,5)),1)
    result=nearest_cross_atom_pairs(dictionary,atoms,coefficients,top_atoms=4,chunk_size=3)
    vectors=dictionary.T[atoms]*coefficients[...,None]
    distance=(vectors[:,:,None,:]-vectors[:,None,:,:]).square().sum((0,3))
    distance.masked_fill_(atoms[0,:,None]==atoms[0,None,:],torch.inf)
    distance[:,0]=torch.inf
    expected=distance[1:].argmin(-1)
    assert torch.equal(result['nearest_code'][1:],expected)
    energy=vectors.square().sum((0,2))
    torch.testing.assert_close(result['relative_vector_rmse'][1:],(distance[1:].min(-1).values/energy[1:]).sqrt())
    assert (result['omitted_direction_lower_bound']==1.).all()
    partial=nearest_cross_atom_pairs(dictionary,atoms,coefficients,top_atoms=1)
    exact=result['relative_vector_rmse'][1:]
    certified=partial['relative_vector_rmse'][1:]<=partial['omitted_direction_lower_bound']
    torch.testing.assert_close(partial['relative_vector_rmse'][1:][certified],exact[certified])
