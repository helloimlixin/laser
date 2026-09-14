import torch
from src.compact_rq_fitting import coefficient_normal_equations, solve_regularized_levels


def test_final_reconstruction_solve_matches_dense_ridge_with_repeated_and_zero_codes():
    torch.manual_seed(127)
    dictionary=torch.randn(5,3,dtype=torch.float64)
    codes=torch.tensor([[0,1,1,6],[3,4,0,3],[2,5,6,1],[0,0,0,0]])
    targets=torch.randn(4,5,dtype=torch.float64)
    normal,rhs=coefficient_normal_equations(dictionary,codes,targets,2)
    design=torch.zeros(4,5,6,dtype=torch.float64)
    for row in range(4):
        for code in codes[row]:
            if code:
                design[row,:,code-1]+=dictionary[:,(code-1)//2]
    design=design.flatten(0,1)
    torch.testing.assert_close(normal.to_dense(),design.T@design)
    torch.testing.assert_close(rhs,design.T@targets.flatten())
    current=torch.randn(3,2,dtype=torch.float64)
    prior=torch.randn_like(current)
    ridge=torch.rand(6,dtype=torch.float64)+.1
    expected=torch.linalg.solve(design.T@design+ridge.diag(),design.T@targets.flatten()+ridge*prior.flatten())
    actual=solve_regularized_levels(normal,rhs,current,prior,ridge)
    torch.testing.assert_close(actual.flatten(),expected,atol=1e-8,rtol=1e-8)
