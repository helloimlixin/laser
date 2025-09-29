from typing import Tuple
import torch
import tensorly as ten
import tltorch as tlt
from functools import lru_cache

# Class for Low Separation Rank Tensor Dot
class LSRTensorDot(torch.nn.Module):
    def __init__(
        self,
        shape: Tuple[int, ...],
        ranks: Tuple[int, ...],
        separation_rank: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        initialize: bool = True):
        """
        Initialize LSRTensorDot module.

        Args:
            shape: tuple of tensor dimensions (number of modes = len(shape))
            ranks: tuple of rank values for each mode
            separation_rank: separation rank of the tensor (number of summation terms in the
                separation rank decomposition)
            dtype: data type of the tensors
            device: device for computation
            initialize: whether to initialize the parameters
        """
        
        super(LSRTensorDot, self).__init__()

        # input validation for critical parameters
        if len(shape) != len(ranks):
            raise ValueError(f"Shape {len(shape)} and ranks {len(ranks)} must have the same length")
        if separation_rank <= 0:
            raise ValueError(f"Separation rank must be positive, got {separation_rank}")
        if separation_rank > len(shape):
            raise ValueError(f"Separation rank {separation_rank} must be less than or equal to the number of modes in the tensor {len(shape)}")
         
        self.dtype = dtype
        self.device = device
        self.shape = shape
        self.ranks = ranks
        self.separation_rank = separation_rank
        self.order = len(shape)  # number of modes in the tensor
        
        self.init_params(initialize)

    def init_params(self, initialize=True):
        # Initialize core tensor as independent standard gaussians
        if not initialize:
            self.core_tensor = torch.nn.parameter.Parameter(torch.zeros(self.ranks, device=self.device))
        else:
            width = 1
            self.core_tensor = torch.nn.parameter.Parameter(width * torch.rand(size=self.ranks, dtype=self.dtype, device=self.device) - width / 2)

        self.factor_matrices = torch.nn.ModuleList()

        # Initialize all factor matrices
        for s in range(self.separation_rank):
            factors_s = torch.nn.ParameterList()

            for k in range(self.order):
                if not initialize:
                    factor_matrix_B = torch.eye(self.shape[k])[:, self.ranks[k]]
                    factors_s.append(torch.nn.parameter.Parameter(factor_matrix_B))
                else:
                    factor_matrix_A = torch.normal(torch.zeros((self.shape[k], self.ranks[k]), dtype=self.dtype, device=self.device),\
                                                   torch.ones((self.shape[k], self.ranks[k]), dtype=self.dtype, device=self.device))

                    # Orthonormalize matrix
                    factor_matrix_B = torch.nn.parameter.Parameter(torch.linalg.qr(factor_matrix_A)[0])
                    factors_s.append(factor_matrix_B)

            self.factor_matrices.append(factors_s)

    # Create a new LSR_tensor using the parameters from the LSR_tensor provided
    @classmethod
    @torch.no_grad()
    def copy(cls, old_tensor, device=None):
        if device is None:
            device = old_tensor.device

        new_tensor = cls(old_tensor.shape, old_tensor.ranks, old_tensor.separation_rank, old_tensor.dtype, device, initialize=False)
        for s in range(old_tensor.separation_rank):
            for k in range(len(old_tensor.ranks)):
                new_tensor.factor_matrices[s][k] = torch.nn.parameter.Parameter(torch.clone(old_tensor.factor_matrices[s][k]).to(device))

        new_tensor.core_tensor = torch.nn.parameter.Parameter(torch.clone(old_tensor.core_tensor).to(device))
        return new_tensor

    # Expand core tensor and factor matrices to full tensor, optionally excluding
    # a given term from the separation rank decomposition
    def expand_to_tensor(self, skip_term=None):
        full_lsr_tensor = 0

        for s, term_s_factors in enumerate(self.factor_matrices):
            if s == skip_term:
                continue

            full_lsr_tensor += ten.tenalg.multi_mode_dot(self.core_tensor, term_s_factors)

        return full_lsr_tensor

    # Expand only one tucker term
    def expand_tucker_term(self, term=0):
        return ten.tenalg.multi_mode_dot(self.core_tensor, self.factor_matrices[term])

    # Regular forward pass
    def forward(self, x):
        return ten.tenalg.inner(x, self.expand_to_tensor(), n_modes=self.order)

    # Absorb all factor matrices and core tensor into the input tensor except for matrix s, k
    # Used during a factor matrix update step of block coordiante descent
    @lru_cache(maxsize=0)
    @torch.no_grad()
    def bcd_factor_update_x(self, s, k, x):
        omega = ten.base.partial_tensor_to_vec(
                ten.base.partial_unfold(x, mode=k) @
                (ten.tenalg.kronecker(self.factor_matrices[s], skip_matrix=k, reverse=True) @
                    ten.base.unfold(self.core_tensor, k).T)
                )
        gamma = ten.tenalg.inner(x, self.expand_to_tensor(skip_term=s), n_modes=self.order)
        x_combined = torch.cat((omega, torch.unsqueeze(gamma, axis=1)), axis=1)
        return x_combined

    # Absorb all factor matrices the input tensor (not the core tensor)
    # Used during a core tensor update step of block coordiante descent
    @lru_cache(maxsize=1)
    @torch.no_grad()
    def bcd_core_update_x(self, x):
        x_vec = torch.unsqueeze(ten.base.partial_tensor_to_vec(x), axis=2)

        kron_factor_sum = 0
        for term_s_factors in self.factor_matrices:
            kron_factor_sum += ten.tenalg.kronecker(term_s_factors, reverse=False).T

        x_combined = kron_factor_sum @ x_vec
        x_combined = torch.squeeze(x_combined, axis=2)
        return x_combined

    # Block coordinate descent core tensor update step 
    def bcd_core_forward(self, x, precombined=False):
        x_combined = self.bcd_core_update_x(x) if not precombined else x
        core_vec = ten.base.tensor_to_vec(self.core_tensor)
        return ten.tenalg.inner(x_combined, core_vec, n_modes=1)

    # Block coordinate descent factor matrix update step 
    def bcd_factor_forward(self, s, k, x, precombined=False):
        x_combined = self.bcd_factor_update_x(s, k, x) if not precombined else x
        factor_expanded = torch.cat((ten.base.tensor_to_vec(self.factor_matrices[s][k]), torch.ones((1), device=self.device)))
        return ten.tenalg.inner(x_combined, factor_expanded, n_modes=1)

    # Orthonormalize the columns of a factor matrix
    @torch.no_grad()
    def orthonorm_factor(self, s, k):
        q, r = torch.linalg.qr(self.factor_matrices[s][k], mode='reduced')
        r_signs = torch.sign(torch.eye(self.ranks[k], device=self.device) * r)
        self.factor_matrices[s][k][:, :self.ranks[k]] = (q @ r_signs)[:, :self.ranks[k]]

