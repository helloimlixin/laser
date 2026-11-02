"""Small-system OMP algebra, with batched FP32 CUDA and portable fallbacks."""
import torch


def _tiny_cuda(matrix):
    return (matrix.device.type == 'cuda' and matrix.dtype == torch.float32
            and matrix.ndim == 3 and matrix.is_contiguous() and 0 < matrix.shape[-1] <= 4)


def solve_lower(matrix, rhs):
    if _tiny_cuda(matrix):
        from src.training.omp_kernels import solve_lower as kernel
        return kernel(matrix, rhs)
    return torch.linalg.solve_triangular(matrix, rhs, upper=False)


def solve_cholesky(matrix, rhs):
    if _tiny_cuda(matrix):
        from src.training.omp_kernels import solve_cholesky as kernel
        return kernel(matrix, rhs)
    return torch.cholesky_solve(rhs, matrix)


def residual_correlation(correlation, coefficients, gram, support):
    if (correlation.device.type == 'cuda' and correlation.dtype == torch.float32
            and correlation.is_contiguous() and coefficients.is_contiguous()
            and gram.is_contiguous() and support.is_contiguous() and coefficients.shape[-1] <= 4):
        from src.training.omp_kernels import residual_correlation as kernel
        return kernel(correlation, coefficients, gram, support)
    return correlation-coefficients.unsqueeze(1).bmm(gram[support]).squeeze(1)


def support_probabilities(residual, available, temperature):
    if (residual.device.type=='cuda' and residual.dtype==torch.float32
            and residual.is_contiguous() and available.is_contiguous()
            and residual.shape[-1]<=16384):
        from src.training.omp_kernels import support_probabilities as kernel
        return kernel(residual,available,temperature)
    scores=residual.square().masked_fill(~available,-torch.inf)
    return ((scores-scores.amax(-1,keepdim=True))/temperature).softmax(-1)


def extend_cholesky(matrix, gram, support, atom):
    previous=support.shape[-1]
    if (gram.device.type=='cuda' and gram.dtype==torch.float32 and gram.is_contiguous()
            and support.is_contiguous() and atom.is_contiguous() and previous<4
            and (matrix is None or matrix.is_contiguous())):
        from src.training.omp_kernels import extend_cholesky as kernel
        return kernel(matrix,gram,support,atom)
    if matrix is None:
        diagonal=gram[atom,atom]
        return diagonal.clamp_min(1.e-7).sqrt()[:,None,None],diagonal<=1.e-7
    cross=gram[support,atom[:,None]].unsqueeze(-1)
    solved=solve_lower(matrix,cross).transpose(1,2)
    diagonal=gram[atom,atom][:,None,None]-solved.square().sum(-1,keepdim=True)
    result=torch.cat((torch.cat((matrix,gram.new_zeros(len(atom),previous,1)),dim=-1),
        torch.cat((solved,diagonal.clamp_min(1.e-7).sqrt()),dim=-1)),dim=-2)
    return result,(diagonal<=1.e-7).reshape(-1)
