import math
import torch
import torch.nn.functional as F


def _to_luma(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor in [-1,1] to single channel luminance-like signal."""
    if x.shape[1] == 1:
        return x
    # Use simple average to avoid extra allocations; assume inputs already normalized
    return x.mean(dim=1, keepdim=True)


_DCT_BASIS_CACHE = {}
_SOBEL_KERNEL_CACHE = {}


def _get_dct_basis(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, torch.float32, n)
    basis = _DCT_BASIS_CACHE.get(key)
    if basis is None:
        arange = torch.arange(n, device=device, dtype=torch.float32)
        theta = (math.pi / n) * (arange.unsqueeze(1) + 0.5) * arange.unsqueeze(0)
        basis = torch.cos(theta)
        basis *= math.sqrt(2.0 / n)
        basis[:, 0] /= math.sqrt(2.0)
        _DCT_BASIS_CACHE[key] = basis
    return basis.to(dtype=dtype)


def _dct_1d_last_dim(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    n = orig_shape[-1]
    x_flat = x.reshape(-1, n)
    basis = _get_dct_basis(n, x.device, x.dtype)
    y = x_flat @ basis
    return y.view(orig_shape)


def _dct_2d(x: torch.Tensor) -> torch.Tensor:
    # Apply along width then height
    y = _dct_1d_last_dim(x)
    y = _dct_1d_last_dim(y.transpose(-1, -2)).transpose(-1, -2)
    return y


def _get_sobel_kernels(device: torch.device, dtype: torch.dtype):
    """Return cached Sobel kernels on the requested device/dtype."""
    key = (device, dtype)
    kernels = _SOBEL_KERNEL_CACHE.get(key)
    if kernels is None:
        base_x = torch.tensor([[[[-1.0, 0.0, 1.0],
                                 [-2.0, 0.0, 2.0],
                                 [-1.0, 0.0, 1.0]]]])
        base_y = torch.tensor([[[[-1.0, -2.0, -1.0],
                                 [0.0, 0.0, 0.0],
                                 [1.0, 2.0, 1.0]]]])
        kernels = (
            base_x.to(device=device, dtype=dtype),
            base_y.to(device=device, dtype=dtype),
        )
        _SOBEL_KERNEL_CACHE[key] = kernels
    return kernels


def _gradient_magnitude(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return Sobel gradient magnitude for a single-channel tensor."""
    sobel_x, sobel_y = _get_sobel_kernels(x.device, x.dtype)
    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + eps)


def multi_resolution_dct_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    num_levels: int = 3,
    alpha: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute multi-resolution DCT magnitude loss between x and y.

    Args:
        x, y: tensors in shape [B, C, H, W]
        num_levels: number of pyramid levels (>=1)
        alpha: power to apply to magnitudes for dynamic-range compression
        eps: small constant for numerical stability
    """
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")

    cur_x, cur_y = x, y
    losses = []
    for level in range(num_levels):
        x_luma = _to_luma(cur_x)
        y_luma = _to_luma(cur_y)

        dct_x = _dct_2d(x_luma)
        dct_y = _dct_2d(y_luma)

        mag_x = (dct_x.abs() + eps) ** alpha
        mag_y = (dct_y.abs() + eps) ** alpha

        losses.append(F.l1_loss(mag_x, mag_y))

        if level < num_levels - 1:
            cur_x = F.avg_pool2d(cur_x, kernel_size=2, stride=2, ceil_mode=False)
            cur_y = F.avg_pool2d(cur_y, kernel_size=2, stride=2, ceil_mode=False)

    return torch.stack(losses).mean()


def multi_resolution_gradient_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    num_levels: int = 3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute multi-resolution Sobel gradient magnitude difference between x and y.

    Args:
        x, y: tensors in shape [B, C, H, W] in the same range (typically [-1, 1]).
        num_levels: number of pyramid levels (>=1).
        eps: numerical stability added inside the gradient magnitude computation.
    """
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")

    cur_x, cur_y = x, y
    losses = []
    for level in range(num_levels):
        x_luma = _to_luma(cur_x)
        y_luma = _to_luma(cur_y)

        grad_x = _gradient_magnitude(x_luma, eps=eps)
        grad_y = _gradient_magnitude(y_luma, eps=eps)
        losses.append(F.l1_loss(grad_x, grad_y))

        if level < num_levels - 1:
            cur_x = F.avg_pool2d(cur_x, kernel_size=2, stride=2, ceil_mode=False)
            cur_y = F.avg_pool2d(cur_y, kernel_size=2, stride=2, ceil_mode=False)

    return torch.stack(losses).mean()
